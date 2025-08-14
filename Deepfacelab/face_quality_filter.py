import os
import cv2
import numpy as np
import mediapipe as mp  #
import shutil
from tqdm import tqdm


class FaceQualityAnalyzer:
    def __init__(self):
        # Initialize MediaPipe face detection
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def calculate_eye_aspect_ratio(self, landmarks, eye_indices):
        """Calculate Eye Aspect Ratio (EAR)"""
        points = []
        for idx in eye_indices:
            points.append([landmarks[idx].x, landmarks[idx].y])
        points = np.array(points)

        v1 = np.linalg.norm(points[1] - points[5])  # Vertical distance 1
        v2 = np.linalg.norm(points[2] - points[4])  # Vertical distance 2
        h = np.linalg.norm(points[0] - points[3])   # Horizontal distance

        ear = (v1 + v2) / (2.0 * h)
        return ear

    def check_eyes_open(self, landmarks):
        """Detect if eyes are open"""
        # Left and right eye keypoints
        LEFT_EYE = [362, 385, 387, 263, 373, 380]
        RIGHT_EYE = [33, 160, 158, 133, 153, 144]

        # Calculate EAR for left and right eyes
        left_ear = self.calculate_eye_aspect_ratio(landmarks, LEFT_EYE)
        right_ear = self.calculate_eye_aspect_ratio(landmarks, RIGHT_EYE)

        # If EAR of any eye is below 0.25 (threshold), consider eyes closed
        if left_ear < 0.25 or right_ear < 0.25:
            return False  # Eyes closed

        return True  # Eyes open

    def is_frontal_face(self, landmarks):
        """Use facial keypoints to determine if it's a frontal face"""
        LEFT_EYE = [33, 133]
        RIGHT_EYE = [362, 263]
        NOSE_TIP = [1]

        def get_point(idx):
            lm = landmarks[idx]
            return np.array([lm.x, lm.y])

        left_eye_x = (get_point(LEFT_EYE[0])[0] + get_point(LEFT_EYE[1])[0]) / 2
        right_eye_x = (get_point(RIGHT_EYE[0])[0] + get_point(RIGHT_EYE[1])[0]) / 2
        nose_point = get_point(NOSE_TIP[0])
        eye_center_x = (left_eye_x + right_eye_x) / 2

        nose_offset = abs(nose_point[0] - eye_center_x)
        frontal_score = max(1 - nose_offset * 3, 0)
        return frontal_score

    def calculate_face_quality_score(self, image_path):
        """Calculate face image quality score with strict eye closure detection"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to read image: {image_path}")
                return -1

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.mp_face_mesh.process(img_rgb)

            if not results.multi_face_landmarks:
                print(f"No face landmarks detected: {image_path}")
                return -1

            landmarks = results.multi_face_landmarks[0].landmark

            # 1. Check if eyes are open
            if not self.check_eyes_open(landmarks):
                print(f"Eyes closed: {image_path}")
                return -1

            # 2. Calculate frontal face score
            frontal_score = self.is_frontal_face(landmarks)
            if frontal_score <= 0.3:
                print(f"Not a frontal face: {image_path}")
                return -1

            score = frontal_score * 50  # Frontal face score, max 50 points

            # 3. Image clarity evaluation
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clarity_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            clarity_score = min(clarity_score / 500, 1)  # Standardized clarity score
            score += clarity_score * 25  # max score = 25

            # 4. Key point visibility assessment
            visible_landmarks = 0
            total_landmarks = len(landmarks)
            for lm in landmarks:
                if 0 <= lm.x <= 1 and 0 <= lm.y <= 1:
                    visible_landmarks += 1

            visibility_score = visible_landmarks / total_landmarks
            if visibility_score < 0.8:
                print(f"Not enough visible landmarks: {image_path}")
                return -1
            score += visibility_score * 25  #  max score = 25

            return score

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return -1


def process_face_images(input_folder, output_folder, analyzer, top_n=1):
    """Process the pictures in a certain face folder and output the picture with the highest score"""
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.endswith('_0.jpg')]

    image_scores = []
    print(f"Found {len(image_files)} images to process in {input_folder}")
    for img_file in tqdm(image_files):
        img_path = os.path.join(input_folder, img_file)
        score = analyzer.calculate_face_quality_score(img_path)
        if score > 0:
            image_scores.append((img_file, score))

    image_scores.sort(key=lambda x: x[1], reverse=True)
    top_images = image_scores[:top_n]

    for img_file, score in top_images:
        src_path = os.path.join(input_folder, img_file)
        dst_path = os.path.join(output_folder, f"HQ_{img_file}")
        shutil.copy2(src_path, dst_path)
        print(f"Copied {img_file} with score {score:.2f} to {dst_path}")


def main():
    base_dir = "/root/autodl-tmp/yufei/DeepFaceLab/output"
    analyzer = FaceQualityAnalyzer()

    for video_folder in os.listdir(base_dir):
        video_path = os.path.join(base_dir, video_folder)
        if os.path.isdir(video_path):
            # Check and process human_0 folder
            input_folder_human0 = os.path.join(video_path, "filtered", "human_0")
            if os.path.exists(input_folder_human0) and os.path.isdir(input_folder_human0):
                output_folder_human0 = os.path.join(video_path, "HQ_face")
                print(f"\nProcessing {video_folder} - human_0...")
                process_face_images(input_folder_human0, output_folder_human0, analyzer)

            # Check and process human_1 folder (if exists)
            input_folder_human1 = os.path.join(video_path, "filtered", "human_1")
            if os.path.exists(input_folder_human1) and os.path.isdir(input_folder_human1):
                output_folder_human1 = os.path.join(video_path, "HQ_face2")
                print(f"\nProcessing {video_folder} - human_1...")
                process_face_images(input_folder_human1, output_folder_human1, analyzer)


if __name__ == "__main__":
    main()
