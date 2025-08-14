import os
import shutil
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import torch
from torchvision import transforms
from sklearn.cluster import DBSCAN
import numpy as np

# Path configuration
aligned_dir = "/root/autodl-tmp/yufei/DeepFaceLab/aligned"
output_base_dir = "/root/autodl-tmp/yufei/DeepFaceLab/aligned_grouped"

os.makedirs(output_base_dir, exist_ok=True)

# Initialize FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Extract embeddings for all faces
embeddings = []
file_paths = []

for file_name in os.listdir(aligned_dir):
    file_path = os.path.join(aligned_dir, file_name)
    try:
        img = Image.open(file_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        embedding = model(img_tensor).detach().numpy().flatten()
        embeddings.append(embedding)
        file_paths.append(file_path)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Convert to NumPy array
embeddings = np.array(embeddings)

# Use DBSCAN clustering
clustering = DBSCAN(eps=0.5, min_samples=5, metric='cosine').fit(embeddings)
labels = clustering.labels_

# Store face classifications
for label, file_path in zip(labels, file_paths):
    if label == -1:  # Noise data (uncategorizable images)
        continue
    output_dir = os.path.join(output_base_dir, f"group_{label}")
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(file_path, os.path.join(output_dir, os.path.basename(file_path)))

print("Face grouping completed!")
