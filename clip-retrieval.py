import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import glob
import tarfile
import numpy as np
import h5py
from tqdm import tqdm
import io
import traceback
from multiprocessing import Pool, cpu_count
import multiprocessing

multiprocessing.set_start_method('spawn', force=True)
import time
import gc


class CLIPImageRetrieval:
    def __init__(self, model_name="openai/clip-vit-base-patch32", feature_dir="clip_features", batch_size=32):
        self.feature_dir = feature_dir
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = 3
        os.makedirs(feature_dir, exist_ok=True)

    def init_model(self):
        """初始化模型（在每个进程中调用）"""
        if not hasattr(self, 'model'):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Initializing model on {self.device}")
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)

            # 设置较小的批处理大小以减少内存使用
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def validate_image(self, image):
        """验证图片是否有效且格式正确"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')

            min_size = 224  # CLIP需要的最小尺寸
            # 计算缩放比例，保持长宽比
            ratio = min_size / min(image.size[0], image.size[1])
            new_size = tuple([int(x * ratio) for x in image.size])
            image = image.resize(new_size, Image.Resampling.LANCZOS)

            return image
        except Exception as e:
            print(f"Error validating image: {str(e)}")
            return None

    def extract_features_batch(self, images):
        """批量提取特征"""
        try:
            self.init_model()

            # 批量处理图像
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                features = features.cpu().numpy()

            # 清理内存
            del inputs
            if self.device == "cuda":
                torch.cuda.empty_cache()

            return features

        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            return None

    def extract_features(self, image_data, image_name=""):
        """从单个图片提取特征"""
        try:
            if isinstance(image_data, str):
                image = Image.open(image_data)
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            else:
                raise ValueError("Unsupported image_data type")

            image = self.validate_image(image)
            if image is None:
                return None

            features = self.extract_features_batch([image])
            return features[0] if features is not None else None

        except Exception as e:
            print(f"Error processing image {image_name}: {str(e)}")
            return None

    def process_tar_chunk(self, tar_path):
        """处理单个tar文件，使用批处理来提高效率"""
        chunk_id = os.path.basename(tar_path).split('.')[0]

        for retry in range(self.max_retries):
            try:
                self.init_model()

                features = []
                image_paths = []
                current_batch = []
                current_batch_paths = []

                # 处理tar文件
                with tarfile.open(tar_path, 'r') as tar:
                    # 获取所有图片文件
                    members = [m for m in tar.getmembers()
                               if m.name.lower().endswith(('.jpg', '.jpeg', '.png'))]

                    # 批量处理图片
                    for member in members:
                        try:
                            f = tar.extractfile(member)
                            if f is None:
                                continue

                            image_data = f.read()
                            try:
                                image = Image.open(io.BytesIO(image_data))
                                image = self.validate_image(image)
                                if image is not None:
                                    current_batch.append(image)
                                    current_batch_paths.append((member.name.strip(), tar_path))

                                    if len(current_batch) >= self.batch_size:
                                        batch_features = self.extract_features_batch(current_batch)
                                        if batch_features is not None:
                                            features.extend(batch_features)
                                            image_paths.extend(current_batch_paths)
                                        current_batch = []
                                        current_batch_paths = []

                            except Exception as e:
                                print(f"Error processing image {member.name}: {str(e)}")
                                continue

                        except Exception as e:
                            print(f"Error extracting {member.name} from tar: {str(e)}")
                            continue

                # 处理最后一个不完整的批次
                if current_batch:
                    try:
                        batch_features = self.extract_features_batch(current_batch)
                        if batch_features is not None:
                            features.extend(batch_features)
                            image_paths.extend(current_batch_paths)
                    except Exception as e:
                        print(f"Error processing final batch: {str(e)}")

                # 保存特征
                if features:
                    try:
                        features = np.vstack(features)
                        chunk_file = os.path.join(self.feature_dir, f'features_chunk_{chunk_id}.h5')
                        with h5py.File(chunk_file, 'w') as f:
                            f.create_dataset('features', data=features)
                            dt = h5py.special_dtype(vlen=str)
                            image_names = [path[0] for path in image_paths]
                            tar_paths = [str(path[1]) for path in image_paths]
                            f.create_dataset('image_names', data=image_names, dtype=dt)
                            f.create_dataset('tar_paths', data=tar_paths, dtype=dt)

                        return chunk_id, len(features)
                    except Exception as e:
                        print(f"Error saving features: {str(e)}")
                        if retry < self.max_retries - 1:
                            continue

                return chunk_id, 0

            except Exception as e:
                print(f"Error processing chunk {tar_path} (attempt {retry + 1}/{self.max_retries}): {str(e)}")
                if retry < self.max_retries - 1:
                    time.sleep(1)
                    continue
                return chunk_id, 0

            finally:
                gc.collect()
                if hasattr(self, 'device') and self.device == "cuda":
                    torch.cuda.empty_cache()

    def find_similar_images(self, query_image_path, output_dir, num_similar=5):
        """查找并保存相似图片"""
        # 初始化模型
        self.init_model()

        # 提取查询图片的特征
        query_features = self.extract_features(query_image_path)
        if query_features is None:
            return

        index_file = os.path.join(self.feature_dir, 'index.txt')
        if not os.path.exists(index_file):
            raise ValueError("Index file not found. Please run preprocess_dataset first.")

        all_similarities = []

        with open(index_file, 'r') as f:
            for chunk_file in tqdm(f.readlines(), desc="Searching chunks"):
                chunk_file = chunk_file.strip()
                with h5py.File(chunk_file, 'r') as f_h5:
                    features = f_h5['features'][:]
                    image_names = f_h5['image_names'][:]
                    tar_paths = f_h5['tar_paths'][:]

                    similarities = np.dot(features, query_features.T).flatten()

                    for sim, img_name, tar_path in zip(similarities, image_names, tar_paths):
                        all_similarities.append((sim, img_name, tar_path))

        all_similarities.sort(key=lambda x: x[0], reverse=True)
        top_similar = all_similarities[:num_similar]

        os.makedirs(output_dir, exist_ok=True)

        for i, (similarity, image_name, tar_path) in enumerate(top_similar):
            with tarfile.open(tar_path, 'r') as tar:
                try:
                    if isinstance(image_name, bytes):
                        member_name = image_name.decode('utf-8')
                    else:
                        member_name = image_name

                    member = tar.getmember(member_name)
                    f = tar.extractfile(member)
                    if f is not None:
                        output_path = os.path.join(output_dir, f"similar_{i + 1}.jpg")
                        image_data = f.read()
                        with open(output_path, 'wb') as out:
                            out.write(image_data)
                        print(f"Saved similar image {i + 1}, similarity: {similarity:.4f}")
                except Exception as e:
                    print(f"Error extracting {image_name}: {str(e)}")

    def preprocess_dataset(self, data_dir, num_processes=6):
        """预处理数据集，使用多进程和分块存储"""
        print(f"Using {num_processes} processes")

        tar_files = []
        for split_dir in sorted(glob.glob(os.path.join(data_dir, "split_*"))):
            tar_files.extend(sorted(glob.glob(os.path.join(split_dir, "*.tar"))))

        print(f"Found {len(tar_files)} tar files")

        with Pool(num_processes) as pool:
            results = list(tqdm(
                pool.imap_unordered(self.process_tar_chunk, tar_files),
                total=len(tar_files),
                desc="Processing tar files"
            ))

        successful_chunks = [r for r in results if r[1] > 0]
        total_processed = sum(count for _, count in successful_chunks)
        print(f"\nProcessing complete:")
        print(f"Successfully processed chunks: {len(successful_chunks)}/{len(tar_files)}")
        print(f"Total images processed: {total_processed}")

        self._create_index()

    def _create_index(self):
        index_file = os.path.join(self.feature_dir, 'index.txt')
        with open(index_file, 'w') as f:
            for chunk_file in glob.glob(os.path.join(self.feature_dir, 'features_chunk_*.h5')):
                f.write(f"{chunk_file}\n")


def main():
    # 1) 设置路径
    dataset_dir = "/mnt/hdd1/yufei/img2dataset/laion_face_data"
    feature_dir = "clip_features"
    tmp_picture_root = "/mnt/hdd1/yufei/img2dataset/tmp_picture"

    # 2) 初始化检索系统
    retrieval = CLIPImageRetrieval(feature_dir=feature_dir)

    # 3) 如果尚未创建 index.txt，则先进行预处理
    index_file = os.path.join(feature_dir, 'index.txt')
    if not os.path.exists(index_file):
        print("Index file not found. Starting preprocessing...")
        retrieval.preprocess_dataset(dataset_dir, num_processes=2)

    # 4) 从 tmp_picture 目录下，依次处理每个子目录
    for video_name in os.listdir(tmp_picture_root):
        video_dir = os.path.join(tmp_picture_root, video_name)
        if not os.path.isdir(video_dir):
            continue  # 跳过非文件夹

        # ============== 第一人 HQ_face ==============
        hq_face_dir = os.path.join(video_dir, "HQ_face")
        if os.path.isdir(hq_face_dir):
            # 如果 local similar_image >=5，跳过本文件夹
            similar_dir_1 = os.path.join(hq_face_dir, "similar_image")
            if os.path.isdir(similar_dir_1):
                sim_files_1 = [
                    f for f in os.listdir(similar_dir_1)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]
                if len(sim_files_1) >= 5:
                    print(f"[Skip] {video_name}/HQ_face: already has >=5 images in similar_image.")
                else:
                    # 需要检索
                    for img_file in os.listdir(hq_face_dir):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            query_image_path = os.path.join(hq_face_dir, img_file)
                            output_dir = os.path.join(hq_face_dir, "similar_image")
                            print(f"\n[INFO] Finding similar images for: {query_image_path}")
                            retrieval.find_similar_images(query_image_path, output_dir)
            else:
                # 如果 similar_image 目录不存在则检索
                for img_file in os.listdir(hq_face_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        query_image_path = os.path.join(hq_face_dir, img_file)
                        output_dir = os.path.join(hq_face_dir, "similar_image")
                        print(f"\n[INFO] Finding similar images for: {query_image_path}")
                        retrieval.find_similar_images(query_image_path, output_dir)

        # ============== 第二人 HQ_face2 ==============
        hq_face_dir2 = os.path.join(video_dir, "HQ_face2")
        if os.path.isdir(hq_face_dir2):
            # 如果 local similar_image2 >=5，跳过本文件夹
            similar_dir_2 = os.path.join(hq_face_dir2, "similar_image2")
            if os.path.isdir(similar_dir_2):
                sim_files_2 = [
                    f for f in os.listdir(similar_dir_2)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]
                if len(sim_files_2) >= 5:
                    print(f"[Skip] {video_name}/HQ_face2: already has >=5 images in similar_image2.")
                else:
                    # 需要检索
                    for img_file in os.listdir(hq_face_dir2):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            query_image_path = os.path.join(hq_face_dir2, img_file)
                            output_dir2 = os.path.join(hq_face_dir2, "similar_image2")
                            print(f"\n[INFO] Finding similar images for: {query_image_path} (HQ_face2)")
                            retrieval.find_similar_images(query_image_path, output_dir2)
            else:
                # 如果 similar_image2 目录不存在则检索
                for img_file in os.listdir(hq_face_dir2):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        query_image_path = os.path.join(hq_face_dir2, img_file)
                        output_dir2 = os.path.join(hq_face_dir2, "similar_image2")
                        print(f"\n[INFO] Finding similar images for: {query_image_path} (HQ_face2)")
                        retrieval.find_similar_images(query_image_path, output_dir2)

    print("\n[INFO] All done.")


if __name__ == "__main__":
    main()
