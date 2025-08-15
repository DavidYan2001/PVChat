
import sys
import os
import numpy as np
from PIL import Image
import torch
import clip
import faiss

def find_similar_images(input_image_path):
   try:
       # Set the directory path
       index_dir = "/mnt/hdd1/yufei/img2dataset/laion_face_index"
       video_dir = input_image_path.split("/")[-3]
       img_name = os.path.basename(input_image_path)
       output_dir = f"/mnt/hdd1/yufei/img2dataset/similar_results/{video_dir}/{img_name.replace('.jpg', '')}"
       os.makedirs(output_dir, exist_ok=True)

       # Load the CLIP model
       device = "cuda" if torch.cuda.is_available() else "cpu"
       model, preprocess = clip.load("ViT-L/14", device=device)

       # Process the input image
       image = preprocess(Image.open(input_image_path)).unsqueeze(0).to(device)
       with torch.no_grad():
           image_features = model.encode_image(image)

       # Read the index
       index = faiss.read_index(os.path.join(index_dir, "face_index.faiss"))

       # Search for similar pictures
       D, I = index.search(image_features.cpu().numpy(), 10)

       # Save result
       with open(os.path.join(output_dir, "results.txt"), "w") as f:
           for i, (dist, idx) in enumerate(zip(D[0], I[0])):
               f.write(f"Similar {i}: index {idx}, distance {dist}\n")

       print(f"Successfully processed {input_image_path}")
       return True
   except Exception as e:
       print(f"Error processing {input_image_path}: {e}")
       return False

if __name__ == "__main__":
   if len(sys.argv) > 1:
       find_similar_images(sys.argv[1])
