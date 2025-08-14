def find_similar_images(input_image_path):
    try:
        # 设置目录路径
        index_dir = "/mnt/hdd1/yufei/img2dataset/laion_face_index"

        # 先检查索引文件
        print(f"Looking for index files in {index_dir}...")
        if os.path.exists(index_dir):
            print("Index directory exists")
            files = os.listdir(index_dir)
            print("Found files:", files)
        else:
            print("Index directory does not exist!")
            return False

        video_dir = input_image_path.split("/")[-3]
        img_name = os.path.basename(input_image_path)
        output_dir = f"/mnt/hdd1/yufei/img2dataset/similar_results/{video_dir}/{img_name.replace('.jpg', '')}"
        os.makedirs(output_dir, exist_ok=True)

        # 尝试列出所有.index文件
        index_files = []
        for root, dirs, files in os.walk(index_dir):
            for file in files:
                if file.endswith('.index') or file.endswith('.faiss'):
                    index_files.append(os.path.join(root, file))
        print("Found index files:", index_files)

        if not index_files:
            print("No index files found!")
            return False

        # 使用找到的第一个索引文件
        index_path = index_files[0]
        print(f"Using index file: {index_path}")

        # 加载CLIP模型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-L/14", device=device)

        # 处理输入图片
        image = preprocess(Image.open(input_image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)

        # 读取索引
        index = faiss.read_index(index_path)

        # 搜索相似图片
        D, I = index.search(image_features.cpu().numpy(), 10)

        # 保存结果
        with open(os.path.join(output_dir, "results.txt"), "w") as f:
            for i, (dist, idx) in enumerate(zip(D[0], I[0])):
                f.write(f"Similar {i}: index {idx}, distance {dist}\\n")

        print(f"Successfully processed {input_image_path}")
        return True
    except Exception as e:
        print(f"Error processing {input_image_path}: {e}")
        return False