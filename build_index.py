import faiss
import numpy as np


def build_index():
    print("开始构建索引...")

    # 加载合并后的embeddings
    embeddings_path = "/mnt/hdd1/yufei/img2dataset/laion_face_embeddings_merged/img_emb/img_emb_0.npy"
    print(f"加载embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)

    print(f"Embeddings shape: {embeddings.shape}")

    # 确保数据类型是float32
    embeddings = embeddings.astype('float32')

    # 标准化
    print("Normalizing vectors...")
    faiss.normalize_L2(embeddings)

    # 创建索引
    dimension = embeddings.shape[1]  # 512维
    print(f"Creating index for {dimension} dimensions")

    # 使用IVF索引以提高搜索速度
    nlist = 65536  # 聚类中心数量
    print("Creating IVF index...")
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

    # 训练索引
    print("Training index...")
    print("This might take a while...")
    index.train(embeddings)

    # 添加向量到索引
    print("Adding vectors to index...")
    index.add(embeddings)

    # 设置搜索参数
    index.nprobe = 256  # 搜索时检查的聚类数量

    # 保存索引
    output_path = "/mnt/hdd1/yufei/img2dataset/laion_face_index/image.index"
    print(f"Saving index to {output_path}")
    faiss.write_index(index, output_path)

    # 保存一些索引信息
    index_info = {
        "total_vectors": int(embeddings.shape[0]),
        "dimension": int(dimension),
        "nlist": int(nlist),
        "nprobe": int(256)
    }

    import json
    with open("/mnt/hdd1/yufei/img2dataset/laion_face_index/index_info.json", "w") as f:
        json.dump(index_info, f, indent=2)

    print("索引构建完成！")
    print(f"Total vectors indexed: {embeddings.shape[0]}")


if __name__ == "__main__":
    try:
        build_index()
    except Exception as e:
        import traceback

        print(f"Error occurred: {str(e)}")
        traceback.print_exc()