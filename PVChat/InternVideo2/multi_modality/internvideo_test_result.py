import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModel

import decord
decord.bridge.set_bridge("torch")


def parse_args():
    parser = argparse.ArgumentParser(description="InternVideo testing script")

    # Required paths
    parser.add_argument("--model_path", type=str,
                        default="/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/InternVideo2_chat_8B_HD",
                        help="Path to the model")
    parser.add_argument("--input_json", type=str,
                        default="/root/autodl-tmp/yufei/Qwen2.5/data_result/Ch/input/internvideo_Ch-1person-kong.json",
                        help="Input JSON file with test cases")
    parser.add_argument("--output_json", type=str,
                        default="/root/autodl-tmp/yufei/Qwen2.5/data_result/Ch/input/internvideo_Ch_final.json",
                        help="Output JSON file to save results")

    # Reference videos (up to 3)
    parser.add_argument("--ref_video1", type=str,
                        default="/root/autodl-tmp/yufei/DeepFaceLab/all_kind_of_data/Friends/Ch/Chandler9.mp4",
                        help="First reference video path")
    parser.add_argument("--ref_prompt1", type=str,
                        default="This is video about <Ch>",
                        help="First reference prompt")

    parser.add_argument("--ref_video2", type=str,
                        default="",
                        help="Second reference video path (optional)")
    parser.add_argument("--ref_prompt2", type=str,
                        default="",
                        help="")

    parser.add_argument("--ref_video3", type=str,
                        default="",
                        help="Third reference video path (optional)")
    parser.add_argument("--ref_prompt3", type=str,
                        default="",
                        help="Third reference prompt (optional)")

    # Video processing parameters
    parser.add_argument("--num_segments", type=int, default=8,
                        help="Number of frames to sample from video")
    parser.add_argument("--resolution", type=int, default=224,
                        help="Resolution for video frames")
    parser.add_argument("--hd_num", type=int, default=6,
                        help="HD number for video processing")

    # 额外可调参数
    parser.add_argument("--videos_per_chunk", type=int, default=5,
                        help="每次处理多少个视频后重启模型")
    args = parser.parse_args()
    return args


def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def HD_transform_padding(frames, image_size=224, hd_num=6):
    def _padding_224(frames):
        _, _, H, W = frames.shape
        tar = int(np.ceil(H / 224) * 224)
        top_padding = (tar - H) // 2
        bottom_padding = tar - H - top_padding
        left_padding = 0
        right_padding = 0

        padded_frames = F.pad(
            frames,
            pad=[left_padding, right_padding, top_padding, bottom_padding],
            mode='constant', value=255
        )
        return padded_frames

    _, _, H, W = frames.shape
    trans = False
    if W < H:
        frames = frames.flip(-2, -1)
        trans = True
        width, height = H, W
    else:
        width, height = W, H

    ratio = width / height
    scale = 1
    while scale * np.ceil(scale / ratio) <= hd_num:
        scale += 1
    scale -= 1
    new_w = int(scale * image_size)
    new_h = int(new_w / ratio)

    resized_frames = F.interpolate(
        frames, size=(new_h, new_w),
        mode='bicubic',
        align_corners=False
    )
    padded_frames = _padding_224(resized_frames)

    if trans:
        padded_frames = padded_frames.flip(-2, -1)

    return padded_frames


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            # 如果想再基于其他规则筛选，也可在这里加判断
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def HD_transform_no_padding(frames, image_size=224, hd_num=6, fix_ratio=(2, 1)):
    min_num = 1
    max_num = hd_num
    _, _, orig_height, orig_width = frames.shape
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    if fix_ratio:
        target_aspect_ratio = fix_ratio
    else:
        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]

    resized_frame = F.interpolate(
        frames, size=(target_height, target_width),
        mode='bicubic', align_corners=False
    )
    return resized_frame


def load_video(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=6, padding=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Normalize(mean, std)
    ])

    frames = vr.get_batch(frame_indices)
    frames = frames.permute(0, 3, 1, 2)

    if padding:
        frames = HD_transform_padding(frames.float(), image_size=resolution, hd_num=hd_num)
    else:
        frames = HD_transform_no_padding(frames.float(), image_size=resolution, hd_num=hd_num)

    frames = transform(frames)
    T_, C, H, W = frames.shape

    sub_img = frames.reshape(
        1, T_, 3, H // resolution, resolution, W // resolution, resolution
    ).permute(0, 3, 5, 1, 2, 4, 6).reshape(-1, T_, 3, resolution, resolution).contiguous()

    glb_img = F.interpolate(
        frames.float(), size=(resolution, resolution), mode='bicubic', align_corners=False
    ).to(sub_img.dtype).unsqueeze(0)

    frames = torch.cat([sub_img, glb_img]).unsqueeze(0)

    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return frames, msg
    else:
        return frames


def load_model_and_references(args):
    """
    加载模型、tokenizer，并把参考视频和参考文字打入模型，返回 (model, tokenizer, chat_history)。
    每调用一次该函数，就会重新加载模型并重新初始化 chat_history。
    """
    print("===== Loading tokenizer and model... =====")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False
    )
    if torch.cuda.is_available():
        model = AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).cuda()
    else:
        model = AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    # 初始化一个空的 chat_history
    chat_history = []

    # 依次打入参考视频
    def process_reference(ref_video, ref_prompt):
        if ref_video and ref_prompt:
            print(f"Processing reference video: {ref_video}")
            ref_video_tensor = load_video(
                ref_video,
                num_segments=args.num_segments,
                resolution=args.resolution,
                hd_num=args.hd_num
            )
            ref_video_tensor = ref_video_tensor.to(model.device)

            response, new_chat_history = model.chat(
                tokenizer,
                '',
                ref_prompt,
                media_type='video',
                media_tensor=ref_video_tensor,
                chat_history=chat_history,
                return_history=True,
                generation_config={'do_sample': False}
            )
            # 更新外部的 chat_history
            return new_chat_history
        else:
            return chat_history

    # 把参考视频一次打进去，得到最后的 chat_history
    chat_history = process_reference(args.ref_video1, args.ref_prompt1)
    chat_history = process_reference(args.ref_video2, args.ref_prompt2)
    chat_history = process_reference(args.ref_video3, args.ref_prompt3)

    return model, tokenizer, chat_history


def unload_model(model, tokenizer=None, chat_history=None):
    """
    卸载模型，释放显存
    """
    del model
    if tokenizer is not None:
        del tokenizer
    if chat_history is not None:
        del chat_history
    torch.cuda.empty_cache()


def main():
    args = parse_args()

    # 加载测试文件
    print(f"Loading test cases from {args.input_json}...")
    with open(args.input_json, 'r') as f:
        test_data = json.load(f)

    # 复制一份数据作为最终输出
    results_data = test_data.copy()
    # 结构假设为 results_data["results"] = [ { "video_path":..., "qa_pairs": [...] }, ... ]

    # 准备一次性处理多少个视频
    num_videos_per_chunk = args.videos_per_chunk

    # 我们需要一个计数器，每处理完 num_videos_per_chunk 个视频后就重启模型
    video_counter = 0
    model = None
    tokenizer = None
    chat_history = None

    total_videos = len(results_data["results"])
    print(f"Total videos to process: {total_videos}")

    # 遍历每个视频
    for i, result in enumerate(tqdm(results_data["results"], desc="Processing Videos")):
        video_path = result["video_path"]

        # 判断该视频是否已经处理完
        all_answered = True
        for qa_pair in result.get("qa_pairs", []):
            if "generated_answer" not in qa_pair:
                all_answered = False
                break
        if all_answered:
            # 如果全部回答已经存在，就跳过
            print(f"  [Skip] Video {video_path} has been fully processed.")
            continue

        # 如果当前还没有模型，或者已经处理了num_videos_per_chunk个，就重新加载模型
        if model is None or video_counter == 0:
            # 如果 model 不为 None，先卸载
            if model is not None:
                unload_model(model, tokenizer, chat_history)
            # 重新加载
            model, tokenizer, chat_history = load_model_and_references(args)

        print(f"Processing video: {video_path}")

        # 推理阶段
        try:
            # 加载视频张量
            video_tensor = load_video(
                video_path,
                num_segments=args.num_segments,
                resolution=args.resolution,
                hd_num=args.hd_num
            )
            video_tensor = video_tensor.to(model.device)

            # 遍历此视频下的每个问题
            for j, qa_pair in enumerate(result["qa_pairs"]):
                if "generated_answer" in qa_pair:
                    continue  # 已有答案就跳过

                question = qa_pair["question"]
                print(f"  Question: {question}")

                try:
                    response, new_chat_history = model.chat(
                        tokenizer,
                        '',
                        question,
                        media_type='video',
                        media_tensor=video_tensor,
                        chat_history=chat_history,
                        return_history=True,
                        generation_config={'do_sample': False}
                    )
                except RuntimeError as e:
                    print(f"  Encountered RuntimeError: {e}")
                    # 先写回已有结果，然后抛出异常
                    with open(args.output_json, 'w') as f:
                        json.dump(results_data, f, indent=2)
                    raise e

                # 保存答案
                results_data["results"][i]["qa_pairs"][j]["generated_answer"] = response
                print(f"  Response: {response}")

                # 更新 chat_history
                chat_history = new_chat_history

        finally:
            # 无论成功或失败，都释放视频相关的显存
            if 'video_tensor' in locals():
                del video_tensor
            torch.cuda.empty_cache()

        # 每处理完一个视频就保存局部结果（防止意外终止）
        with open(args.output_json, 'w') as f:
            json.dump(results_data, f, indent=2)

        # 视频计数 +1
        video_counter += 1
        # 如果达到阈值，就重置计数器，使得下一个视频会触发重载模型
        if video_counter == num_videos_per_chunk:
            video_counter = 0

    # 循环结束后，如果还存在模型也可以卸载
    if model is not None:
        unload_model(model, tokenizer, chat_history)

    print("All test cases done! Results saved to:", args.output_json)


if __name__ == "__main__":
    main()
