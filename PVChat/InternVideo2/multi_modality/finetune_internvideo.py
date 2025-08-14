import pdb
import random  # Used for random.choice
import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import json
from pathlib import Path
import logging
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.utils.rnn as rnn_utils
from decord import VideoReader, cpu
from PIL import Image
import numpy as np
import decord
import warnings
from decord import VideoReader, cpu
from random import shuffle
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import PILToTensor
from torchvision import transforms
from transformers import AutoConfig
from torchvision.transforms.functional import InterpolationMode

decord.bridge.set_bridge("torch")
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'
warnings.filterwarnings("ignore", message="Keyword arguments {'add_special_tokens': False} not recognized.")
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.")
logger = logging.getLogger(__name__)

IMG_TOKEN = "[<IMG_PLH>]"
VID_TOKEN = "[<VID_PLH>]"

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_UNK_TOKEN = "<unk>"

DEFAULT_IMAGE_TOKEN = "[IMAGETOKEN]"
DEFAULT_VIDEO_TOKEN = "[VIDEOTOKEN]"

DEFAULT_IMG_PLACEHOLDER = "[<IMG_PLH>]"
DEFAULT_VID_PLACEHOLDER = "[<VID_PLH>]"
# Set HF_TOKEN environment variable
os.environ["HF_TOKEN"] =  os.environ['HF_TOKEN']


def load_video(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=4, padding=False):
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
    # print(frames.shape)
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
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return frames, msg
    else:
        return frames


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
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def HD_transform_no_padding(frames, image_size=224, hd_num=6, fix_ratio=(2, 1)):
    min_num = 1
    max_num = hd_num
    _, _, orig_height, orig_width = frames.shape
    aspect_ratio = orig_width / orig_height

    # calculate the existing video aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    if fix_ratio:
        target_aspect_ratio = fix_ratio
    else:
        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the frames
    resized_frame = F.interpolate(
        frames, size=(target_height, target_width),
        mode='bicubic', align_corners=False
    )
    return resized_frame


class PersonalizedVideoDataset(Dataset):
    def __init__(
            self,
            json_path: str,  # /root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/<yufei>.json
            tokenizer,
            device,
            split='train'  # 可以添加train/val划分
    ):
        # 加载json文件
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.videos = data['videos']
        self.sks_name = Path(json_path).stem
        # 获取sks名称 (从json路径中提取)
        #  self.sks_name = args.sks_name
        self.split = split
        self.tokenizer = tokenizer

        self.all_qa_pairs = self.flatten_qa_pairs(self.videos)
        # 如果需要划分训练集和验证集
        # print(self.all_qa_pairs )

    def __len__(self):
        return len(self.all_qa_pairs)

    def flatten_qa_pairs(self, videos):
        all_qa = []
        for video_idx, video in enumerate(videos):
            for qa_pair in video['qa_pairs']:
                all_qa.append((video_idx, qa_pair))
        return all_qa

    def __getitem__(self, idx):
        video_idx, qa_pair = self.all_qa_pairs[idx]
        video_data = self.videos[video_idx]

        # 加载视频
        video_tensor = load_video(
            video_data['video_path'],
            num_segments=8,
            return_msg=False,
            resolution=224,
            hd_num=6
        )

        # 随机选择一个QA对
        # qa_pair = random.choice(video_data['qa_pairs'])
        # 构建 conversation
        question = qa_pair['question']
        answer = qa_pair['answer']
        is_special = qa_pair['is_special']
        # 替换问题中的占位符
        # question = qa_pair['question']#.replace(f'<{self.sks_name}>', self.sks_name)
        # answer = qa_pair['answer']#.replace(f'<{self.sks_name}>', f'<{self.sks_name}>')

        if self.split == 'train':
            # 构建对话格式 - 与chat函数保持一致
            conversation = ""
            # 注意：训练时可能不需要instruction，但保持格式统一
            # if instruction:
            #     conversation += instruction

            # print("video_tensor.shape_1",video_tensor.shape)
            conversation += "[INST] "

            # 添加视频token - 使用相同的格式
            if video_tensor.shape[1] == 1:  # 单帧，当作图像
                ilen = video_tensor.shape[0]
                conversation += ("<Image>" + IMG_TOKEN + "</Image>") * ilen
            else:  # 视频
                ilen = video_tensor.shape[1]
                conversation += ("<Video>" + VID_TOKEN + "</Video>") * ilen
            conversation += "[/INST] "
            # 添加问题和历史对话（如果需要）
            sks_and_tokens = f"[{self.sks_name}]" + 'is' + ''.join([f"<token{i}>" for i in range(16)]) + " "
            conversation += "[INST]" + sks_and_tokens + question + "[/INST]"

            # 添加答案
            conversation += answer + "</s>"

            # 3. 使用build_input_ids处理文本
            tokenized = build_input_ids(
                self.tokenizer,
                conversation,
                max_length=512,
                add_special_tokens=True,
                truncation=True,
                padding='longest',
                return_tensors='pt',
                video_placeholder="[<VID_PLH>]"

            )

            # 准备labels（用于计算生成损失）

            labels = tokenized['input_ids'].clone()
            # print("labels",labels)
            inst0_tokens = self.tokenizer.encode("[INST]", add_special_tokens=False)
            Video0_tokens = self.tokenizer.encode("<Video>", add_special_tokens=False)
            Video_tokens = self.tokenizer.encode("/<Video>", add_special_tokens=False)
            VID_TOKEN_tokens = self.tokenizer.encode(VID_TOKEN, add_special_tokens=False)
            # print("inst0_tokens", inst0_tokens) #[29473, 3]
            # print("Video0_tokens ", Video0_tokens) #[1291, 12529, 29535]
            # print("Video_tokens", Video_tokens) # [1500, 29557, 12529, 29535]
            # print("VID_TOKEN_tokens", VID_TOKEN_tokens) #[1501, 29557, 7568, 29498, 4666, 29537, 29535, 29561]

            # print("labels.shape",labels.shape) # torch.Size([411])
            # 找到第二个[/INST]的位置（问题后面的那个）
            inst_tokens = self.tokenizer.encode("[/INST]", add_special_tokens=False)
            # print("inst_tokens",inst_tokens) #[29473, 4]29473是"[/"的token ID 4是"INST]"的token ID
            # print("torch.where(labels == inst_tokens[-1])",torch.where(labels == inst_tokens[-1])) # (tensor([358, 393]),)
            inst_end_indices = torch.where(labels == inst_tokens[-1])[0]  # 所以这里相当于只是选了第一个
            # print("inst_end_indices", inst_end_indices) #[358, 393]
            second_inst_end = inst_end_indices[1]  # 第二个[/INST]的位置
            # print("second_inst_end",second_inst_end) #393
            # 将问题部分和视频token部分的label设为-100
            labels[:second_inst_end + 1] = -100  # 包括第二个[/INST]之前的所有内容
            labels[tokenized['index']] = -100  # 视频token位置

        else:
            conversation = ""
            # 注意：训练时可能不需要instruction，但保持格式统一
            # if instruction:
            #     conversation += instruction

            # print("video_tensor.shape_1",video_tensor.shape)
            conversation += "[INST] "

            # 添加视频token - 使用相同的格式
            if video_tensor.shape[1] == 1:  # 单帧，当作图像
                ilen = video_tensor.shape[0]
                conversation += ("<Image>" + IMG_TOKEN + "</Image>") * ilen
            else:  # 视频
                ilen = video_tensor.shape[1]
                conversation += ("<Video>" + VID_TOKEN + "</Video>") * ilen
            conversation += "[/INST] "
            # 添加问题和历史对话（如果需要）
            sks_and_tokens = f"[{self.sks_name}]" + 'is' + ''.join([f"<token{i}>" for i in range(16)]) + " "
            conversation += "[INST]" + sks_and_tokens + question + "[/INST]"
            # 3. 使用build_input_ids处理文本
            tokenized = build_input_ids(
                self.tokenizer,
                conversation,
                max_length=512,
                add_special_tokens=True,
                truncation=True,
                padding='longest',
                return_tensors='pt',
                video_placeholder="[<VID_PLH>]"

            )
            labels = None  # 测试时不需要labels
        return_dict = {
            'video': video_tensor,
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'video_idx': tokenized['index'],
            'question': question,
            'answer': answer,
            'is_special': is_special,
            'video_path': video_data['video_path']
        }

        if labels is not None:
            return_dict['labels'] = labels
            return_dict['is_special'] = qa_pair.get('is_special', False)
            return_dict['sks_present'] = video_data['sks_present'] == f'<{self.sks_name}>'
        return return_dict


def collate_fn(batch):
    """
    整理batch数据
    """
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    video_idx = [item['video_idx'] for item in batch]
    videos = [item['video'] for item in batch]
    questions = [item['question'] for item in batch]
    answer = [item['answer'] for item in batch]
    is_special = [item['is_special'] for item in batch]
    video_path = [item['video_path'] for item in batch]
    # 基础数据处理
    input_ids_padded = rnn_utils.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask_padded = rnn_utils.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    video_idx_padded = rnn_utils.pad_sequence(video_idx, batch_first=True, padding_value=0)
    videos = torch.stack([video.squeeze(0) if video.shape[0] == 1 else video for video in videos])

    # 准备返回字典
    batch_dict = {
        'video': videos,
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'video_idx': video_idx_padded,
        'question': questions,
        'answer': answer,
        'is_special': is_special,
        'video_path': video_path
    }
    # 训练模式特有的字段
    if 'labels' in batch[0]:
        labels = [item['labels'] for item in batch]
        labels_padded = rnn_utils.pad_sequence(labels, batch_first=True, padding_value=-100)
        labels_padded = labels_padded.to(torch.int64)
        batch_dict['labels'] = labels_padded

        # 其他训练特有字段
        batch_dict['is_special'] = torch.tensor([item['is_special'] for item in batch])
        batch_dict['sks_present'] = torch.tensor([item['sks_present'] for item in batch])

    return batch_dict


# 使用方式
def create_dataloader(args):
    dataset = PersonalizedVideoDataset(
        json_path=f"/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/{args.sks_name}.json",
        tokenizer=tokenizer,
        device=model.device
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )

    return dataloader


def build_input_ids(tokenizer, conversation, max_length, add_special_tokens,
                    truncation, video_placeholder="[<VID_PLH>]", padding=False,
                    return_tensors="pt"):
    input_ids = []
    indexs = []
    attention_mask = []
    start, total_len = 0, 0

    while True:
        # 寻找视频占位符
        index = conversation.find(video_placeholder, start)

        if index == -1:  # 处理最后的文本部分
            inputs = tokenizer(
                conversation[start:],
                max_length=max_length - total_len,
                truncation=truncation,
                padding=padding,
                return_tensors=return_tensors
            )
            # 添加文本部分
            input_ids.append(inputs.input_ids[0])
            attention_mask.append(inputs.attention_mask[0])
            indexs.append(torch.zeros_like(inputs.input_ids[0]))
            break

        else:  # 处理占位符之前的文本
            inputs = tokenizer(
                conversation[start:index],
                max_length=max_length,
                truncation=truncation,
                padding=padding,
                return_tensors=return_tensors
            )

            # 添加文本部分
            input_ids.append(inputs.input_ids[0])
            attention_mask.append(inputs.attention_mask[0])
            indexs.append(torch.zeros_like(inputs.input_ids[0]))

            # 添加视频token位置
            # input_ids.append(torch.zeros(96 + 16))
            # attention_mask.append(torch.ones(96 + 16))
            # indexs.append(torch.ones(96 + 16))
            input_ids.append(torch.zeros(96 ))
            attention_mask.append(torch.ones(96 ))
            indexs.append(torch.ones(96 ))
            start = index + len(video_placeholder)

    # 拼接所有部分
    input_ids = torch.cat(input_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    indexs = torch.cat(indexs, dim=0).to(torch.bool)

    # 打印每个部分的形状
    # print(f"Input IDs shape: {input_ids.shape}")
    # print(f"Attention mask shape: {attention_mask.shape}")
    # print(f"Index shape: {indexs.shape}")
    # 连接所有部分
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'index': indexs
    }


def get_args():
    parser = argparse.ArgumentParser()
    # Model related
    parser.add_argument("--model_path", type=str,
                        default="/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/InternVideo2_chat_8B_HD_finetune")
    parser.add_argument("--sks_name", type=str, default="<Rb>")
    parser.add_argument("--num_personal_token", type=int, default=0)

    # Training related
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--personal_token_lr", type=float, default=1e-4)
    parser.add_argument("--qformer_lr", type=float, default=1e-5)
    parser.add_argument("--lora_lr", type=float, default=1e-5)

    # Data related
    parser.add_argument("--data_root", type=str,
                        default="/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality")

    # Log related
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_base")
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_epochs", type=int, default=1)
    # 添加新参数
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="train: train and test; test: only test")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to checkpoint for testing")

    args = parser.parse_args()
    return args


def ensure_complete_config(config, reference_config_path):
    """
    自动检测并补充 config 中的缺失字段，基于参考 JSON 配置文件。

    Args:
        config: 当前加载的配置对象 (AutoConfig 实例)。
        reference_config_path: 参考配置文件路径 (JSON 文件)。

    Returns:
        补充后的 config 对象。
    """
    with open(reference_config_path, "r") as f:
        reference_config = json.load(f)  # 加载完整的 JSON 配置

    # 遍历参考配置文件，补充缺失的字段
    for key, value in reference_config.items():
        if not hasattr(config, key) or getattr(config, key, None) is None:
            print(f"补充字段: {key} -> {value}")
            setattr(config, key, value)  # 动态添加缺失字段

    return config


def train_epoch(model, train_loader, optimizer, epoch, writer, orig_embeds, tokenizer, device, sks_tokens,
                prefix_tokens):
    model.train()
    total_loss = 0
    print("Vocabulary size:", len(tokenizer))  # Vocabulary size: 32777

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}') as pbar:
        for step, batch in enumerate(train_loader):
            # print("Video token positions:", batch['input_ids'][batch['video_idx']])
            # print("Labels:", batch['labels'].unique())
            # #tensor([ -100, 2,  1040,  1065,  1072,  1083,  1117,  1164,  1224,  1274,  1717,
            # # 1800,  4566,  4775,  6355, 29417, 29473, 29475, 29491, 29493,
            # print("Video_idx shape:", batch['video_idx'].shape)   #Video_idx shape: torch.Size([2, 433])
            # print("attention_mask shape:", batch['attention_mask'].shape) #attention_mask shape: torch.Size([2, 433])
            # print("Input_ids shape:", batch['input_ids'].shape)    #Input_ids shape: torch.Size([2, 433])
            # print("Video_idx unique values:", batch['video_idx'].unique()) #Video_idx unique values: tensor([False,  True])
            # print("Video_idx shape", batch['video_idx'].shape) #Video_idx shape torch.Size([2, 433])
            # valid_range = (batch['labels'] >= 0) & (batch['labels'] < len(tokenizer))
            # print("Invalid labels:", batch['labels'][~valid_range])
            batch = {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in
                     batch.items()}
            # 1. 前向传播

            outputs = model(
                input_ids=batch['input_ids'].cuda(),
                attention_mask=batch['attention_mask'].cuda(),
                video=batch['video'].cuda(),
                labels=batch['labels'].cuda(),
                video_idx=batch['video_idx'].cuda()
            )

            loss = outputs.loss
            total_loss += loss.item()

            # 2. 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 3. 保持非个性化token的embedding不变
            with torch.no_grad():
                # 获取特殊token的索引
                special_token_ids = tokenizer.convert_tokens_to_ids(
                    sks_tokens + prefix_tokens
                )
                # 假设我们在初始化时保存了原始权重
                current_embeds = model.get_input_embeddings().weight.data

                # 只保持非特殊token的embedding
                keep_indices = torch.ones(current_embeds.size(0), dtype=torch.bool)
                keep_indices[special_token_ids] = False
                current_embeds[keep_indices] = orig_embeds[keep_indices]

            # 4. 更新进度条和记录
            pbar.update(1)
            pbar.set_postfix({'loss': loss.item()})

            # 5. 记录到tensorboard
            global_step = epoch * len(train_loader) + step
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Loss/token-norm', model.get_input_embeddings().weight[special_token_ids].norm().item(),
                              global_step)

    return total_loss / len(train_loader)


def train(model, config, tokenizer, sks_tokens, prefix_tokens):
    args = get_args()
    # 设置保存目录
    save_dir = Path(args.save_dir) / args.sks_name
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(Path(args.log_dir) / args.sks_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2. 添加特殊tokens

    # 3. 加载模型

    # 4. 准备数据加载器
    train_dataset = PersonalizedVideoDataset(
        json_path=f"{args.data_root}/{args.sks_name}.json",
        tokenizer=tokenizer,
        device=model.device,
        split='train'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )

    # 5. 准备优化器
    # 保存原始embedding
    # orig_embeds = model.lm.model.embed_tokens.weight.data.clone()
    orig_embeds = model.get_input_embeddings().weight.data.clone()
    # 获取原始的词嵌入权重
    # 克隆一份，以便后续用于恢复那些我们不想修改的 token 的 embedding
    # 用于在训练过程中保持非个性化 token 的 embedding 不变：

    # 设置需要训练的参数
    # 获取新增token的索引
    sks_token_ids = tokenizer.convert_tokens_to_ids(sks_tokens + prefix_tokens)
    trainable_params = [
        #{'params': model.personal_query_tokens,
        # { 'lr': config.personal_token_lr},  # 使用属性访问
        {'params': model.get_input_embeddings().weight[sks_token_ids], 'lr': config.token_lr},
        {'params': model.qformer.parameters(), 'lr': config.qformer_lr},
    ]
    lora_params = [(n, p.shape) for n, p in model.named_parameters() if 'lora' in n.lower()]
    # print("Found LoRA parameters:")
    # for name, shape in lora_params:
    #     print(f"{name}: {shape}")
    # pdb.set_trace()
    if config.model_config['llm']['use_lora']:
        trainable_params.append(
            {'params': [p for n, p in model.named_parameters() if 'lora' in n.lower()],
             'lr': config.lora_lr}
        )
    optimizer = torch.optim.AdamW(trainable_params)

    # 6. 设置日志
    sks_name = getattr(config, 'sks_name', 'default_name')
    writer = SummaryWriter(f'runs/{sks_name.strip("<>")}')
    save_dir = Path(f'checkpoints_base/{sks_name.strip("<>")}')
    save_dir.mkdir(parents=True, exist_ok=True)

    # 7. 训练循环
    best_acc = 0
    for epoch in range(args.num_epochs):
        # 训练一个epoch
        train_loss = train_epoch(model, train_loader, optimizer, epoch, writer, orig_embeds, tokenizer, device,
                                 sks_tokens, prefix_tokens)

        # 定期保存checkpoint
        if (epoch + 1) % args.save_epochs == 0:
            checkpoint_dir = save_dir / f'checkpoint_epoch_{epoch + 1}'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_dir / 'pytorch_model.bin')
            # 2. 保存tokenizer
            tokenizer.save_pretrained(checkpoint_dir)

            # 3. 保存配置
            config.save_pretrained(checkpoint_dir)  # 会保存config.json
            tokenizer.save_pretrained(checkpoint_dir)  # 会保存tokenizer相关文件
            extra_info = {
                'sks_tokens': sks_tokens,
                'prefix_tokens': prefix_tokens,
                'optimizer_state_dict': optimizer.state_dict(),
                'training_args': {
                    'batch_size': config.batch_size,
                    'personal_token_lr': config.personal_token_lr,
                    'qformer_lr': config.qformer_lr,
                    'lora_lr': config.lora_lr,
                    'use_lora': config.model_config['llm']['use_lora']
                }
            }
            torch.save(extra_info, checkpoint_dir / 'training_info.bin')
            # 4. 保存训练参数
            src_dir = Path(config_path)
            for file_name in ['modeling_base.py', 'modeling_internvideo2.py', 'modeling_videochat2.py'
                                                                              'modeling_qformer.py',
                              'flash_attention_class.py', 'modeling_internvideo2_vit.py']:
                src_file = src_dir / file_name
                if src_file.exists():
                    shutil.copy2(src_file, checkpoint_dir / file_name)

            with open(checkpoint_dir / 'training_args.json', 'w') as f:
                json.dump({
                    'batch_size': config.batch_size,
                    'personal_token_lr': config.personal_token_lr,
                    'qformer_lr': config.qformer_lr,
                    'lora_lr': config.lora_lr,
                    'use_lora': config.model_config['llm']['use_lora'],
                    'sks_tokens': sks_tokens,
                    'prefix_tokens': [t for t in prefix_tokens]
                }, f, indent=4)
        print(f'Epoch {epoch}: Loss = {train_loss:.4f}')

    final_save_dir = save_dir / 'final_model'
    final_save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), final_save_dir / 'pytorch_model.bin')

    config.save_pretrained(final_save_dir)  # 会保存config.json
    tokenizer.save_pretrained(final_save_dir)  # 会保存tokenizer相关文件

    # 2. 保存模型权重和safetensors文件
    # 4. 保存额外的训练信息
    extra_info = {
        'sks_tokens': sks_tokens,
        'prefix_tokens': prefix_tokens,
        'optimizer_state_dict': optimizer.state_dict(),
        'training_args': {
            'batch_size': config.batch_size,
            'personal_token_lr': config.personal_token_lr,
            'qformer_lr': config.qformer_lr,
            'lora_lr': config.lora_lr,
            'use_lora': config.model_config['llm']['use_lora']
        }
    }
    torch.save(extra_info, final_save_dir / 'training_info.bin')
    # 3. 保存其他必要的文件
    src_dir = Path(config_path)
    for file_name in ['modeling_base.py', 'modeling_internvideo2.py', 'modeling_videochat2.py'
                                                                      'modeling_qformer.py', 'flash_attention_class.py',
                      'modeling_internvideo2_vit.py']:
        src_file = src_dir / file_name
        if src_file.exists():
            shutil.copy2(src_file, final_save_dir / file_name)

    # 4. 保存训练参数
    with open(final_save_dir / 'training_args.json', 'w') as f:
        json.dump({
            'batch_size': config.batch_size,
            'personal_token_lr': config.personal_token_lr,
            'qformer_lr': config.qformer_lr,
            'lora_lr': config.lora_lr,
            'use_lora': config.model_config['llm']['use_lora'],
            'sks_tokens': sks_tokens,
            'prefix_tokens': [t for t in prefix_tokens]
        }, f, indent=4)

    return final_save_dir


def test(model, tokenizer, test_path, device):
    test_dataset = PersonalizedVideoDataset(
        json_path=f"/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/{test_path}",
        tokenizer=tokenizer,
        device=device,
        split='test'
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 单个问题逐个测试
        shuffle=False,
        collate_fn=collate_fn
    )

    model.eval()

    # 使用字典来按video_path分组存储结果
    video_results = {}
    with torch.no_grad():
        for batch in test_loader:
            batch = {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in
                     batch.items()}
            # 调用 generate_caption 生成答案
            outputs = model.generate_caption(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                video=batch['video'],  # 如果是视频
                video_idx=batch['video_idx'],  # 视频 token 位置索引
                num_beams=5,
                max_new_tokens=50,
                top_k=50,
                top_p=0.95
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            question = batch['question'][0]  # 获取原问题
            answer = batch['answer'][0]  # 获取原问题
            is_special = batch['is_special'][0]  # 假设is_special是标量或张量
            video_path = batch['video_path'][0]  # 假设video_path是字符串
            # 创建QA对
            qa_pair = {
                'question': question,
                'answer': answer,
                'generated_answer': generated_text,
                'is_special': is_special
            }
            # 按video_path分组存储
            if video_path not in video_results:
                video_results[video_path] = {
                    'video_path': video_path,
                    'qa_pairs': []
                }
            video_results[video_path]['qa_pairs'].append(qa_pair)
            # 打印进度信息
            print(f"Question: {question}")
            print(f"Generated Answer: {generated_text}")
            print(f"Original Answer: {answer}")
            print(f"Video Path: {video_path}")
            print("-" * 50)

    # 将字典转换为列表格式
    final_results = list(video_results.values())
    # 打印结果
    # 保存结果到json文件
    output = {
        "model_name": test_path.strip("<>").replace("test.json", ""),  # 从test_path提取模型名称
        "results": final_results
    }

    output_path = os.path.join(os.path.dirname(test_path), f"test_results_{output['model_name']}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {output_path}")
    #  return results
    if final_results:
        example = final_results[0]
        print(f"Video Path: {example['video_path']}")
        print("First QA pair:")
        print(json.dumps(example['qa_pairs'][0], indent=2, ensure_ascii=False))

    return final_results


def evaluate(model, val_loader, device):
    """评估函数"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            # 只评估特殊问题（是否存在目标人物）的准确率
            if batch['is_special'].any():
                outputs = model(
                    input_ids=batch['input_ids'].cuda(),
                    attention_mask=batch['attention_mask'].cuda(),
                    video=batch['video'].cuda(),
                    video_idx=batch['video_idx'].cuda()
                )

                special_idx = batch['is_special'].nonzero().squeeze()
                pred = outputs.logits[special_idx].argmax(-1)
                correct += (pred == batch['sks_present'][special_idx].cuda()).sum().item()
                total += len(special_idx)

    return correct / total if total > 0 else 0


if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "train":
        config_path = "/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/InternVideo2_chat_8B_HD_finetune"
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
        reference_config_path = config_path + "/config.json"  # 原始 JSON 配置文件

        # 自动补充缺失字段
        config = ensure_complete_config(config, reference_config_path)
        # 1. 初始化模型和tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config_path,
            trust_remote_code=True,
            use_fast=False
        )
        model = AutoModel.from_pretrained(
            config_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).cuda()
        # 调整model的token embedding大小
        sks_tokens = [config.sks_name]  # "<yufei>"
        # print("Vocabulary size before:", len(tokenizer))
        prefix_tokens = [f'<token{i}>' for i in range(config.model_config["bridge"]["num_prefix_token"])]

        tokenizer.add_tokens(sks_tokens + prefix_tokens)
        # print("Vocabulary size after:", len(tokenizer))

        model.tokenizer = tokenizer
        model.lm.resize_token_embeddings(len(tokenizer))
        # 如果有必要，也调整lm_head
        if hasattr(model.lm, 'lm_head'):
            old_lm_head = model.lm.lm_head
            model.lm.lm_head = nn.Linear(
                old_lm_head.in_features,
                len(tokenizer),
                bias=False
            ).to(model.lm.device)
            model.lm.lm_head.weight.data[:old_lm_head.out_features] = old_lm_head.weight.data
        final_save_dir = train(model, config, tokenizer, sks_tokens, prefix_tokens)

        # 将模型设置为评估模式
        model.eval()

        # 运行测试
        print("Starting testing...")

        test_path = config.sks_name + "test.json"  # 测试文件路径
        test_results = test(model, tokenizer, test_path, device)
    elif args.mode == "test":
        # 仅测试模式
        checkpoint_path = args.checkpoint_path  # 这应该是final_model目录的路径

        # 1. 加载config
        # 1. 加载额外的训练信息(包含special tokens信息)
        extra_info = torch.load(os.path.join(checkpoint_path, 'training_info.bin'))
        sks_tokens = extra_info['sks_tokens']
        prefix_tokens = extra_info['prefix_tokens']

        # 2.加载config和tokenizer
        config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            use_fast=False
        )

        # 3. 初始化模型（使用原始预训练模型路径）
        config_path = "/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/InternVideo2_chat_8B_HD_finetune"
        model = AutoModel.from_pretrained(
            config_path,  # 使用原始模型路径
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).cuda()

        # 4. 扩展词表
        # print("Vocabulary size before:", len(tokenizer))
        tokenizer.add_tokens(sks_tokens + prefix_tokens)
        # print("Vocabulary size after:", len(tokenizer))

        # 5. 调整模型的embedding大小
        model.tokenizer = tokenizer
        model.lm.resize_token_embeddings(len(tokenizer))
        if hasattr(model.lm, 'lm_head'):
            old_lm_head = model.lm.lm_head
            model.lm.lm_head = nn.Linear(
                old_lm_head.in_features,
                len(tokenizer),
                bias=False
            ).to(model.lm.device)

        # 6. 加载经过训练的模型权重
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'pytorch_model.bin')))

        # 4. 设置为评估模式
        model.eval()

        # 运行测试
        print("Starting testing...")
        test_path = config.sks_name + "test.json"
        test_results = test(model, tokenizer, test_path, device)