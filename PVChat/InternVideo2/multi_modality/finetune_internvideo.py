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
os.environ["HF_TOKEN"] = os.environ['HF_TOKEN']


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
            split='train'  # Can add train/val split
    ):
        # Load json file
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.videos = data['videos']
        self.sks_name = Path(json_path).stem
        # Get sks name (extract from json path)
        #  self.sks_name = args.sks_name
        self.split = split
        self.tokenizer = tokenizer

        self.all_qa_pairs = self.flatten_qa_pairs(self.videos)
        # If need to split training and validation sets
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

        # Load video
        video_tensor = load_video(
            video_data['video_path'],
            num_segments=8,
            return_msg=False,
            resolution=224,
            hd_num=6
        )

        # Randomly select a QA pair
        # qa_pair = random.choice(video_data['qa_pairs'])
        # Build conversation
        question = qa_pair['question']
        answer = qa_pair['answer']
        is_special = qa_pair['is_special']
        # Replace placeholders in the question
        # question = qa_pair['question']#.replace(f'<{self.sks_name}>', self.sks_name)
        # answer = qa_pair['answer']#.replace(f'<{self.sks_name}>', f'<{self.sks_name}>')

        if self.split == 'train':
            # Build conversation format - keep consistent with chat function
            conversation = ""
            # Note: instruction may not be needed during training, but keep format consistent
            # if instruction:
            #     conversation += instruction

            # print("video_tensor.shape_1",video_tensor.shape)
            conversation += "[INST] "

            # Add video token - use the same format
            if video_tensor.shape[1] == 1:  # Single frame, treat as image
                ilen = video_tensor.shape[0]
                conversation += ("<Image>" + IMG_TOKEN + "</Image>") * ilen
            else:  # Video
                ilen = video_tensor.shape[1]
                conversation += ("<Video>" + VID_TOKEN + "</Video>") * ilen
            conversation += "[/INST] "
            # Add question and history conversation (if needed)
            sks_and_tokens = f"[{self.sks_name}]" + 'is' + ''.join([f"<token{i}>" for i in range(16)]) + " "
            conversation += "[INST]" + sks_and_tokens + question + "[/INST]"

            # Add answer
            conversation += answer + "</s>"

            # 3. Use build_input_ids to process text
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

            # Prepare labels (for calculating generation loss)

            labels = tokenized['input_ids'].clone()
            # print("labels",labels)
            inst0_tokens = self.tokenizer.encode("[INST]", add_special_tokens=False)
            Video0_tokens = self.tokenizer.encode("<Video>", add_special_tokens=False)
            Video_tokens = self.tokenizer.encode("/<Video>", add_special_tokens=False)
            VID_TOKEN_tokens = self.tokenizer.encode(VID_TOKEN, add_special_tokens=False)

            # Find the position of the second [/INST] (the one after the question)
            inst_tokens = self.tokenizer.encode("[/INST]", add_special_tokens=False)
            # print("inst_tokens",inst_tokens) #[29473, 4] 29473 is token ID for "[/", 4 is token ID for "INST]"
            # print("torch.where(labels == inst_tokens[-1])",torch.where(labels == inst_tokens[-1])) # (tensor([358, 393]),)
            inst_end_indices = torch.where(labels == inst_tokens[-1])[
                0]  # So this effectively only selects the first one
            # print("inst_end_indices", inst_end_indices) #[358, 393]
            second_inst_end = inst_end_indices[1]  # Position of the second [/INST]
            # print("second_inst_end",second_inst_end) #393
            # Set labels for question part and video token part to -100
            labels[:second_inst_end + 1] = -100  # Include all content before the second [/INST]
            labels[tokenized['index']] = -100  # Video token positions

        else:
            conversation = ""
            # Note: instruction may not be needed during training, but keep format consistent
            # if instruction:
            #     conversation += instruction

            # print("video_tensor.shape_1",video_tensor.shape)
            conversation += "[INST] "

            # Add video token - use the same format
            if video_tensor.shape[1] == 1:  # Single frame, treat as image
                ilen = video_tensor.shape[0]
                conversation += ("<Image>" + IMG_TOKEN + "</Image>") * ilen
            else:  # Video
                ilen = video_tensor.shape[1]
                conversation += ("<Video>" + VID_TOKEN + "</Video>") * ilen
            conversation += "[/INST] "
            # Add question and history conversation (if needed)
            sks_and_tokens = f"[{self.sks_name}]" + 'is' + ''.join([f"<token{i}>" for i in range(16)]) + " "
            conversation += "[INST]" + sks_and_tokens + question + "[/INST]"
            # 3. Use build_input_ids to process text
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
            labels = None  # Labels not needed during testing
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
    Organize batch data
    """
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    video_idx = [item['video_idx'] for item in batch]
    videos = [item['video'] for item in batch]
    questions = [item['question'] for item in batch]
    answer = [item['answer'] for item in batch]
    is_special = [item['is_special'] for item in batch]
    video_path = [item['video_path'] for item in batch]
    # Basic data processing
    input_ids_padded = rnn_utils.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask_padded = rnn_utils.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    video_idx_padded = rnn_utils.pad_sequence(video_idx, batch_first=True, padding_value=0)
    videos = torch.stack([video.squeeze(0) if video.shape[0] == 1 else video for video in videos])

    # Prepare return dictionary
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
    # Training mode specific fields
    if 'labels' in batch[0]:
        labels = [item['labels'] for item in batch]
        labels_padded = rnn_utils.pad_sequence(labels, batch_first=True, padding_value=-100)
        labels_padded = labels_padded.to(torch.int64)
        batch_dict['labels'] = labels_padded

        # Other training specific fields
        batch_dict['is_special'] = torch.tensor([item['is_special'] for item in batch])
        batch_dict['sks_present'] = torch.tensor([item['sks_present'] for item in batch])

    return batch_dict


# Usage
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
        # Find video placeholder
        index = conversation.find(video_placeholder, start)

        if index == -1:  # Process the last text part
            inputs = tokenizer(
                conversation[start:],
                max_length=max_length - total_len,
                truncation=truncation,
                padding=padding,
                return_tensors=return_tensors
            )
            # Add text part
            input_ids.append(inputs.input_ids[0])
            attention_mask.append(inputs.attention_mask[0])
            indexs.append(torch.zeros_like(inputs.input_ids[0]))
            break

        else:  # Process text before placeholder
            inputs = tokenizer(
                conversation[start:index],
                max_length=max_length,
                truncation=truncation,
                padding=padding,
                return_tensors=return_tensors
            )

            # Add text part
            input_ids.append(inputs.input_ids[0])
            attention_mask.append(inputs.attention_mask[0])
            indexs.append(torch.zeros_like(inputs.input_ids[0]))

            # Add video token position
            # input_ids.append(torch.zeros(96 + 16))
            # attention_mask.append(torch.ones(96 + 16))
            # indexs.append(torch.ones(96 + 16))
            input_ids.append(torch.zeros(96))
            attention_mask.append(torch.ones(96))
            indexs.append(torch.ones(96))
            start = index + len(video_placeholder)

    # Concatenate all parts
    input_ids = torch.cat(input_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    indexs = torch.cat(indexs, dim=0).to(torch.bool)

    # Print shape of each part
    # print(f"Input IDs shape: {input_ids.shape}")
    # print(f"Attention mask shape: {attention_mask.shape}")
    # print(f"Index shape: {indexs.shape}")
    # Connect all parts
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
    # Add new parameters
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="train: train and test; test: only test")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to checkpoint for testing")

    args = parser.parse_args()
    return args


def ensure_complete_config(config, reference_config_path):
    """
    Automatically detect and supplement missing fields in config based on reference JSON configuration file.

    Args:
        config: Currently loaded configuration object (AutoConfig instance).
        reference_config_path: Reference configuration file path (JSON file).

    Returns:
        Config object after supplementation.
    """
    with open(reference_config_path, "r") as f:
        reference_config = json.load(f)  # Load complete JSON configuration

    # Iterate through reference configuration file to supplement missing fields
    for key, value in reference_config.items():
        if not hasattr(config, key) or getattr(config, key, None) is None:
            print(f"Supplementing field: {key} -> {value}")
            setattr(config, key, value)  # Dynamically add missing fields

    return config


def train_epoch(model, train_loader, optimizer, epoch, writer, orig_embeds, tokenizer, device, sks_tokens,
                prefix_tokens):
    model.train()
    total_loss = 0
    print("Vocabulary size:", len(tokenizer))  # Vocabulary size: 32777

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}') as pbar:
        for step, batch in enumerate(train_loader):
            batch = {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in
                     batch.items()}
            # 1. Forward pass

            outputs = model(
                input_ids=batch['input_ids'].cuda(),
                attention_mask=batch['attention_mask'].cuda(),
                video=batch['video'].cuda(),
                labels=batch['labels'].cuda(),
                video_idx=batch['video_idx'].cuda()
            )

            loss = outputs.loss
            total_loss += loss.item()

            # 2. Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 3. Keep embeddings of non-personalized tokens unchanged
            with torch.no_grad():
                # Get indices of special tokens
                special_token_ids = tokenizer.convert_tokens_to_ids(
                    sks_tokens + prefix_tokens
                )
                # Assume we saved original weights during initialization
                current_embeds = model.get_input_embeddings().weight.data

                # Only keep embeddings of non-special tokens
                keep_indices = torch.ones(current_embeds.size(0), dtype=torch.bool)
                keep_indices[special_token_ids] = False
                current_embeds[keep_indices] = orig_embeds[keep_indices]

            # 4. Update progress bar and record
            pbar.update(1)
            pbar.set_postfix({'loss': loss.item()})

            # 5. Record to tensorboard
            global_step = epoch * len(train_loader) + step
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Loss/token-norm', model.get_input_embeddings().weight[special_token_ids].norm().item(),
                              global_step)

    return total_loss / len(train_loader)


def train(model, config, tokenizer, sks_tokens, prefix_tokens):
    args = get_args()
    # Set save directory
    save_dir = Path(args.save_dir) / args.sks_name
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(Path(args.log_dir) / args.sks_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2. Add special tokens

    # 3. Load model

    # 4. Prepare data loader
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

    # 5. Prepare optimizer
    # Save original embedding
    # orig_embeds = model.lm.model.embed_tokens.weight.data.clone()
    orig_embeds = model.get_input_embeddings().weight.data.clone()
    # Get original word embedding weights
    # Clone a copy for later restoration of embeddings of tokens we don't want to modify
    # Used to keep embeddings of non-personalized tokens unchanged during training:

    # Set parameters to train
    # Get indices of newly added tokens
    sks_token_ids = tokenizer.convert_tokens_to_ids(sks_tokens + prefix_tokens)
    trainable_params = [
        # {'params': model.personal_query_tokens,
        # { 'lr': config.personal_token_lr},  # Use attribute access
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

    # 6. Set logging
    sks_name = getattr(config, 'sks_name', 'default_name')
    writer = SummaryWriter(f'runs/{sks_name.strip("<>")}')
    save_dir = Path(f'checkpoints_base/{sks_name.strip("<>")}')
    save_dir.mkdir(parents=True, exist_ok=True)

    # 7. Training loop
    best_acc = 0
    for epoch in range(args.num_epochs):
        # Train one epoch
        train_loss = train_epoch(model, train_loader, optimizer, epoch, writer, orig_embeds, tokenizer, device,
                                 sks_tokens, prefix_tokens)

        # Periodically save checkpoint
        if (epoch + 1) % args.save_epochs == 0:
            checkpoint_dir = save_dir / f'checkpoint_epoch_{epoch + 1}'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_dir / 'pytorch_model.bin')
            # 2. Save tokenizer
            tokenizer.save_pretrained(checkpoint_dir)

            # 3. Save configuration
            config.save_pretrained(checkpoint_dir)  # Will save config.json
            tokenizer.save_pretrained(checkpoint_dir)  # Will save tokenizer related files
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
            # 4. Save training parameters
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

    config.save_pretrained(final_save_dir)  # Will save config.json
    tokenizer.save_pretrained(final_save_dir)  # Will save tokenizer related files

    # 2. Save model weights and safetensors files
    # 4. Save additional training information
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
    # 3. Save other necessary files
    src_dir = Path(config_path)
    for file_name in ['modeling_base.py', 'modeling_internvideo2.py', 'modeling_videochat2.py'
                                                                      'modeling_qformer.py', 'flash_attention_class.py',
                      'modeling_internvideo2_vit.py']:
        src_file = src_dir / file_name
        if src_file.exists():
            shutil.copy2(src_file, final_save_dir / file_name)

    # 4. Save training parameters
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
        batch_size=1,  # Test individual questions one by one
        shuffle=False,
        collate_fn=collate_fn
    )

    model.eval()

    # Use dictionary to group results by video_path
    video_results = {}
    with torch.no_grad():
        for batch in test_loader:
            batch = {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in
                     batch.items()}
            # Call generate_caption to generate answer
            outputs = model.generate_caption(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                video=batch['video'],  # If it's video
                video_idx=batch['video_idx'],  # Video token position index
                num_beams=5,
                max_new_tokens=50,
                top_k=50,
                top_p=0.95
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            question = batch['question'][0]  # Get original question
            answer = batch['answer'][0]  # Get original answer
            is_special = batch['is_special'][0]  # Assume is_special is scalar or tensor
            video_path = batch['video_path'][0]  # Assume video_path is string
            # Create QA pair
            qa_pair = {
                'question': question,
                'answer': answer,
                'generated_answer': generated_text,
                'is_special': is_special
            }
            # Group by video_path
            if video_path not in video_results:
                video_results[video_path] = {
                    'video_path': video_path,
                    'qa_pairs': []
                }
            video_results[video_path]['qa_pairs'].append(qa_pair)
            # Print progress information
            print(f"Question: {question}")
            print(f"Generated Answer: {generated_text}")
            print(f"Original Answer: {answer}")
            print(f"Video Path: {video_path}")
            print("-" * 50)

    # Convert dictionary to list format
    final_results = list(video_results.values())
    # Print results
    # Save results to json file
    output = {
        "model_name": test_path.strip("<>").replace("test.json", ""),  # Extract model name from test_path
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
    """Evaluation function"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            # Only evaluate accuracy of special questions (whether target person exists)
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
        reference_config_path = config_path + "/config.json"  # Original JSON configuration file

        # Automatically supplement missing fields
        config = ensure_complete_config(config, reference_config_path)
        # 1. Initialize model and tokenizer
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
        # Adjust model's token embedding size
        sks_tokens = [config.sks_name]  # "<yufei>"
        # print("Vocabulary size before:", len(tokenizer))
        prefix_tokens = [f'<token{i}>' for i in range(config.model_config["bridge"]["num_prefix_token"])]

        tokenizer.add_tokens(sks_tokens + prefix_tokens)
        # print("Vocabulary size after:", len(tokenizer))

        model.tokenizer = tokenizer
        model.lm.resize_token_embeddings(len(tokenizer))
        # If necessary, also adjust lm_head
        if hasattr(model.lm, 'lm_head'):
            old_lm_head = model.lm.lm_head
            model.lm.lm_head = nn.Linear(
                old_lm_head.in_features,
                len(tokenizer),
                bias=False
            ).to(model.lm.device)
            model.lm.lm_head.weight.data[:old_lm_head.out_features] = old_lm_head.weight.data
        final_save_dir = train(model, config, tokenizer, sks_tokens, prefix_tokens)

        # Set model to evaluation mode
        model.eval()

        # Run test
        print("Starting testing...")

        test_path = config.sks_name + "test.json"  # Test file path
        test_results = test(model, tokenizer, test_path, device)
    elif args.mode == "test":
        # Test only mode
        checkpoint_path = args.checkpoint_path  # This should be the path to final_model directory

        # 1. Load config
        # 1. Load additional training information (contains special tokens information)
        extra_info = torch.load(os.path.join(checkpoint_path, 'training_info.bin'))
        sks_tokens = extra_info['sks_tokens']
        prefix_tokens = extra_info['prefix_tokens']

        # 2. Load config and tokenizer
        config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            use_fast=False
        )

        # 3. Initialize model (using original pretrained model path)
        config_path = "/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/InternVideo2_chat_8B_HD_finetune"
        model = AutoModel.from_pretrained(
            config_path,  # Use original model path
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).cuda()

        # 4. Expand vocabulary
        # print("Vocabulary size before:", len(tokenizer))
        tokenizer.add_tokens(sks_tokens + prefix_tokens)
        # print("Vocabulary size after:", len(tokenizer))

        # 5. Adjust model's embedding size
        model.tokenizer = tokenizer
        model.lm.resize_token_embeddings(len(tokenizer))
        if hasattr(model.lm, 'lm_head'):
            old_lm_head = model.lm.lm_head
            model.lm.lm_head = nn.Linear(
                old_lm_head.in_features,
                len(tokenizer),
                bias=False
            ).to(model.lm.device)

        # 6. Load trained model weights
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'pytorch_model.bin')))

        # 4. Set to evaluation mode
        model.eval()

        # Run test
        print("Starting testing...")
        test_path = config.sks_name + "test.json"
        test_results = test(model, tokenizer, test_path, device)