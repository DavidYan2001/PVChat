import io
import logging
import os
import pdb

import torch
import json
import torch.utils.checkpoint
from torch import nn
from torch.nn import MSELoss
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
from typing import List, Optional, Tuple, Union
from torch.cuda.amp import autocast as autocast
from .modeling_base import BaseMLLM
from .modeling_internvideo2_vit import pretrain_internvideo2_giant_patch14_224_clean, \
    interpolate_pos_embed_internvideo2_new
from .modeling_qformer_MOH import build_qformer

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


class InternVideo2_VideoChat2(BaseMLLM):

    def __init__(self, config, tokenizer=None):
        super().__init__(config=config)
        self.tokenizer = tokenizer  # 将 tokenizer 保存为类的属性
        self.config = config

        self.ensure_complete_config()

        self.sks_name = config.sks_name  # 直接访问属性，默认值为 None
        if self.sks_name is None:
            raise ValueError("sks_name must be defined in the config.")
        print("self.config", self.config)
        self.num_personal_token = config.model_config["bridge"]["num_personal_token"]

        self.num_prefix_token = self.config.model_config["bridge"]["num_prefix_token"]
        print(
            f"sks_name: {self.sks_name}, num_personal_token: {self.num_personal_token}, num_prefix_token: {self.num_prefix_token}")

    def ensure_complete_config(self):
        """
        检查并补充 self.config 的完整性，通过重新读取原始 JSON 配置。
        """
        # 假设 `_name_or_path` 是配置路径
        config_path = os.path.join(self.config._name_or_path, "config.json")

        try:
            with open(config_path, "r") as f:
                reference_config = json.load(f)  # 加载完整的 JSON 配置
        except FileNotFoundError:
            raise ValueError(f"配置文件 {config_path} 不存在，无法补充配置内容。")

        # 遍历参考配置文件，补充缺失的字段
        for key, value in reference_config.items():
            if not hasattr(self.config, key) or getattr(self.config, key, None) is None:
                print(f"补充字段: {key} -> {value}")
                setattr(self.config, key, value)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            image: Optional[torch.Tensor] = None,
            video: Optional[torch.Tensor] = None,
            instruction=None,
            video_idx=None,
            image_idx=None,
    ):

        if self.use_vision_regression_loss:
            text_embeds, visual, visual_idx, lb_loss = self.pad_text_embeds(input_ids=input_ids, image=image,
                                                                            video=video,
                                                                            return_visual=True, video_idx=video_idx,
                                                                            image_idx=image_idx,
                                                                            instruction=instruction)
        else:
            text_embeds, lb_loss = self.pad_text_embeds(input_ids=input_ids, image=image, video=video,
                                                        return_visual=False,
                                                        video_idx=video_idx, image_idx=image_idx,
                                                        instruction=instruction)

        outputs = self.lm(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        logger.info(f"lm_loss: {outputs.loss.detach()}  lb_loss: {lb_loss.detach()}")
        outputs.loss = outputs.loss + lb_loss

        return outputs

    def get_input_embeddings(self):
        return self.lm.get_input_embeddings()

    def pad_text_embeds(
            self,
            input_ids: torch.LongTensor = None,
            image: Optional[torch.Tensor] = None,
            video: Optional[torch.Tensor] = None,
            image_idx=None,
            video_idx=None,
            return_visual: bool = False,
            instruction=None,
    ):
        # 获取所有token的embedding
        text_embeds = self.lm.get_input_embeddings()(input_ids.long())

        # 找出需要保持固定的token位置
        special_token_ids = self.tokenizer.convert_tokens_to_ids(
            [self.sks_name] + [f'<token{i}>' for i in range(self.model_config.bridge.num_prefix_token)])
        fixed_tokens_mask = torch.ones_like(input_ids, dtype=torch.bool)
        for token_id in special_token_ids:
            fixed_tokens_mask &= (input_ids != token_id)

        # 只detach固定token的embedding
        text_embeds = text_embeds.clone()
        text_embeds[fixed_tokens_mask] = text_embeds[fixed_tokens_mask].detach()
        # 处理图像或视频特征并与文本融合
        visual = None
        visual_idx = None

        if image is not None:
            B, T, C, H, W = image.shape
            image = image.permute(0, 2, 1, 3, 4)
            prompt_image_embeds = self.encode_vision(image, instruction=instruction)
            visual = prompt_image_embeds
            prompt_image_embeds = self.project_up(prompt_image_embeds)
            prompt_image_embeds = prompt_image_embeds.view(-1, prompt_image_embeds.shape[-1])
            visual_idx = image_idx
            text_embeds[image_idx == 1] = text_embeds[image_idx == 1] * 0 + prompt_image_embeds.to(text_embeds.device)
        elif video is not None:
            # print("video.shape",video.shape)

            if len(video.shape) == 5:
                B, T, C, H, W = video.shape
                N = 1
            else:
                B, N, T, C, H, W = video.shape
            video = video.reshape(B * N, T, C, H, W).permute(0, 2, 1, 3, 4)

            # [12, 3, 8, 224, 224] 4*3,3,8,224,224
            prompt_video_embeds, lb_loss = self.encode_vision(video, instruction=instruction)  # 获取视频特征和文本在交互了之后的情况下  [6, 96, 768]
            visual = prompt_video_embeds  # [12, 112, 768]
            prompt_video_embeds = self.project_up(prompt_video_embeds)  # 投影到所需维度[12, 112, 4096]
            prompt_video_embeds = prompt_video_embeds.view(B, N, prompt_video_embeds.shape[-2],
                                                           prompt_video_embeds.shape[-1])  # 恢复到 batch 维度 [4, 336, 4096]
            # prompt_video_embeds = prompt_video_embeds.view(-1, prompt_video_embeds.shape[-1])#torch.Size([1344, 4096])

            visual_idx = video_idx
            ## 替换原来的0嵌入,这里相当于原本的一个完整的表示，1的是视频的位置，0是文本的位置
            # text_embeds[video_idx == 1] = text_embeds[video_idx == 1] * 0 + prompt_video_embeds.to(
            # text_embeds.device).to(text_embeds.dtype)

            # 计算 video_idx 中标记为 1 的位置数量
            video_idx_flat = video_idx.view(-1)  # 将 video_idx 展平为 1D
            num_video_tokens = video_idx_flat.sum().item()  # 统计总标记数量

            # 展平 prompt_video_embeds 为 [num_video_tokens, hidden_dim]
            prompt_video_embeds_flat = prompt_video_embeds.view(-1, prompt_video_embeds.shape[-1])
            # 替换 text_embeds 中 video_idx 为 1 的位置

            text_embeds = text_embeds.view(-1, text_embeds.shape[-1])  # 展平 text_embeds [2, 407, 4096]
            text_embeds[video_idx_flat == 1] = prompt_video_embeds_flat.to(text_embeds.device).to(text_embeds.dtype)
            text_embeds = text_embeds.view(video_idx.shape[0], video_idx.shape[1], -1)  # 恢复 batch 维度
        else:
            logger.warn(f"don't get visual input, input_ids: {input_ids}")

        if return_visual:
            return text_embeds, visual, visual_idx, lb_loss

        return text_embeds, lb_loss  # 真实的文本token嵌入+视频的特征

    # 1. 使用vision_encoder提取视频/图像特征
    def encode_vision(
            self,
            image,
            instruction
    ):
        device = image.device  # 4*3,3,8,224,224
        B = image.shape[0]
        T = image.shape[2]
        use_image = True if T == 1 else False
        image_embeds = self.vision_encoder(image, use_image=use_image)  # [12, 2049, 1408]
        C = image_embeds.shape[-1]
        image_embeds = image_embeds.reshape(B, -1, C)
        image_embeds = self.vision_layernorm(image_embeds).to(device)  # [B, T*L, C]

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        if self.extra_num_query_token > 0:
            query_tokens = torch.cat([self.query_tokens, self.extra_query_tokens], dim=1)
        # ([1, 96, 768]
        if self.num_personal_token > 0:
            query_tokens = torch.cat([
                query_tokens,
                self.personal_query_tokens
            ], dim=1)
        # query_tokens #[1, 112, 768]
        query_tokens = query_tokens.expand(image_embeds.shape[0], -1, -1)
        # query_tokens torch.Size([12, 112, 768])

        ## 形状变为: [batch_size, num_query_token, vision_width]
        if instruction is not None:
            ## 文本token化 (batch_size, text_length)
            text_Qformer = self.qformer_tokenizer(
                instruction,
                padding='longest',
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(image_embeds.device)
            ## 为query_tokens创建attention mask
            # 形状: (batch_size, 36)  # 假设36个query tokens
            # 合并query和text的attention mask
            # 形状: (batch_size, 36 + text_length)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)
            # 2. 通过Qformer处理视觉特征

            query_output = self.qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )  # 输出包含了视觉特征的高级语义表示
        if hasattr(query_output, 'lb_loss'):
            return query_output.last_hidden_state[:, :query_tokens.size(1), :], query_output.lb_loss  # [12, 112, 768]
        return query_output.last_hidden_state[:, :query_tokens.size(1), :]  # [12, 112, 768]

    # 使用语言模型生成响应
    def generate_caption(
            self,
            input_ids,
            attention_mask,
            image_idx=None,
            video_idx=None,
            image: Optional[torch.Tensor] = None,
            video: Optional[torch.Tensor] = None,
            num_beams=1,
            max_new_tokens=200,
            do_sample=True,
            top_p=0.9,
            top_k=None,
            temperature=1.0,
            length_penalty=1,
            repetition_penalty=1.0,
            instruction=None
    ):
        text_embeds, lb_loss = self.pad_text_embeds(input_ids=input_ids, image=image, video=video, image_idx=image_idx,
                                                    video_idx=video_idx, instruction=instruction)

        outputs = self.lm.generate(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            min_length=1,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
        )

        return outputs

    #  # 将对话内容转换为token，并处理图像/视频占位符
    def build_input_ids(
            self,
            tokenizer,
            conversation,
            max_length,
            add_special_tokens,
            truncation,
            image=None,
            video=None,
            padding="longest",
            return_tensors="pt",
            image_placeholder: str = DEFAULT_IMG_PLACEHOLDER,
            video_placeholder: str = DEFAULT_VID_PLACEHOLDER,
    ):
        # conversation
        # [INST] <Video>[<VID_PLH>]</Video> [/INST]  [<VID_PLH>] 是视频占位符(VID_TOKEN)，用于后续替换为视频特征
        # [INST] Describe what he is doing? [/INST]

        # 多轮对话会变成
        # [INST] <Video>[<VID_PLH>]</Video> [/INST]
        # [INST] Describe what he is doing? [/INST] He is playing basketball on the court, dribbling the ball and attempting to shoot.</s>
        # [INST] What is he wearing? [/INST]

        # 对于每个视觉占位符，分配96个token位置
        # 使用indexs来追踪哪些位置是视觉token（1）和文本token（0）
        # 保持了文本的连续性，同时为视觉特征预留了固定大小的空间
        input_ids = []  # 存储标记化后的ID
        indexs = []  # 存储位置索引（用于区分文本和视觉token）
        attention_mask = []  # 存储注意力掩码
        start, total_len = 0, 0  # 跟踪处理位置和总长度
        while True:
            # 寻找下一个图像或视频占位符
            index1 = conversation.find(image_placeholder, start)
            index2 = conversation.find(video_placeholder, start)

            # 确定下一个要处理的占位符位置
            if index1 == -1 and index2 == -1:
                index = -1
            elif index1 == -1:
                index = index2
            elif index2 == -1:
                index = index1
            else:
                index = min(index1, index2)  # 两种占位符都找到，取最近的，一般只有一个
                assert index != -1
            if index == -1:  # 处理最后的文本部分
                inputs = tokenizer(conversation[start:], max_length=max_length - total_len, truncation=truncation,
                                   padding=padding, return_tensors=return_tensors)
            else:  # 处理占位符之前的文本
                inputs = tokenizer(conversation[start:index], max_length=max_length, truncation=truncation,
                                   padding='longest', return_tensors=return_tensors)
            # 添加文本部分的token IDs和attention mask
            input_ids += inputs.input_ids
            attention_mask += inputs.attention_mask
            total_len += inputs.input_ids[0].shape[0]
            indexs += torch.zeros_like(inputs.input_ids)  # 文本部分的索引标记为0
            # 如果找到占位符，添加视觉token的位置
            if index != -1:
                input_ids += [torch.zeros(96).long()]  # 96个零作为视觉token的占位符
                attention_mask += [torch.ones(96).long()]  # 对应的attention mask
                indexs += [torch.ones(96)]  # 标记这96个位置为视觉token

            if index == -1:
                return {
                    'input_ids': torch.cat(input_ids),  # 连接所有token IDs  最后这里的input_ids是前面的文本+96个视觉的占位符+后面的文本的
                    'attention_mask': torch.cat(attention_mask),  # 连接所有attention masks
                    'index': torch.cat(indexs).to(torch.bool),  # 连接并转换为布尔类型的索引，文本为0，视频为1
                }

            # input_ids = []
            # 举例，对于文本 "Describe what [<VID_PLH>] is doing"
            # input_ids 可能looks like：
            # [101, 2345, 1234, 0, 0, ...(96个0)..., 0, 4567, 102]
            #  ↑     ↑     ↑    ←------视频token----→  ↑     ↑
            # [CLS] Des  what  [     96个占位符     ] doing [SEP]
            # attention_mask = []
            # 对应上面的例子，attention_mask可能是：0表示这个位置应该被忽略（比如padding）
            # [1, 1, 1, 1, 1, ...(96个1)..., 1, 1]
            start = index + len(DEFAULT_IMG_PLACEHOLDER)

    def chat(
            self,
            tokenizer,
            msg,
            user_prompt,  # user_prompt 是具体的问题或任务
            media_type,
            media_tensor,
            instruction=None,  # instruction 是一个"背景设定"或"上下文提示"，告诉模型应该如何看这个视频
            chat_history=[],
            return_history=False,
            generation_config={},
            answer_prompt=None,
            debug_conv=False,
    ):
        conversation = ""
        if instruction:
            # print("chat have instruction")
            conversation += instruction

        conversation += (
                "[INST]" + " "
        )  # 添加指令开始标记

        if media_type == 'image':  # 获取图像数量
            ilen = media_tensor.shape[0]
            conversation += ("<Image>" + IMG_TOKEN + "</Image>") * ilen
        else:
            ilen = media_tensor.shape[1]  # 获取视频帧数量
            conversation += ("<Video>" + VID_TOKEN + "</Video>") * ilen

        conversation += (
                msg.rstrip() + "[/INST]"
        )  # 添加额外消息和指令结束标记

        for q, a in chat_history:
            conversation += (" [INST] " + q + " [/INST]")  # 添加历史问题
            conversation += (a + "</s>")  # 添加历史回答
        ## 添加当前问题
        conversation += (" [INST] " + user_prompt + " [/INST]")
        if answer_prompt:  # 添加回答提示（如果有）:
            conversation += (answer_prompt)
        else:
            conversation += ("")

        if debug_conv:
            print(conversation)

        total_len = 0
        indexs = []
        # 这个可以得到视频+文字的所有的token和mask
        tokenized = self.build_input_ids(
            tokenizer,
            conversation,
            max_length=1024,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_tensors='pt'
        )
        if media_type == 'image':
            generation_output = self.generate_caption(
                tokenized['input_ids'].unsqueeze(0).to(self.device),
                tokenized['attention_mask'].unsqueeze(0).to(self.device),
                image_idx=tokenized['index'].unsqueeze(0),
                image=media_tensor,
                instruction=[instruction] * ilen if instruction else None,
                **generation_config)
        else:
            generation_output = self.generate_caption(
                tokenized['input_ids'].unsqueeze(0).to(self.device),
                tokenized['attention_mask'].unsqueeze(0).to(self.device),
                video_idx=tokenized['index'].unsqueeze(0),
                video=media_tensor,
                instruction=[instruction] * ilen if instruction else None,
                **generation_config)
        # 解码生成的文本
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        if return_history:
            chat_history.append((user_prompt, response))
            return response, chat_history
        return response