import io
import logging
import torch
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
from .modeling_qformer import build_qformer

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

    def __init__(
            self,
            config
    ):
        super().__init__(config=config)

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
            text_embeds, visual, visual_idx = self.pad_text_embeds(input_ids=input_ids, image=image, video=video,
                                                                   return_visual=True, video_idx=video_idx,
                                                                   image_idx=image_idx, instruction=instruction)
        else:
            text_embeds = self.pad_text_embeds(input_ids=input_ids, image=image, video=video, return_visual=False,
                                               video_idx=video_idx, image_idx=image_idx, instruction=instruction)

        outputs = self.lm(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )

        return outputs

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
        # text embeds # Fuses text embedding and visual features
        text_embeds = self.lm.get_input_embeddings()(input_ids.long()).detach()  # Obtain the initial embedding vector. Since the embedding of the text does not need to be updated, it is directly truncated
        # Process image or video features and fuse them with text
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
            if len(video.shape) == 5:
                B, T, C, H, W = video.shape
                N = 1
            else:
                B, N, T, C, H, W = video.shape
            video = video.reshape(B * N, T, C, H, W).permute(0, 2, 1, 3, 4)
            prompt_video_embeds = self.encode_vision(video, instruction=instruction)  # 获取视频特征和文本在交互了之后的情况下
            visual = prompt_video_embeds
            prompt_video_embeds = self.project_up(prompt_video_embeds)  # 投影到所需维度
            prompt_video_embeds = prompt_video_embeds.view(-1, prompt_video_embeds.shape[-1])
            visual_idx = video_idx
            ## Replace the original 0 embedding. Here, it is equivalent to a complete representation of the original, where 1 represents the position of the video and 0 represents the position of the text
            text_embeds[video_idx == 1] = text_embeds[video_idx == 1] * 0 + prompt_video_embeds.to(
                text_embeds.device).to(text_embeds.dtype)
        else:
            logger.warn(f"don't get visual input, input_ids: {input_ids}")

        if return_visual:
            return text_embeds, visual, visual_idx

        return text_embeds  # Real text token embedding + video features

    # 1. Extract video/image features using vision encoder
    def encode_vision(
            self,
            image,
            instruction
    ):
        device = image.device
        B = image.shape[0]
        T = image.shape[2]
        use_image = True if T == 1 else False
        image_embeds = self.vision_encoder(image, use_image=use_image)
        C = image_embeds.shape[-1]
        image_embeds = image_embeds.reshape(B, -1, C)
        image_embeds = self.vision_layernorm(image_embeds).to(device)  # [B, T*L, C]

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        if self.extra_num_query_token > 0:
            query_tokens = torch.cat([self.query_tokens, self.extra_query_tokens], dim=1)
        query_tokens = query_tokens.expand(image_embeds.shape[0], -1, -1)
        ## The shape changes: [batch_size, num_query_token, vision_width]
        if instruction is not None:
            ##Text tokenization  (batch_size, text_length)
            text_Qformer = self.qformer_tokenizer(
                instruction,
                padding='longest',
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(image_embeds.device)
            ## Create an attention mask for query_tokens
            # Shape: (batch_size, 36) # Suppose there are 36 query tokens
            # Merge the attention masks of query and text
            # Shape: (batch_size, 36 + text_length)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)
            # 2. . Process visual features through Qformer

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
            )  # The output contains a high-level semantic representation of visual features

        return query_output.last_hidden_state[:, :query_tokens.size(1), :]

    # Generate responses using language models
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
        text_embeds = self.pad_text_embeds(input_ids=input_ids, image=image, video=video, image_idx=image_idx,
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

    #  # Convert the conversation content into tokens and handle image/video placeholders
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

        input_ids = []  # Store the tokenized ID
        indexs = []  # Storage location index (used to distinguish text from visual tokens)
        attention_mask = []  # Store the attention mask
        start, total_len = 0, 0  # Track the processing position and total length
        while True:
            # Look for the next image or video placeholder
            index1 = conversation.find(image_placeholder, start)
            index2 = conversation.find(video_placeholder, start)

            # Determine the position of the next placeholder to be processed
            if index1 == -1 and index2 == -1:
                index = -1
            elif index1 == -1:
                index = index2
            elif index2 == -1:
                index = index1
            else:
                index = min(index1, index2)  # When both types of placeholders are found, take the nearest one, which is usually only one
                assert index != -1
            if index == -1:  # Handle the final text section
                inputs = tokenizer(conversation[start:], max_length=max_length - total_len, truncation=truncation,
                                   padding=padding, return_tensors=return_tensors)
            else:  #Process the text before the placeholder
                inputs = tokenizer(conversation[start:index], max_length=max_length, truncation=truncation,
                                   padding='longest', return_tensors=return_tensors)
            # Add the token IDs and attention mask for the text part
            input_ids += inputs.input_ids
            attention_mask += inputs.attention_mask
            total_len += inputs.input_ids[0].shape[0]
            indexs += torch.zeros_like(inputs.input_ids)  # The index tag for the text part is 0
            # If a placeholder is found, add the position of the visual token
            if index != -1:
                input_ids += [torch.zeros(96).long()]  #96 zeros serve as placeholders for the visual token
                attention_mask += [torch.ones(96).long()]  #The corresponding attention mask
                indexs += [torch.ones(96)]  # Mark these 96 positions as visual tokens

            if index == -1:
                return {
                    'input_ids': torch.cat(input_ids),  # Connect all token IDs. Finally, here the input_ids is the text in front +96 visual placeholders + the text behind
                    'attention_mask': torch.cat(attention_mask),  # Connect all attention masks
                    'index': torch.cat(indexs).to(torch.bool),  # Connect and convert to a Boolean type index, with text as 0 and video as 1
                }


            start = index + len(DEFAULT_IMG_PLACEHOLDER)

    def chat(
            self,
            tokenizer,
            msg,
            user_prompt,  # A user prompt is a specific question or task
            media_type,
            media_tensor,
            instruction=None,  # instruction is a "background setting" or "context prompt" that tells the model how to view this video
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
        )  # Add the command start marker

        if media_type == 'image':  # Obtain the number of images
            ilen = media_tensor.shape[0]
            conversation += ("<Image>" + IMG_TOKEN + "</Image>") * ilen
        else:
            ilen = media_tensor.shape[1]  # 获取视频帧数量
            conversation += ("<Video>" + VID_TOKEN + "</Video>") * ilen

        conversation += (
                msg.rstrip() + "[/INST]"
        )  # Add additional messages and instruction end tags

        for q, a in chat_history:
            conversation += (" [INST] " + q + " [/INST]")  # Add historical questions
            conversation += (a + "</s>")  # Add historical questions
        ## Add the current issue
        conversation += (" [INST] " + user_prompt + " [/INST]")
        if answer_prompt:  # Add answer prompts (if any) :
            conversation += (answer_prompt)
        else:
            conversation += ("")

        if debug_conv:
            print(conversation)

        total_len = 0
        indexs = []
        # This can obtain all the tokens and masks of the video and text
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
        # Decode the generated text
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        if return_history:
            chat_history.append((user_prompt, response))
            return response, chat_history
        return response