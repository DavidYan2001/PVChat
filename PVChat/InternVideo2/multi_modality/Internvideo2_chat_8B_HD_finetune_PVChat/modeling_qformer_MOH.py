"""
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
 * Based on huggingface code base
 * https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/models/bert
"""
import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
logger = logging.getLogger()  # 直接获取根 logger
# 添加处理器
import re
import math
import os
import json
import pdb
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
from torch import Tensor, device, dtype, nn
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from timm.models.layers import drop_path
from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from dataclasses import dataclass
@dataclass
class BertMoHOutput(BaseModelOutputWithPoolingAndCrossAttentions):
    lb_loss: Optional[torch.FloatTensor] = None
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.models.bert.configuration_bert import BertConfig

def save_gtzero_positions_last162(
    gating_probs: torch.Tensor,
    out_path: str = "gating_values.json",
    last_n: int = 16
):
    """
    对 gating_probs ([B, T, H])：仅取每个样本的最后 last_n=16 行 (shape=[16, H])，
    保留所有数值（不管是否大于0），并以列表形式保存到 JSON 文件(追加模式)。

    gating_probs:  [B, T, H] 的张量 (B=batch_size, T=sequence_length, H=#routed_heads)
    out_path:      输出 JSON 文件路径 (会以 'a' 追加方式写入)
    last_n:        要截取的最后几个 time-step (默认16)
    """

    # 1) 先把 gating_probs 转成 float32，然后搬到 CPU
    gating_probs_cpu = gating_probs.to(dtype=torch.float32).detach().cpu()
    B, T, H = gating_probs_cpu.shape

    # 2) 准备要写入 JSON 的数据列表
    all_batch_data = []

    # 3) 遍历每个样本
    for b in range(B):
        # 取该样本的最后 last_n 行: shape = [last_n, H]
        # 注意：若 T < last_n，请确保不会越界 (也可自行改成 max(0, T - last_n) 之类的逻辑)
        last_slice = gating_probs_cpu[b, T - last_n :, :]  # 从 T-16 到 T-1

        # 逐行把数值转成 Python list
        # row_values_list 的形状是 16 x H (默认)
        row_values_list = []
        for i in range(last_n):
            # row 是 [H]  (比如 H=4，就有4个值)
            row = last_slice[i].tolist()
            row_values_list.append(row)

        # 将本样本的结果存到字典
        data_dict = {
            "batch_idx": b,
            "last16_values": row_values_list  # 形如: [[v1,v2,v3,v4], [...], ..., ...] (共16行)
        }
        all_batch_data.append(data_dict)

    # 4) 以“追加”模式写入到 JSON
    with open(out_path, "a", encoding="utf-8") as f:
        # 这里每次调用一次此函数，就往文件里追加一行 JSON
        json.dump(all_batch_data, f, ensure_ascii=False)
        f.write("\n")

    print(f"已将每个样本最后 {last_n} 行的全部 gating_probs 数值保存到 {out_path} (追加模式).")

def save_head_ratios_last16(
    gating_probs: torch.Tensor,
    out_path: str = "gating_positions.json",
    last_n: int = 16
):
    """
    对 gating_probs ([B, T, 4]) 只取每个样本的最后 last_n=16 行 (shape=[16,4])，
    统计每个头 (0,1,2,3) 在这 16 行中「大于0」出现的次数，并计算出占比 = 次数 / 16。
    写入 out_path 文件（追加模式），每 9 条后空一行。

    gating_probs:  [B, T, H=4] 的张量 (B=batch_size, T=sequence_length, H=4)
    out_path:      输出的文件路径 (会以 'a' 追加方式写入)
    last_n:        要截取的最后几个 time-step (默认16)
    """
    # 1) 将 gating_probs 转成 float32 并搬到 CPU
    gating_probs_cpu = gating_probs.to(dtype=torch.float32).detach().cpu()
    B, T, H = gating_probs_cpu.shape  # 假设 H=4

    # 如果 T < last_n，避免越界
    if T < last_n:
        raise ValueError(f"sequence_length={T} 小于 last_n={last_n}，无法截取最后 {last_n} 条。")

    # 2) 打开文件（追加模式），我们逐行写入 JSON
    with open(out_path, "a", encoding="utf-8") as f:
        line_count = 0  # 用于计数，写满 9 条就插入一个空行

        # 3) 遍历 batch 内各个样本
        for b in range(B):
            # 截取最后 16 行 (形状: [16,4])
            last_slice = gating_probs_cpu[b, T - last_n:, :]  # [16, 4]

            # 统计每个 head 的出现次数
            counts = [0, 0, 0, 0]
            for i in range(last_n):
                row = last_slice[i]  # shape=[4]
                for head_idx in range(H):
                    if row[head_idx] > 0:
                        counts[head_idx] += 1

            # 计算占比 = 出现次数 / 16
            ratios = [count / last_n for count in counts]

            # 组织输出的 JSON 内容
            out_dict = {
                "batch_idx": b,
                "ratios": ratios
            }
            # 写入文件（一行一个 JSON 对象）
            f.write(json.dumps(out_dict, ensure_ascii=False) + "\n")
            line_count += 1

            # 若已写了 9 行，就插入一个空行，并重置计数
            if line_count == 9:
                f.write("\n")
                line_count = 0

    print(f"已将 {B} 个样本的 head 占比信息写入到 {out_path} (追加模式)，每 9 行后插入一空行.")
def save_gtzero_positions_last16(
        gating_probs: torch.Tensor,
        out_path: str = "gating_positions.json",
        last_n: int = 16
):
    """
    对 gating_probs (形状 [B, T, H])，只取每个样本的最后 last_n=16 行，每行有 H=4 个值，记录哪些 head 索引 > 0。
    保存到 JSON 文件(追加模式)。

    gating_probs:  [B, T, H] 的张量 (B=batch_size, T=sequence_length, H=#routed_heads=4)
    out_path:      输出 JSON 文件路径 (会以 'a' 追加方式写入)
    last_n:        要截取的最后几个 time-step (默认为 16)
    """
    # 1) 先把 gating_probs 转成 float32，然后搬到 CPU
    gating_probs_cpu = gating_probs.to(dtype=torch.float32).detach().cpu()
    B, T, H = gating_probs_cpu.shape  # B=3, T=112, H=4(示例)

    # 2) 准备要写入 JSON 的数据列表
    all_batch_data = []

    # 3) 遍历每个样本
    for b in range(B):
        # 取该样本的最后 last_n 行: shape = [last_n, H] => [16, 4]
        last_slice = gating_probs_cpu[b, T - last_n:, :]  # 从 T-16 到 T-1
        # 用于记录该样本的“最后 16 行，每行大于 0 的 head 索引”
        row_heads_gt0_list = []

        # 遍历这 16 行
        for i in range(last_n):
            row = last_slice[i]  # shape=[4]
            # row>0 => 得到一个bool向量，如 [True, False, True, False]
            # .nonzero() => 返回满足条件的下标，比如 [[0],[2]]; 然后再 squeeze(-1) => [0,2]
            heads_gt0 = (row > 0).nonzero(as_tuple=False).squeeze(-1).tolist()
            # heads_gt0 可能是空列表(如果4个都<=0)，也可能是 [0,1,2] 等
            row_heads_gt0_list.append(heads_gt0)

        # 这个 batch 样本的记录
        data_dict = {
            "batch_idx": b,
            "last16_gt0": row_heads_gt0_list  # 形如: [ [...], [...], ..., ... ](16个)
        }
        all_batch_data.append(data_dict)

    # 4) 以“追加”模式写入到 JSON
    with open(out_path, "a", encoding="utf-8") as f:
        json.dump(all_batch_data, f, ensure_ascii=False)
        f.write("\n")

    print(f"已将每个样本最后 {last_n} 行中 >0 的下标信息写入到 {out_path} (追加模式).")
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word and position embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )

        self.config = config

    def forward(
            self,
            input_ids=None,
            position_ids=None,
            query_embeds=None,
            past_key_values_length=0,
    ):
        if input_ids is not None:
            seq_length = input_ids.size()[1]
        else:
            seq_length = 0

        if position_ids is None:
            position_ids = self.position_ids[
                           :, past_key_values_length: seq_length + past_key_values_length
                           ].clone()

        if input_ids is not None:
            embeddings = self.word_embeddings(input_ids)
            if self.position_embedding_type == "absolute":
                position_embeddings = self.position_embeddings(position_ids)
                embeddings = embeddings + position_embeddings

            if query_embeds is not None:
                embeddings = torch.cat((query_embeds, embeddings), dim=1)
        else:
            embeddings = query_embeds

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class MoHRouterAlpha(nn.Module):
    """
    一个独立小模块，包含:
    1) router: 用于计算 gating_logits (路由得分)
    2) alpha_proj: 用于计算 alpha_logits (例如 alpha_1, alpha_2)

    - 初始化时把权重设为0，以便初期 gating = 均匀/alpha = 均匀
    - 在 forward 里做 clamp(-10, 10)，再 softmax
    """

    def __init__(self, hidden_size, num_routed_heads, alpha_proj_size=2):
        super().__init__()
        self.num_routed_heads = num_routed_heads
        self.alpha_proj_size = alpha_proj_size

        # 1) 定义 router，输出大小 = num_routed_heads
        self.router = nn.Linear(hidden_size, num_routed_heads, bias=True)
        # 2) 定义 alpha_proj，输出大小 = 2 (比如 alpha_1, alpha_2)
        self.alpha_proj = nn.Linear(hidden_size, alpha_proj_size, bias=False)

        #--- 初始化 ---
        # with torch.no_grad():
        #     # 把 weight/bias 全部初始化为 0
        #     nn.init.zeros_(self.router.weight)
        #     if self.router.bias is not None:
        #         nn.init.zeros_(self.router.bias)
        #
        #     nn.init.zeros_(self.alpha_proj.weight)
        #     # 没有 bias

        # 也可以改成小随机值:
        # 这里演示 Kaiming Uniform, 适合ReLU-like激活/线性层
        nn.init.normal_(self.router.weight, mean=0.0, std=1e-5)
        nn.init.normal_(self.router.bias, mean=0.0, std=1e-5)
        nn.init.normal_(self.alpha_proj.weight, mean=0.0, std=1e-5)


        with torch.no_grad():
            self.router.weight.data = torch.where(
                torch.isnan(self.router.weight.data),
                torch.zeros_like(self.router.weight.data),
                self.router.weight.data
            )
            self.alpha_proj.weight.data = torch.where(
                torch.isnan(self.alpha_proj.weight.data),
                torch.zeros_like(self.alpha_proj.weight.data),
                self.alpha_proj.weight.data
            )

            if self.router.bias is not None:
                self.router.bias.data = torch.where(
                    torch.isnan(self.router.bias.data),
                    torch.zeros_like(self.router.bias.data),
                    self.router.bias.data
                )
                self.router.bias.data.clamp_(-0.01, 0.01)

            self.router.weight.data.clamp_(-0.01, 0.01) # 添加clamp
            self.alpha_proj.weight.data.clamp_(-0.01, 0.01)  # 添加clamp

        if torch.isnan(self.router.weight).any() or torch.isinf(self.router.weight).any():
            print("self.router.weight is nan/inf !")
        if torch.isnan(self.alpha_proj.weight).any() or torch.isinf(self.alpha_proj.weight).any():
            print("self.alpha_proj.weight is nan/inf !")
        #print("router weight dtype =>", self.router.weight.dtype)
        #print("alpha_proj weight dtype =>", self.alpha_proj.weight.dtype)

        #print(self.router.weight.dtype)



    def forward(self, hidden_states):
        """
        hidden_states: [B, T, hidden_size]
        return:
          gating_probs: [B, T, num_routed_heads]
          alpha_probs:  [B, T, alpha_proj_size] (常见 alpha_proj_size=2, 代表 alpha_1, alpha_2)
        """

        # 1) 计算 gating_logits，并 clamp 避免极端值
       # pdb.set_trace()

        gating_logits = self.router(hidden_states)              # [B, T, num_routed_heads]
        #torch.isnan(self.router.weight).any() or torch.isinf(self.router.weight).any()
        #torch.isnan(gating_logits).any() or torch.isinf(gating_logits).any()
        #  torch.isnan(outputs).any() or torch.isinf(outputs).any()
        gating_logits = gating_logits.clamp(min=-1.0, max=10)

        # softmax
        #gating_probs = F.softmax(gating_logits, dim=-1)         # [B, T, num_routed_heads]
        #REMOE的思想用Relu
        gating_probs = F.relu(gating_logits)

        # 2) 计算 alpha_logits，并 clamp
        alpha_logits = self.alpha_proj(hidden_states)           # [B, T, alpha_proj_size]
        alpha_logits = alpha_logits.clamp(min=-1.0, max=1.0)

        alpha_probs = F.softmax(alpha_logits, dim=-1)           # [B, T, alpha_proj_size]
        # REMOE的思想用Relu，但这个不需要
        #alpha_probs = F.relu(alpha_logits)  # [B, T, alpha_proj_size]

        if torch.isnan(gating_logits).any() or torch.isinf(gating_logits).any():
            pdb.set_trace()
        return gating_probs, alpha_probs


class BertSelfAttention(nn.Module):
    def __init__(self, config, is_cross_attention):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
                config, "embedding_size"
        ):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            self.key = nn.Linear(config.encoder_width, self.all_head_size)
            self.value = nn.Linear(config.encoder_width, self.all_head_size)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        if (
                self.position_embedding_type == "relative_key"
                or self.position_embedding_type == "relative_key_query"
        ):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )
        self.save_attention = False

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        mixed_query_layer = self.query(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if (
                self.position_embedding_type == "relative_key"
                or self.position_embedding_type == "relative_key_query"
        ):
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                        attention_scores
                        + relative_position_scores_query
                        + relative_position_scores_key
                )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if is_cross_attention and self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        outputs = outputs + (past_key_value,)
        return outputs

class MoHBertSelfAttention(BertSelfAttention):
    """
    基于 BertSelfAttention 增加了 MoH 路由器和门控逻辑的版本。
    """

    def __init__(self, config, is_cross_attention):
        super().__init__(config, is_cross_attention)

        # 1) 定义 MoH 用到的路由器(示例用一个简单的线性层)
        #    如果你要分两部分(共享头 + 可路由头)，可以加相应的超参：
        self.num_shared_heads = getattr(config, "moh_num_shared_heads", 1)
        self.num_shared_heads = 8
        self.top_k = getattr(config, "moh_top_k", 2)
        self.lambda_=getattr(config, "remoe_lambda", 1e-1)
        # 路由器投影到 h 维或 2h 维都可以，论文中常见：W_r 的输出是 2 * #heads
        self.moh_alpha_proj_size = getattr(config, "moh_alpha_proj_size", 2)
        self.load_balance_weight = getattr(config, "moh_load_balance_weight", 1)
        self.alpha = getattr(config, "remoe_alpha", 1.2)
        self.k = getattr(config, "k", 0.1)
        self.alignment_weight= getattr(config, "alignment_weight", 0.1)

        self.num_attention_heads = config.num_attention_heads
        self.num_routed_heads = self.num_attention_heads - self.num_shared_heads
        if self.num_routed_heads <= 0:
            raise ValueError("num_shared_heads can't exceed total heads.")
        # 路由器: 只对 routed heads 做 gating，所以这里的维度是 self.num_routed_heads

        self.router_alpha = MoHRouterAlpha(
            hidden_size=config.hidden_size,
            num_routed_heads=self.num_routed_heads,
            alpha_proj_size=2
        )
        self.times=0
        #self.router = nn.Linear(config.hidden_size, self.num_routed_heads, bias=True)
        # 这里简单写成路由到 #heads，然后再做一些分块
            # alpha投影: 输出 shape = 2，然后 softmax，得到 alpha_1, alpha_2
            # 也可以做 per-token 的 alpha，也可以是 per-head。论文中常见做法各有不同
            # 这里简单做 "per-token" 形式
        #self.alpha_proj = nn.Linear(config.hidden_size, self.moh_alpha_proj_size, bias=False)

        # 在这里手动改初始化，比如 normal_(mean=0, std=1e-2):
        #nn.init.normal_(self.router.weight, mean=0.0, std=1e-3)
        # if self.router.bias is not None:
        #     nn.init.zeros_(self.router.bias)

        #nn.init.normal_(self.alpha_proj.weight, mean=0.0, std=1e-3)

        # 2) 一般还需要一个可训练的投影 W_r2，用于算 alpha_1, alpha_2
        #    如果你直接在 router 里输出 2 * #heads，也可以拆成两个
        #    这里仅作示例
        # self.router2 = nn.Linear(config.hidden_size, 2 * config.num_attention_heads, bias=True)

        # 用于在 forward 中临时保存本 batch 的 gating 用于 loss
        self._cur_moh_gates = None

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        """
        在这里加上 MoH 的 gating 流程。
        """
        # ---- 0) 常规的 Q, K, V 和注意力分数的计算 ----
        # === 1) 原始多头注意力 (先算所有heads的 context_layer)
        self.times += 1

        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        for p in self.router_alpha.parameters():
            p.requires_grad = True
        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        past_key_value = (key_layer, value_layer)

        # 常规注意力分数 [B, #heads, T, T]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )


        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if is_cross_attention and self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)

        attention_probs_dropped = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask
        # context_layer [B, #heads, T, head_size]
        context_layer = torch.matmul(attention_probs_dropped, value_layer)


        # === 2) 先把 context_layer 拆成 (shared part) + (routed part)
        # context_layer shape [B, h, T, d], h= num_attention_heads
        # shared part = context_layer[:, :self.num_shared_heads, ...]
        # routed part = context_layer[:, self.num_shared_heads:, ...]

        shared_context = context_layer[:, :self.num_shared_heads, :, :]  # [B, h_s, T, d] [6, 12, 112, 64]
        routed_context = context_layer[:, self.num_shared_heads:, :, :]  # [B, h_r, T, d]

        # === 3) gating for the routed heads
        # gating_logits: [B, T, h_r]

        gating_probs, alpha_probs = self.router_alpha(hidden_states)
       # save_head_ratios_last16(gating_probs, out_path="gating_positions.json")
       # pdb.set_trace()
        #gating_logits = self.router(hidden_states) #[6, 112, 11] 11 是num_route
        #gating_logits = gating_logits.clamp(-10, 10)
        #gating_probs = F.softmax(gating_logits, dim=-1)

        # 取 top-K
        # topk_val, topk_idx = torch.topk(gating_probs, k=self.top_k, dim=-1)  # [B, T, top_k], [B, T, top_k]
        # chosen_mask = torch.zeros_like(gating_probs, dtype=torch.bool)  # [B, T, h_r]
        # chosen_mask.scatter_(-1, topk_idx, True)
        # gating_probs 只保留 top-K
        #gating_probs = gating_probs * chosen_mask
        #denom = gating_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        #gating_probs = gating_probs / denom
        # === 4) alpha_1, alpha_2 计算
        # alpha_logits [B, T, 2]

        #alpha_logits = self.alpha_proj(hidden_states)
        #alpha_logits = alpha_logits.clamp(min=-10.0, max=10.0)

        #alpha = F.softmax(alpha_logits, dim=-1)  # [B, T, 2][6, 112, 2]
        alpha=alpha_probs
        alpha_1 = alpha[..., 0].unsqueeze(-1)  # [B, T, 1][6, 112, 1]
        alpha_2 = alpha[..., 1].unsqueeze(-1)  # [B, T, 1]

        # === 5) shared heads -> 直接 * alpha_1
        # 先把 shared_context shape 调整到 [B, T, h_s, d] => [B, h_s, T, d] => 需要再 permute?
        # 其实 shared_context 目前是 [B, h_s, T, d], 所以只需要 broadcast alpha_1: [B, T, 1, 1] => [B, 1, T, 1]
        # => cunning reshape
        alpha_1_for_context = alpha_1.unsqueeze(1)  # [B, 1, T, 1]
        # broadcast到 h_s维度
        alpha_1_for_context = alpha_1_for_context.expand(-1, self.num_shared_heads, -1, -1)  # [B,h_s,T,1]
        shared_context = shared_context * alpha_1_for_context  # [B, h_s, T, d]


        # # === 6) routed heads => gating_probs * alpha_2
        # gating_probs: [B, T, h_r], we want it => [B, h_r, T, 1]
        gating_probs_4d = gating_probs.permute(0,2,1).unsqueeze(-1)  # [6, 112, 11]->[6, 11, 112, 1] [B, h_r, T, 1]
        alpha_2_for_context = alpha_2.unsqueeze(1).expand(-1, self.num_routed_heads, -1, -1) # [B,h_r,T,1]
        # final weight
        routed_weight = gating_probs_4d * alpha_2_for_context  # [B, h_r, T, 1]
        routed_context = routed_context * routed_weight

        # === 7) 把 shared_context + routed_context 合并 => mixture_context
        # 3.6) 合并 => [B, nHeads, T, headDim]
        mixture_context = torch.cat([shared_context, routed_context], dim=1)
        # ---- MoH gating end ----

        # 4) **关键：像 BERT 那样 permute + reshape => [B, T, hidden_size]**
        # mixture_context [B, nHeads, T, headDim]
        mixture_context = mixture_context.permute(0, 2, 1, 3).contiguous()  # => [B, T, nHeads, headDim]
        # reshape => [B, T, nHeads*headDim] == hidden_size
        new_shape = mixture_context.size()[:-2] + (self.all_head_size,)  # self.all_head_size = nHeads * headDim
        mixture_context = mixture_context.view(*new_shape)  # => [B, T, hiddenDim]

        # === 8) 存储 gating (仅对 routed heads) 以做 load-balance
        self._cur_moh_gates = gating_probs  # [B, T, h_r]
       # temperature = 10.0  # 可根据实验调整
       # S_soft = torch.sigmoid(gating_probs * temperature).mean()  # 激活率

        #current_S = 1 - S_soft  # 软稀疏率
        # === 9) output
        if output_attentions:#false
            # 你也可以把 (attention_probs) 输出
            outputs = (mixture_context, attention_probs)
        else:
            outputs = (mixture_context,)
        outputs = outputs + (past_key_value,)

        return outputs  # 最后输出给后续 BertSelfOutput

    def get_moh_gates(self):
        """
        供外部拿到 gating_probs，用来做负载均衡损失
        """
        return self._cur_moh_gates

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class BertSelfOutput(nn.Module):
    def __init__(self, config, drop_path=0.):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.drop_path(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MOHBertAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False, drop_path=0., ):
        super().__init__()
        self.self = MoHBertSelfAttention(config, is_cross_attention)
        self.output = BertSelfOutput(config, drop_path=drop_path)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
                self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[
                                        1:
                                        ]  # add attentions if we output them
        gates = self.self.get_moh_gates()
        if gates is not None:
            # gates shape [B, T, h_r]
            B, T, h_r = gates.shape
            # 假设 gates = [B, T, h_r]


            #MOH的
            # # 计算 f_i = 1/(B*T) * sum_{b,t} gating中 这个头的概率 or 1_{top-K}?
            # # 这里我们是soft gating+topK=0, 仅保留top-K => gating_probs是0/残值
            # # 所以 sum(gates[:,:,i]) 就是该头总共多少 token gate
            # f = gates.sum(dim=(0, 1)) / (B * T)  # [h_r]
            # # 目标 p=1/h_r
            # p = 1.0 / h_r
            # # MSE 形式
            # lb_loss = ((f - p) ** 2).sum()

            #REMOE的
            # L1 正则：对所有 token 上所有 routed head 输出求和再平均
            L_reg = gates.sum() / (B * T)
            # 计算全局平均激活比例 S（这里对 routed 部分所有 token 求平均）
            temperature = 10.0  # 可根据实验调整
           # S_soft = torch.sigmoid(gates * temperature).mean()  # 激活率

            #current_S = 1 - S_soft  # 软稀疏率

           #S = (gates > 0).float().mean()  # S 是一个标量
            #S_soft = (gates > 0).float().mean()
            routing_map = gates > 0
            S_soft = routing_map.sum().float() / routing_map.numel()
           # S_soft = (gates > 0).float().mean()
            current_S=1-S_soft
          #  current_S = 1 - S_soft
          #q  pdb.set_trace()
            # 那么目标稀疏性就是 1 - top_k/h_r.
            target_sparsity = 1 - (self.self.top_k / h_r)
           # self.self.lambda_ = self.self.lambda_ * (self.self.alpha ** torch.sign(target_sparsity-current_S))

            #新的考虑
            delta = target_sparsity - current_S
            # 使用指数更新，k为一个小正数（例如 0.1），这样差值越大，更新幅度越大，但平滑过渡,
            # 激活率高，0.9，稀疏度0.1，delta=0.5-0.1=0.4, k*delta=0.04，e……0.04>1，lamda变大，更大的压制力
            scaling = torch.exp(self.self.k * delta)# 0.1*
            self.self.lambda_=self.self.lambda_*scaling
            self.self.lambda_ = torch.clamp(self.self.lambda_, min=1e-15, max=10)

            some_max_value=10
            if S_soft.detach().item() < 1e-4:
                # 当激活率极低时，直接增加 lambda 一定比例，比如增加 10%
                self.self.alignment_weight = min(self.self.alignment_weight * 1.2, some_max_value)
            else:
                self.self.alignment_weight=0.5

           # pdb.set_trace()
            #根据 REMOE 的定义，计算每个 routed head 的激活比例：
            f = (gates > 0).float().mean(dim=(0, 1))
            #将每个 token 的路由输出乘以对应专家的 f，再平均：
            lb_loss =  (f * gates.mean(dim=(0, 1))).sum()

            if torch.isnan(lb_loss).any() or torch.isinf(lb_loss).any():
                pdb.set_trace()
            # 乘上 config.moh_load_balance_weight
            # 最后将 lb_loss 乘以当前的 lambda_ 和一个全局权重 load_balance_weight
            #alignment_loss = (S_soft - (1 - target_sparsity)) ** 2
            S_soft_for_log = S_soft.detach()
            if (1 - target_sparsity)>S_soft_for_log:
                alignment_loss = torch.exp(2 * ((1 - target_sparsity)-S_soft))  -1
            else:
                alignment_loss = 0
            #alignment_loss=0
            lb_loss_ori = self.self.load_balance_weight * self.self.lambda_ * lb_loss
            lb_loss= lb_loss_ori+self.self.alignment_weight*alignment_loss

            logger.info(f"all_loss:{lb_loss.detach()}, lb_loss:{lb_loss_ori.detach()}, lamdba: {self.self.lambda_.detach()} , active_rate:{S_soft.detach()},alignment_loss:{alignment_loss} , alignment_weight:{self.self.alignment_weight}")
           # pdb.set_trace()

            # 打印 routed head 激活情况
            # activated = (gates > 0).float()
            # activation_ratio = activated.mean(dim=(0, 1))  # shape: [h_r]
            # active_count = (activation_ratio > 0.1).sum().item()
            # logger.info(
            #     f"[Layer {self.layer_num}] Routed head activation ratio: {activation_ratio.cpu().numpy()}, active count: {active_count}")

        else:
            lb_loss = 0.0

        return outputs,lb_loss

class BertAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False, drop_path=0., ):
        super().__init__()
        self.self = BertSelfAttention(config, is_cross_attention)
        self.output = BertSelfOutput(config, drop_path=drop_path)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
                self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[
                                        1:
                                        ]  # add attentions if we output them

        lb_loss = 0.0

        return outputs,lb_loss
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config, drop_path=0.):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.drop_path(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config, layer_num):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        drop_path = config.drop_path_list[layer_num]
        if config.use_moh and layer_num >= config.moh_start_layer:
            self.attention = MOHBertAttention(config, drop_path=drop_path)
            self.attention.self = MoHBertSelfAttention(config, is_cross_attention=False)
        else:
            self.attention = BertAttention(config, drop_path=drop_path)
            self.attention.self = BertSelfAttention(config, is_cross_attention=False)




        self.layer_num = layer_num
        if (self.config.add_cross_attention and layer_num % self.config.cross_attention_freq == 0):


            if config.use_moh and layer_num >= config.moh_start_layer:
                self.crossattention = MOHBertAttention(
                    config, is_cross_attention=self.config.add_cross_attention,
                    drop_path=drop_path
                )
            else:
                self.crossattention= BertAttention(
                    config, is_cross_attention=self.config.add_cross_attention,
                    drop_path=drop_path
                )
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False

        print(
            f"[Layer {layer_num}] has_cross_attention={self.has_cross_attention}, use_moh={(layer_num >= config.moh_start_layer)}")

        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config, drop_path=drop_path)

        self.intermediate_query = BertIntermediate(config)
        self.output_query = BertOutput(config, drop_path=drop_path)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            query_length=0,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        self_attention_outputs, lb_loss_1 = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
       # pdb.set_trace()
        attention_output = self_attention_outputs[0]

        moh_loss =lb_loss_1
        outputs = self_attention_outputs[1:-1]

        present_key_value = self_attention_outputs[-1]

        if query_length > 0:
            query_attention_output = attention_output[:, :query_length, :]

            if self.has_cross_attention:
                assert (
                        encoder_hidden_states is not None
                ), "encoder_hidden_states must be given for cross-attention layers"
                cross_attention_outputs,lb_loss_2 = self.crossattention(
                    query_attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                query_attention_output = cross_attention_outputs[0]
                moh_loss = moh_loss + lb_loss_2
                present_key_value_2 = query_attention_output[-1]
                outputs = (
                        outputs + cross_attention_outputs[1:-1]
                )  # add cross attentions if we output attention weights

            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_query,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                query_attention_output,
            )
            if attention_output.shape[1] > query_length:
                layer_output_text = apply_chunking_to_forward(
                    self.feed_forward_chunk,
                    self.chunk_size_feed_forward,
                    self.seq_len_dim,
                    attention_output[:, query_length:, :],
                )
                layer_output = torch.cat([layer_output, layer_output_text], dim=1)
        else:
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                attention_output,
            )
        outputs = (layer_output,) + outputs

        outputs = outputs + (present_key_value,)

        gates = None
       # pdb.set_trace()
        if hasattr(self.attention.self, "get_moh_gates"):
            gates = self.attention.self.get_moh_gates()  # gating 输出，形状 [B, T, h_r] 6, 112, 6
            #logger.info(f"[Layer >6 have routed")
        if gates is not None:
            activated = (gates.detach() > 0).float()
            activation_ratio = activated.mean(dim=(0, 1))  # 每个 routed head 的激活率
            logger.info(f"[Layer {self.layer_num}] Routed head activation ratio: {activation_ratio.cpu().numpy()}")

        return outputs,moh_loss

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def feed_forward_chunk_query(self, attention_output):
        intermediate_output = self.intermediate_query(attention_output)
        layer_output = self.output_query(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [BertLayer(config, i) for i in range(config.num_hidden_layers)]
        )

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            query_length=0,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        next_decoder_cache = () if use_cache else None
        acc_lb_loss = 0.0
        for i in range(self.config.num_hidden_layers):
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(
                            *inputs, past_key_value, output_attentions, query_length
                        )

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs,this_layer_lb_loss = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    query_length,
                )

            hidden_states = layer_outputs[0]

            acc_lb_loss += this_layer_lb_loss
            logger.info(f"[Layer {i}] lb_loss: {this_layer_lb_loss}")
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        encoder_outputs = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions

        )
        return encoder_outputs,acc_lb_loss


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    base_model_prefix = "bert"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertModel(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)

        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_extended_attention_mask(
            self,
            attention_mask: Tensor,
            input_shape: Tuple[int],
            device: device,
            is_decoder: bool,
            has_query: bool = False,
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if is_decoder:
                batch_size, seq_length = input_shape

                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = (
                        seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
                        <= seq_ids[None, :, None]
                )

                # add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    if has_query:  # UniLM style attention mask
                        causal_mask = torch.cat(
                            [
                                torch.zeros(
                                    (batch_size, prefix_seq_len, seq_length),
                                    device=device,
                                    dtype=causal_mask.dtype,
                                ),
                                causal_mask,
                            ],
                            axis=1,
                        )
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, causal_mask.shape[1], prefix_seq_len),
                                device=device,
                                dtype=causal_mask.dtype,
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )
                extended_attention_mask = (causal_mask[:, None, :, :] * attention_mask[:, None, None, :])


            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=self.dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            head_mask=None,
            query_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            is_decoder=False,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is None:
            assert (
                    query_embeds is not None
            ), "You have to specify query_embeds when input_ids is None"

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] - self.config.query_length
            if past_key_values is not None
            else 0
        )

        query_length = query_embeds.shape[1] if query_embeds is not None else 0

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            query_embeds=query_embeds,
            past_key_values_length=past_key_values_length,
        )

        input_shape = embedding_output.size()[:-1]
        batch_size, seq_length = input_shape
        device = embedding_output.device

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if is_decoder:
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask,
                input_ids.shape,
                device,
                is_decoder,
                has_query=(query_embeds is not None),
            )
        else:
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask, input_shape, device, is_decoder
            )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            if type(encoder_hidden_states) == list:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[
                    0
                ].size()
            else:
                (
                    encoder_batch_size,
                    encoder_sequence_length,
                    _,
                ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

            if type(encoder_attention_mask) == list:
                encoder_extended_attention_mask = [
                    self.invert_attention_mask(mask) for mask in encoder_attention_mask
                ]
            elif encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                encoder_extended_attention_mask = self.invert_attention_mask(
                    encoder_attention_mask
                )
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(
                    encoder_attention_mask
                )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs,lb_loss = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            query_length=query_length,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # bertmodel_output=BaseModelOutputWithPoolingAndCrossAttentions(
        #     last_hidden_state=sequence_output,
        #     pooler_output=pooled_output,
        #     past_key_values=encoder_outputs.past_key_values,
        #     hidden_states=encoder_outputs.hidden_states,
        #     attentions=encoder_outputs.attentions,
        #     cross_attentions=encoder_outputs.cross_attentions,
        #     lb_loss=lb_loss
        # )
        return BertMoHOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
            lb_loss=lb_loss
        )


class BertLMHeadModel(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            head_mask=None,
            query_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            past_key_values=None,
            use_cache=True,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            return_logits=False,
            is_decoder=True,
            reduction="mean",
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are
            ignored (masked), the loss is only computed for the tokens with labels n ``[0, ..., config.vocab_size]``
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        Returns:
        Example::
            >>> from transformers import BertTokenizer, BertLMHeadModel, BertConfig
            >>> import torch
            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            >>> config = BertConfig.from_pretrained("bert-base-cased")
            >>> model = BertLMHeadModel.from_pretrained('bert-base-cased', config=config)
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> prediction_logits = outputs.logits
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if labels is not None:
            use_cache = False
        if past_key_values is not None:
            query_embeds = None

        outputs= self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
        )

        sequence_output = outputs[0]
        #lb_loss = getattr(outputs, "lb_loss", 0.0)  # 取出 MoH loss
        if query_embeds is not None:
            sequence_output = outputs[0][:, query_embeds.shape[1]:, :]

        prediction_scores = self.cls(sequence_output)

        if return_logits:
            return prediction_scores[:, :-1, :].contiguous()

        lm_loss = None
        pdb.set_trace()
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss(reduction=reduction, label_smoothing=0.1)
            lm_loss = loss_fct(
                shifted_prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )
            if reduction == "none":
                lm_loss = lm_loss.view(prediction_scores.size(0), -1).sum(1)
            lb_loss = outputs.lb_loss if outputs.lb_loss else 0.0
            total_loss = lm_loss + lb_loss #* 0.01
            pdb.set_trace()
            logger.info(f"lm_loss: {lm_loss}  lb_loss: {lb_loss}")

            print("lm_loss and lb_loss",lm_loss,lb_loss)
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=total_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, query_embeds, past=None, attention_mask=None, **model_kwargs
    ):
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)
        query_mask = input_ids.new_ones(query_embeds.shape[:-1])
        attention_mask = torch.cat([query_mask, attention_mask], dim=-1)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "query_embeds": query_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past,
            "encoder_hidden_states": model_kwargs.get("encoder_hidden_states", None),
            "encoder_attention_mask": model_kwargs.get("encoder_attention_mask", None),
            "is_decoder": True,
        }

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx) for past_state in layer_past
                ),
            )
        return reordered_past


class BertForMaskedLM(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            head_mask=None,
            query_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            return_logits=False,
            is_decoder=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
        )

        if query_embeds is not None:
            sequence_output = outputs[0][:, query_embeds.shape[1]:, :]
        prediction_scores = self.cls(sequence_output)

        if return_logits:
            return prediction_scores

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def build_qformer(num_query_token, vision_width,  ## 查询token的数量 # 视觉特征的维度
                  qformer_hidden_dropout_prob=0.1,  # # 隐藏层dropout概率
                  qformer_attention_probs_dropout_prob=0.1,  ## 注意力层dropout概率
                  qformer_drop_path_rate=0.,  ## drop path的比率
                  bert_type="bert-base-uncased" , ## 使用的bert基础模型
                  num_shared_heads=None  # 新增参数
                  ):
    encoder_config = BertConfig.from_pretrained(bert_type)
    encoder_config.encoder_width = vision_width
    # insert cross-attention layer every other block
    encoder_config.add_cross_attention = True  ## 启用交叉注意力层
    encoder_config.cross_attention_freq = 2  ## 每隔一个block插入一个交叉注意力层
    encoder_config.query_length = num_query_token
    encoder_config.hidden_dropout_prob = qformer_hidden_dropout_prob  ## 隐藏层dropout
    encoder_config.attention_probs_dropout_prob = qformer_attention_probs_dropout_prob  ## 注意力层dropout
    encoder_config.drop_path_list = [x.item() for x in torch.linspace(0, qformer_drop_path_rate,
                                                                      encoder_config.num_hidden_layers)]  ## 线性增加的drop path率
    # MoH相关配置
    if num_shared_heads is None:
        num_shared_heads = encoder_config.num_attention_heads // 4  # 默认25%的头作为共享头
        num_shared_heads = 8
    encoder_config.num_shared_heads = num_shared_heads
    encoder_config.use_moh = True  # 启用MoH
    encoder_config.moh_top_k = 2

    encoder_config.remoe_lambda = 1e-8
    encoder_config.remoe_alpha=1.2
    encoder_config.k = 0.8
    encoder_config.alignment_weight =0.5
    # 路由器输出 alpha_1, alpha_2 的投影
    encoder_config.moh_alpha_proj_size = 2
    # load balance loss 的加权系数
    encoder_config.moh_load_balance_weight = 1  # 你可以自己定义
    encoder_config.moh_start_layer = 6

    logger.info(f"Drop_path:{encoder_config.drop_path_list}")
    logger.info(encoder_config)

    Qformer = BertLMHeadModel(encoder_config)  ## 创建BERT语言模型
    query_tokens = nn.Parameter(
        torch.zeros(1, num_query_token, encoder_config.hidden_size)
    )  ## 初始化查询tokens
    query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)  ## 使用正态分布初始化query_tokens
    return Qformer, query_tokens


