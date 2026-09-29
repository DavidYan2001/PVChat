"""Qwen3.5个性化token的轻量实现。

这里不直接训练完整的25万词表矩阵。完整矩阵的梯度会占用数GB显存，
但我们真正需要更新的只有人物token和16个细节token。因此使用两个很小的
可训练行矩阵覆盖输入embedding和输出lm_head中的对应行。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_person_token(person_token: str) -> str:
    """把Sheldon和<Sheldon>统一成<Sheldon>。"""

    value = str(person_token).strip()
    if not value:
        raise ValueError("person_token不能为空。")
    return value if value.startswith("<") and value.endswith(">") else f"<{value}>"


def build_personalized_tokens(person_token: str, num_detail_tokens: int = 16) -> list[str]:
    """返回人物token和<sks_token1>...<sks_token16>。"""

    if num_detail_tokens < 0:
        raise ValueError("num_detail_tokens不能小于0。")
    person_token = normalize_person_token(person_token)
    return [person_token] + [f"<sks_token{i}>" for i in range(1, num_detail_tokens + 1)]


def build_identity_prefix(question: str, personalized_tokens: Sequence[str]) -> str:
    """把个性化token放在问题前，让每条视频QA都能训练这些embedding。"""

    if not personalized_tokens:
        return question
    return "".join(personalized_tokens) + "\n" + question


class PersonalizedEmbedding(nn.Module):
    """冻结原始embedding，只替换指定token的输出。"""

    def __init__(
        self,
        base_embedding: nn.Module,
        token_ids: Sequence[int],
        initial_rows: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.base_embedding = base_embedding
        self.base_embedding.requires_grad_(False)
        token_ids = tuple(int(token_id) for token_id in token_ids)
        if not token_ids:
            raise ValueError("至少需要一个个性化token id。")
        self.register_buffer("token_ids", torch.tensor(token_ids, dtype=torch.long), persistent=True)

        if initial_rows is None:
            initial_rows = self.base_embedding.weight.detach()[list(token_ids)].clone()
        self.personal_rows = nn.Parameter(initial_rows.detach().clone())

    @property
    def weight(self):
        # 兼容Transformers中读取embedding.weight的工具代码。
        return self.base_embedding.weight

    @property
    def embedding_dim(self):
        return self.base_embedding.embedding_dim

    @property
    def num_embeddings(self):
        return self.base_embedding.num_embeddings

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        base_output = self.base_embedding(input_ids)
        token_ids = self.token_ids.to(input_ids.device)
        matches = input_ids.unsqueeze(-1).eq(token_ids)
        is_personal = matches.any(dim=-1)
        row_indices = matches.to(torch.int64).argmax(dim=-1)
        personal_output = F.embedding(row_indices, self.personal_rows)
        return torch.where(is_personal.unsqueeze(-1), personal_output, base_output)


class PersonalizedLMHead(nn.Module):
    """冻结原始lm_head，只替换个性化token对应的logit行。"""

    def __init__(
        self,
        base_lm_head: nn.Module,
        token_ids: Sequence[int],
        initial_rows: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.base_lm_head = base_lm_head
        self.base_lm_head.requires_grad_(False)
        token_ids = tuple(int(token_id) for token_id in token_ids)
        self.register_buffer("token_ids", torch.tensor(token_ids, dtype=torch.long), persistent=True)
        if initial_rows is None:
            initial_rows = self.base_lm_head.weight.detach()[list(token_ids)].clone()
        self.personal_rows = nn.Parameter(initial_rows.detach().clone())

    @property
    def weight(self):
        return self.base_lm_head.weight

    @property
    def in_features(self):
        return self.base_lm_head.in_features

    @property
    def out_features(self):
        return self.base_lm_head.out_features

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_logits = self.base_lm_head(hidden_states)
        personal_logits = F.linear(hidden_states, self.personal_rows)
        return torch.index_copy(
            base_logits,
            dim=-1,
            index=self.token_ids.to(base_logits.device),
            source=personal_logits,
        )


@dataclass
class PersonalizedTokenModules:
    tokens: list[str]
    token_ids: list[int]
    embedding: PersonalizedEmbedding
    lm_head: PersonalizedLMHead


def add_personalized_tokens(tokenizer, person_token: str, num_detail_tokens: int = 16):
    """向tokenizer加入普通token；普通token在decode时不会被自动删除。"""

    tokens = build_personalized_tokens(person_token, num_detail_tokens)
    tokenizer.add_tokens(tokens, special_tokens=False)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    if any(token_id is None or token_id < 0 for token_id in token_ids):
        raise RuntimeError("个性化token没有被tokenizer正确注册。")
    return tokens, [int(token_id) for token_id in token_ids]


def _initial_person_rows(model, tokenizer, person_token: str, token_ids: Sequence[int]):
    """用人物名字原有子词的均值初始化，避免新增token从完全随机向量开始。"""

    bare_name = normalize_person_token(person_token).strip("<>")
    source_ids = tokenizer.encode(bare_name, add_special_tokens=False)
    source_ids = source_ids or [getattr(tokenizer, "unk_token_id", 0)]
    input_weight = model.get_input_embeddings().weight.detach()
    output_weight = model.get_output_embeddings().weight.detach()
    source = torch.tensor(source_ids, dtype=torch.long, device=input_weight.device)
    input_row = input_weight.index_select(0, source).mean(dim=0, keepdim=True)
    output_row = output_weight.index_select(0, source.to(output_weight.device)).mean(dim=0, keepdim=True)
    return input_row.expand(len(token_ids), -1).clone(), output_row.expand(len(token_ids), -1).clone()


def install_personalized_token_modules(
    model,
    tokenizer,
    person_token: str,
    num_detail_tokens: int = 16,
) -> PersonalizedTokenModules:
    """扩展词表并安装只训练17行参数的输入/输出包装器。"""

    tokens, token_ids = add_personalized_tokens(tokenizer, person_token, num_detail_tokens)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    input_rows, output_rows = _initial_person_rows(model, tokenizer, person_token, token_ids)

    embedding = PersonalizedEmbedding(model.get_input_embeddings(), token_ids, input_rows)
    lm_head = PersonalizedLMHead(model.get_output_embeddings(), token_ids, output_rows)
    model.set_input_embeddings(embedding)
    model.set_output_embeddings(lm_head)
    return PersonalizedTokenModules(tokens, token_ids, embedding, lm_head)


def find_personalized_modules(model):
    """从PEFT/DDP包装后的模型中找到个性化输入和输出模块。"""

    embedding = None
    lm_head = None
    for module in model.modules():
        if isinstance(module, PersonalizedEmbedding):
            embedding = module
        elif isinstance(module, PersonalizedLMHead):
            lm_head = module
    if embedding is None or lm_head is None:
        raise RuntimeError("模型中没有安装完整的个性化token模块。")
    return embedding, lm_head
