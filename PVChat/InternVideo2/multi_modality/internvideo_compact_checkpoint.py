"""InternVideo2 紧凑checkpoint（pvchat_delta.pt）的加载与保存。

紧凑格式 `pvchat_internvideo_delta_v1` 只保存个性化状态：
- dense_state: Q-Former/ReMoH、LLM LoRA、personal_query_tokens 等完整小张量；
- row_state:   embedding / lm_head 中个性化 token 对应的稀疏行
               （每项为 {"indices": LongTensor[k], "values": Tensor[k, hidden]}）；
- vocab_size:  应用行更新前模型词表必须已扩展到的行数。

冻结的视觉编码器与基础语言模型由所有人物共享，不进 delta。加载顺序：
构建共享基础模型 → 按 tokenizer 扩展词表 → apply_compact_delta。
保存时沿用加载得到的键集（CompactDeltaSpec），确保 Stage 3 产物与
Stage 2 delta 结构一致、可被同一加载器消费。
"""

from __future__ import annotations

from pathlib import Path

import torch

DELTA_FORMAT = "pvchat_internvideo_delta_v1"
DELTA_FILENAME = "pvchat_delta.pt"
FULL_FILENAME = "pytorch_model.bin"

# 旧机器打包的delta在row_state里同时含peft包装路径和未包装别名
# （如 lm.lm_head.weight ≡ lm.base_model.model.lm_head.weight，索引和数值
# 完全相同）。当前构建的模型只暴露包装路径；若别名项与其包装项逐位相等，
# 跳过别名而不是报错。
_ROW_KEY_ALIASES = {
    "lm.lm_head.weight": "lm.base_model.model.lm_head.weight",
}


def _redundant_alias_keys(row_state, state):
    """返回可安全跳过的别名键：模型没有该键，但delta中它与包装路径项完全相同。"""

    skippable = set()
    for alias, canonical in _ROW_KEY_ALIASES.items():
        if alias not in row_state or alias in state:
            continue
        entry = row_state.get(alias)
        canon = row_state.get(canonical)
        if (
            canon is not None
            and canonical in state
            and torch.equal(entry["indices"], canon["indices"])
            and torch.equal(entry["values"], canon["values"])
        ):
            skippable.add(alias)
    return skippable


class CompactDeltaSpec:
    """记录一次 delta 加载覆盖了哪些键，供保存时提取同一集合。"""

    def __init__(self, dense_keys, row_indices, vocab_size):
        self.dense_keys = sorted(dense_keys)
        self.row_indices = {key: value.clone() for key, value in row_indices.items()}
        self.vocab_size = int(vocab_size)


def is_compact_checkpoint(checkpoint_dir) -> bool:
    return (Path(checkpoint_dir) / DELTA_FILENAME).is_file()


def apply_compact_delta(model, payload, source="") -> CompactDeltaSpec:
    """把 delta 应用到已构建、已扩词表的模型上；就地写入参数。"""

    if not isinstance(payload, dict) or payload.get("format") != DELTA_FORMAT:
        raise ValueError(
            f"不是{DELTA_FORMAT}格式的紧凑checkpoint: {source or type(payload)}"
        )
    vocab_size = int(payload["vocab_size"])
    dense_state = payload["dense_state"]
    row_state = payload["row_state"]
    state = model.state_dict()

    skipped_aliases = _redundant_alias_keys(row_state, state)
    if skipped_aliases:
        print(
            "[Checkpoint] Skipped redundant alias row keys (identical to wrapped path): "
            + ", ".join(sorted(skipped_aliases)),
            flush=True,
        )
        row_state = {k: v for k, v in row_state.items() if k not in skipped_aliases}

    missing = [key for key in dense_state if key not in state]
    missing += [key for key in row_state if key not in state]
    if missing:
        raise KeyError(
            f"delta包含当前模型没有的{len(missing)}个键（模型结构或配置不匹配）: "
            + ", ".join(missing[:5])
            + (" ..." if len(missing) > 5 else "")
            + (f"  来源: {source}" if source else "")
        )

    with torch.no_grad():
        for key, value in dense_state.items():
            target = state[key]
            if tuple(target.shape) != tuple(value.shape):
                raise ValueError(
                    f"delta键{key}形状不匹配: 模型{tuple(target.shape)} vs delta{tuple(value.shape)}"
                )
            target.copy_(value.to(dtype=target.dtype))

        row_indices = {}
        for key, entry in row_state.items():
            target = state[key]
            indices = entry["indices"].to(torch.long).reshape(-1)
            values = entry["values"]
            if target.shape[0] < vocab_size:
                raise ValueError(
                    f"{key}只有{target.shape[0]}行，小于delta要求的词表{vocab_size}——"
                    "应先按checkpoint tokenizer调用resize_token_embeddings再加载delta。"
                )
            if indices.numel() and int(indices.max()) >= target.shape[0]:
                raise ValueError(f"{key}的行索引{int(indices.max())}超出{target.shape[0]}行")
            target[indices.to(target.device)] = values.to(
                device=target.device, dtype=target.dtype
            )
            row_indices[key] = indices

    return CompactDeltaSpec(dense_state.keys(), row_indices, vocab_size)


def _load_full_state(model, state_path, torch_module):
    """旧式完整pytorch_model.bin的兼容加载（历史行为原样保留）。"""

    state_dict = torch_module.load(state_path, map_location="cpu")
    model_keys = set(model.state_dict().keys())
    filtered = {key: value for key, value in state_dict.items() if key in model_keys}
    dropped = sorted(set(state_dict) - model_keys)
    if dropped:
        print(
            "[Checkpoint] Ignored checkpoint keys not present in current model: "
            + ", ".join(dropped[:10])
            + (" ..." if len(dropped) > 10 else ""),
            flush=True,
        )
    model.load_state_dict(filtered, strict=True)


def load_checkpoint_state(model, checkpoint_dir, torch_module=torch):
    """按目录内容分派：紧凑delta或旧式完整state。

    返回 CompactDeltaSpec（紧凑格式）或 None（旧式完整格式）。
    """

    checkpoint_dir = Path(checkpoint_dir)
    delta_file = checkpoint_dir / DELTA_FILENAME
    if delta_file.is_file():
        payload = torch_module.load(delta_file, map_location="cpu", weights_only=False)
        spec = apply_compact_delta(model, payload, source=str(delta_file))
        print(
            f"[Checkpoint] Applied compact delta ({len(spec.dense_keys)} dense keys, "
            f"{len(spec.row_indices)} row entries) from {delta_file}",
            flush=True,
        )
        return spec
    full_file = checkpoint_dir / FULL_FILENAME
    if full_file.is_file():
        _load_full_state(model, full_file, torch_module)
        return None
    raise FileNotFoundError(
        f"{checkpoint_dir}中既没有{DELTA_FILENAME}也没有{FULL_FILENAME}，无法加载checkpoint。"
    )


def save_compact_checkpoint(
    model,
    spec: CompactDeltaSpec,
    output_dir,
    tokenizer=None,
    config=None,
    training_info=None,
):
    """按spec键集只保存可训练部分，产物结构与Stage 2 delta一致。"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()

    dense_state = {}
    for key in spec.dense_keys:
        if key not in state:
            raise KeyError(f"保存时模型缺少delta键{key}")
        dense_state[key] = state[key].detach().cpu().clone()

    row_state = {}
    for key, indices in spec.row_indices.items():
        if key not in state:
            raise KeyError(f"保存时模型缺少行更新键{key}")
        target = state[key]
        row_state[key] = {
            "indices": indices.detach().cpu().clone(),
            "values": target[indices].detach().cpu().clone(),
        }

    payload = {
        "format": DELTA_FORMAT,
        "vocab_size": spec.vocab_size,
        "dense_state": dense_state,
        "row_state": row_state,
    }
    torch.save(payload, output_dir / DELTA_FILENAME)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    if config is not None:
        config.save_pretrained(output_dir)
    if training_info is not None:
        info = dict(training_info)
        info["checkpoint_format"] = DELTA_FORMAT
        torch.save(info, output_dir / "training_info.bin")
    return output_dir / DELTA_FILENAME
