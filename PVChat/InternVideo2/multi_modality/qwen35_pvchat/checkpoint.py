"""只保存PVChat新增参数的轻量checkpoint。"""

from __future__ import annotations

import json
from pathlib import Path

import torch


TRAINABLE_STATE_NAME = "pvchat_trainable.pt"
METADATA_NAME = "pvchat_config.json"
OPTIMIZER_STATE_NAME = "optimizer.pt"


def save_trainable_state(model, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    torch.save(state, path)
    return len(state)


def load_trainable_state(model, path: str | Path):
    path = Path(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint含有当前模型不认识的参数: {incompatible.unexpected_keys}")
    return {
        "loaded_keys": len(state),
        "missing_keys": incompatible.missing_keys,
    }


def save_checkpoint(directory, model, processor, metadata: dict, optimizer=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    count = save_trainable_state(model, directory / TRAINABLE_STATE_NAME)
    processor.save_pretrained(directory)
    payload = dict(metadata)
    payload["num_saved_trainable_tensors"] = count
    (directory / METADATA_NAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if optimizer is not None:
        torch.save(optimizer.state_dict(), directory / OPTIMIZER_STATE_NAME)
    return directory


def load_optimizer_state(optimizer, directory: str | Path):
    """恢复AdamW状态；参数组不匹配时让PyTorch直接给出明确错误。"""

    path = Path(directory) / OPTIMIZER_STATE_NAME
    if not path.exists():
        raise FileNotFoundError(f"找不到optimizer状态，无法真正续训: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state)
    return {
        "path": str(path.resolve()),
        "state_entries": len(state.get("state", {})),
        "param_groups": len(state.get("param_groups", [])),
    }


def load_checkpoint_metadata(directory: str | Path) -> dict:
    path = Path(directory) / METADATA_NAME
    if not path.exists():
        raise FileNotFoundError(f"找不到Qwen3.5 PVChat checkpoint元数据: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
