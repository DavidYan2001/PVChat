"""基于Qwen3.5视频语言骨干的PVChat组件。"""

from .remoh_attention import (
    DEFAULT_REMOH_LAYER_INDICES,
    DEFAULT_ROUTED_HEAD_INDICES,
    Qwen35ReMoHAttention,
    Qwen35ReMoHRouter,
    patch_qwen35_remoh_layers,
)

__all__ = [
    "DEFAULT_REMOH_LAYER_INDICES",
    "DEFAULT_ROUTED_HEAD_INDICES",
    "Qwen35ReMoHAttention",
    "Qwen35ReMoHRouter",
    "patch_qwen35_remoh_layers",
]
