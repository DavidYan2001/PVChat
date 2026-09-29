"""Shared Stage 3 policy-ablation runner for Qwen3.5 PVChat."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from pvchat_dataset_validation import describe_training_summary, validate_training_json_classes

from .adaptive_policy import (
    CATEGORIES,
    DynamicRolloutDecision,
    PersonalizedAdaptiveState,
    decide_dynamic_rollout,
)
from .checkpoint import (
    METADATA_NAME,
    TRAINABLE_STATE_NAME,
    load_checkpoint_metadata,
    load_optimizer_state,
    save_checkpoint,
)
from .constraint_anchored_dynamic_policy import (
    VALIDITY_INVALID,
    constraint_anchored_advantages,
    decide_constraint_anchored_rollout,
    score_constraint_anchored_answer,
)
from .data import (
    DynamicVideoBudget,
    collate_qwen35_features,
    encode_record,
    load_qa_records,
    video_overrides_from_token_budget,
)
from .distributed import barrier, move_batch_to_device, unwrap_model
from .evaluation import generation_runtime, run_distributed_evaluation, run_metrics, strip_thinking
from .grpo import build_completion_labels, reference_kl_loss, select_causal_positions, selected_token_logprobs
from .identity_constrained_policy import (
    decide_fixed_icd_rollout,
    decide_vector_icd_rollout,
    identity_constrained_advantages,
    score_icd_answer,
)
from .identity_gated_dynamic_policy import (
    IDENTITY_CORRECT,
    IDENTITY_UNKNOWN,
    IDENTITY_WRONG,
    decide_identity_gated_rollout,
    identity_gated_advantages,
    score_identity_gated_answer,
)
from .modeling import build_optimizer, load_qwen35_pvchat_model, set_lora_dropout, trainable_parameter_summary
from .policy_objectives import (
    connected_zero_loss,
    decoupled_component_advantages,
    normalize_group_advantages,
    sequence_policy_loss,
    supervised_anchor_loss,
    token_policy_loss,
)
from .remoh_attention import remoh_generation_masks
from .remoh_losses import AdaptiveReMoHLoss
from .identity_presence import ABSENT, detect_identity_presence
from .rewards import clean_text, classify_qa_type, pa_component_weights, score_answer, score_pa_answer


MANIFEST_NAME = "experiment_manifest.json"
RUNTIME_STATE_NAME = "runtime_state.pt"
PHASES = ("training", "evaluation", "metrics")


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    loss_level: str
    advantage_alpha: float | None
    rollout_mode: str
    reward_mode: str
    component_alphas: Mapping[str, float]

    @property
    def is_dynamic(self) -> bool:
        return self.rollout_mode in {
            "dynamic",
            "vector_dynamic",
            "identity_dynamic",
            "constraint_dynamic",
        }

    @property
    def is_pa(self) -> bool:
        return self.reward_mode == "pa"

    @property
    def is_icd(self) -> bool:
        return self.reward_mode == "icd"

    @property
    def is_vector_dynamic(self) -> bool:
        return self.rollout_mode == "vector_dynamic"

    @property
    def is_identity_gated(self) -> bool:
        return self.reward_mode == "identity_gated"

    @property
    def is_constraint_anchored(self) -> bool:
        return self.reward_mode == "constraint_anchored"

    @property
    def is_buffered(self) -> bool:
        # buffered调度让窗口内第2组起 ratio!=1, clip/drift才在数学上生效;
        # *_buffered基线变体用于消融"锚定(buffer+drift)本身"的贡献。
        return self.name in {
            "ca_dynamic_gspo",
            "ca_dynamic_gspo_buffered",
            "token_grpo_buffered",
            "gspo_buffered",
            "dynamic_gspo_buffered",
        }


_PA_COMPONENT_ALPHAS = {
    "identity": 0.0,
    "action": 0.5,
    "clothing": 0.5,
    "location": 0.5,
    "emotion": 0.5,
    "open": 0.5,
}

ALGORITHM_REGISTRY = {
    "token_grpo": AlgorithmSpec("token_grpo", "token", 1.0, "fixed4", "legacy", {}),
    "token_grpo_buffered": AlgorithmSpec("token_grpo_buffered", "token", 1.0, "fixed4", "legacy", {}),
    "gspo": AlgorithmSpec("gspo", "sequence", 1.0, "fixed4", "legacy", {}),
    "gspo_buffered": AlgorithmSpec("gspo_buffered", "sequence", 1.0, "fixed4", "legacy", {}),
    "dr_gspo": AlgorithmSpec("dr_gspo", "sequence", 0.0, "fixed4", "legacy", {}),
    "dynamic_gspo": AlgorithmSpec("dynamic_gspo", "sequence", 1.0, "dynamic", "legacy", {}),
    "dynamic_gspo_buffered": AlgorithmSpec("dynamic_gspo_buffered", "sequence", 1.0, "dynamic", "legacy", {}),
    "pa_gspo": AlgorithmSpec(
        "pa_gspo",
        "sequence",
        None,
        "dynamic",
        "pa",
        _PA_COMPONENT_ALPHAS,
    ),
    "icd_gspo_v0": AlgorithmSpec(
        "icd_gspo_v0",
        "sequence",
        None,
        "fixed8",
        "icd",
        {},
    ),
    "icd_gspo_v1": AlgorithmSpec(
        "icd_gspo_v1",
        "sequence",
        None,
        "vector_dynamic",
        "icd",
        {},
    ),
    "ig_dynamic_gspo": AlgorithmSpec(
        "ig_dynamic_gspo",
        "sequence",
        1.0,
        "identity_dynamic",
        "identity_gated",
        {},
    ),
    "ca_dynamic_gspo": AlgorithmSpec(
        "ca_dynamic_gspo",
        "sequence",
        1.0,
        "constraint_dynamic",
        "constraint_anchored",
        {},
    ),
    "ca_dynamic_gspo_buffered": AlgorithmSpec(
        "ca_dynamic_gspo_buffered",
        "sequence",
        1.0,
        "constraint_dynamic",
        "constraint_anchored",
        {},
    ),
}


def get_algorithm_spec(name: str) -> AlgorithmSpec:
    try:
        return ALGORITHM_REGISTRY[str(name)]
    except KeyError as error:
        choices = ", ".join(ALGORITHM_REGISTRY)
        raise ValueError(f"unknown Stage 3 algorithm {name!r}; expected one of: {choices}") from error


def _identity_gate_states(component_rows: Sequence[Mapping[str, float]]) -> list[float]:
    states = []
    for row in component_rows:
        if "identity_gate" not in row:
            raise ValueError("identity-gated component row is missing identity_gate")
        states.append(float(row["identity_gate"]))
    return states


def build_epoch_slots(
    length: int,
    world_size: int,
    epoch_seed: int,
    max_steps_per_epoch: int = 0,
) -> list[list[int | None]]:
    """Shuffle once, stride by rank, then pad only the shorter rank tails."""

    length = int(length)
    world_size = int(world_size)
    max_steps_per_epoch = int(max_steps_per_epoch)
    if length < 0:
        raise ValueError("length must be non-negative")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if max_steps_per_epoch < 0:
        raise ValueError("max_steps_per_epoch must be non-negative")
    if length < world_size:
        raise ValueError(
            "exact-once DDP requires at least one real record per rank; "
            f"got {length} records for {world_size} ranks"
        )

    indices = list(range(length))
    random.Random(int(epoch_seed)).shuffle(indices)
    slots = [indices[rank::world_size] for rank in range(world_size)]
    max_length = max((len(rank_slots) for rank_slots in slots), default=0)
    padded = [rank_slots + [None] * (max_length - len(rank_slots)) for rank_slots in slots]
    if max_steps_per_epoch:
        padded = [rank_slots[:max_steps_per_epoch] for rank_slots in padded]
    return padded


def exact_once_epoch_slots(*args, **kwargs):
    """Compatibility alias for callers that prefer an explicit helper name."""

    return build_epoch_slots(*args, **kwargs)


def active_loss_scale(world_size: int, active_count: int, is_active: bool) -> float:
    world_size = int(world_size)
    active_count = int(active_count)
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not 0 <= active_count <= world_size:
        raise ValueError("active_count must be between zero and world_size")
    if not is_active:
        return 0.0
    if active_count == 0:
        raise ValueError("an active rank requires at least one active rank")
    return world_size / active_count


def candidate_chunk_plan(spec: AlgorithmSpec | str) -> tuple[int, ...]:
    spec = get_algorithm_spec(spec) if isinstance(spec, str) else spec
    if spec.rollout_mode == "fixed8":
        return (8,)
    if spec.is_vector_dynamic:
        return (4, 4)
    return (2, 2, 4) if spec.is_dynamic else (4,)


def candidate_chunk_size(
    spec: AlgorithmSpec | str,
    rewards: Sequence[float],
    component_rows: Sequence[Mapping[str, float]] | None = None,
    qa_type: str = "open",
    is_positive: bool = True,
    identity_threshold: float = 0.5,
    identity_margin: float = 1.0,
    soft_clip: float = 0.5,
    conflict_threshold: float = 0.25,
    ca_semantic_threshold: float = 0.60,
    ca_informative_margin: float = 0.10,
) -> int:
    """Return the next generation chunk size, or zero when sampling is complete."""

    spec = get_algorithm_spec(spec) if isinstance(spec, str) else spec
    rewards = list(rewards)
    if spec.is_constraint_anchored:
        if len(rewards) == 0:
            return 2
        if component_rows is None:
            raise ValueError("constraint-anchored rollout requires component_rows after generation")
        decision = decide_constraint_anchored_rollout(
            component_rows,
            semantic_threshold=ca_semantic_threshold,
            informative_margin=ca_informative_margin,
        )
        return max(0, decision.target_count - len(rewards)) if decision.needs_more else 0
    if spec.is_identity_gated:
        if len(rewards) == 0:
            return 2
        if component_rows is None:
            raise ValueError("identity-gated rollout requires component_rows after generation")
        decision = decide_identity_gated_rollout(rewards, _identity_gate_states(component_rows))
        return max(0, decision.target_count - len(rewards)) if decision.needs_more else 0
    if spec.rollout_mode == "fixed8":
        if len(rewards) == 0:
            return 8
        if len(rewards) == 8:
            return 0
        raise ValueError("fixed8 rollout must contain zero or eight rewards")
    if spec.is_vector_dynamic:
        if len(rewards) == 0:
            return 4
        if len(rewards) not in (4, 8):
            raise ValueError("vector-dynamic rollout must contain zero, four, or eight rewards")
        if component_rows is None:
            raise ValueError("vector-dynamic rollout requires component_rows after generation")
        decision = decide_vector_icd_rollout(
            component_rows,
            qa_type,
            is_positive,
            identity_threshold=identity_threshold,
            identity_margin=identity_margin,
            soft_clip=soft_clip,
            conflict_threshold=conflict_threshold,
        )
        return max(0, decision.target_count - len(rewards)) if decision.needs_more else 0
    if not spec.is_dynamic:
        if len(rewards) == 0:
            return 4
        if len(rewards) == 4:
            return 0
        raise ValueError("fixed4 rollout must contain zero or four rewards")
    if len(rewards) == 0:
        return 2
    decision = decide_dynamic_rollout(rewards)
    return max(0, decision.target_count - len(rewards)) if decision.needs_more else 0


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: str | Path) -> dict[str, str]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"fingerprint input does not exist: {resolved}")
    return {"path": str(resolved), "sha256": file_sha256(resolved)}


def collect_input_fingerprints(
    stage2_checkpoint: str | Path,
    train_json: str | Path,
    test_json: str | Path,
) -> dict[str, dict[str, str]]:
    stage2_checkpoint = Path(stage2_checkpoint).expanduser().resolve()
    return {
        "stage2_trainable": _fingerprint(stage2_checkpoint / TRAINABLE_STATE_NAME),
        "stage2_config": _fingerprint(stage2_checkpoint / METADATA_NAME),
        "train_json": _fingerprint(train_json),
        "test_json": _fingerprint(test_json),
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def fixed_experiment_args(args: Any) -> dict[str, Any]:
    values = vars(args) if hasattr(args, "__dict__") else dict(args)
    # Phase controls and judge routing can change when only evaluation/metrics
    # need to be resumed; they do not define the trained policy.
    excluded = {
        "algorithm",
        "no_resume",
        "output_dir",
        "skip_evaluation",
        "skip_metrics",
        "judge_backend",
        "judge_model",
        "judge_fallback_models",
        "api_num_workers",
        "local_judge_model_path",
        "local_judge_device",
        "local_judge_batch_size",
        "local_judge_es_items_per_prompt",
        "local_judge_dc_items_per_prompt",
        "local_judge_max_new_tokens",
        # 旧五种算法的manifest必须与新增ICD代码之前保持兼容。ICD构建
        # manifest/checkpoint时会在下面按算法显式加入这些参数。
        "identity_threshold",
        "identity_margin",
        "icd_soft_clip",
        "icd_conflict_threshold",
        "ig_identity_margin",
        "ig_soft_clip",
        "ca_semantic_threshold",
        "ca_informative_margin",
        "ca_sft_weight",
        "ca_fallback_sft_weight",
        "rollout_buffer_size",
        # 批量生成只影响墙钟时间与RNG消耗次序，不定义训练出的策略。
        "disable_batched_rollout",
        # 逐视频动态视觉预算：启用时在build_experiment_manifest里显式写入，
        # 未启用（None）时不写入，保持旧manifest可继续resume。
        "video_budget_per_10_seconds",
    }
    return {key: _json_value(value) for key, value in sorted(values.items()) if key not in excluded}


def build_experiment_manifest(
    algorithm: str,
    args: Any,
    stage2_checkpoint: str | Path,
    fingerprints: Mapping[str, Any],
) -> dict[str, Any]:
    spec = get_algorithm_spec(algorithm)
    fixed_args = fixed_experiment_args(args)
    budget_per_10s = getattr(args, "video_budget_per_10_seconds", None)
    if budget_per_10s:
        fixed_args["video_budget_per_10_seconds"] = int(budget_per_10s)
    if spec.is_icd:
        fixed_args.update(
            {
                "identity_threshold": float(args.identity_threshold),
                "identity_margin": float(args.identity_margin),
                "icd_soft_clip": float(args.icd_soft_clip),
                "icd_conflict_threshold": float(args.icd_conflict_threshold),
            }
        )
    if spec.is_identity_gated:
        fixed_args.update(
            {
                "ig_identity_margin": float(args.ig_identity_margin),
                "ig_soft_clip": float(args.ig_soft_clip),
            }
        )
    if spec.is_constraint_anchored:
        fixed_args.update(
            {
                "ca_semantic_threshold": float(args.ca_semantic_threshold),
                "ca_informative_margin": float(args.ca_informative_margin),
                "ca_sft_weight": float(args.ca_sft_weight),
                "ca_fallback_sft_weight": float(args.ca_fallback_sft_weight),
            }
        )
    if spec.is_buffered:
        fixed_args["rollout_buffer_size"] = int(getattr(args, "rollout_buffer_size", 4))
    return {
        "schema_version": 1,
        "algorithm": spec.name,
        "algorithm_spec": _json_value(asdict(spec)),
        "common_stage2_checkpoint": str(Path(stage2_checkpoint).expanduser().resolve()),
        "input_fingerprints": _json_value(fingerprints),
        "fixed_args": fixed_args,
    }


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(_json_value(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_name = handle.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name and Path(temporary_name).exists():
            Path(temporary_name).unlink()
    return path


def validate_or_write_manifest(path: str | Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(path)
    expected = _json_value(expected)
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        # Evaluation batching changes only wall-clock performance. Permit it to
        # change when resuming a trained/evaluated policy while retaining the
        # originally recorded value in the on-disk manifest.
        actual_for_resume = json.loads(json.dumps(actual))
        expected_for_resume = json.loads(json.dumps(expected))
        actual_for_resume.get("fixed_args", {}).pop("eval_batch_size", None)
        expected_for_resume.get("fixed_args", {}).pop("eval_batch_size", None)
        # A downloaded runtime bundle may be repacked under a new directory
        # (for example remaining23 -> remaining22) without changing any input
        # bytes.  The fingerprints below protect those inputs by SHA256, so
        # their absolute locations are relocation metadata, not policy state.
        actual_for_resume.pop("common_stage2_checkpoint", None)
        expected_for_resume.pop("common_stage2_checkpoint", None)
        for candidate in (actual_for_resume, expected_for_resume):
            for name in ("checkpoint_path", "train_json", "test_json"):
                candidate.get("fixed_args", {}).pop(name, None)
            for fingerprint in candidate.get("input_fingerprints", {}).values():
                if isinstance(fingerprint, dict):
                    fingerprint.pop("path", None)
        # The policy checkpoint depends on the Stage 2 checkpoint and training
        # JSON, but not on which held-out examples are evaluated afterwards.
        # Permit a test-set-only refresh while continuing to reject every
        # training-affecting change.  A refreshed test set invalidates only the
        # downstream evaluation/metrics markers; the training marker and its
        # checkpoint remain valid.
        actual_for_resume.get("input_fingerprints", {}).pop("test_json", None)
        expected_for_resume.get("input_fingerprints", {}).pop("test_json", None)
        if actual_for_resume != expected_for_resume:
            raise ValueError(
                "experiment manifest mismatch; refusing stale resume\n"
                f"expected={json.dumps(expected, sort_keys=True)}\n"
                f"actual={json.dumps(actual, sort_keys=True)}"
            )
        actual_test = actual.get("input_fingerprints", {}).get("test_json")
        expected_test = expected.get("input_fingerprints", {}).get("test_json")
        if actual_test != expected_test:
            atomic_write_json(path, expected)
            status_dir = path.parent / "status"
            for pattern in (
                "epoch_*.evaluation_complete.json",
                "epoch_*.metrics_complete.json",
            ):
                for marker in status_dir.glob(pattern):
                    marker.unlink(missing_ok=True)
            return expected
        return actual
    atomic_write_json(path, expected)
    return expected


def phase_marker_path(output_dir: str | Path, epoch: int, phase: str) -> Path:
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}; expected one of {PHASES}")
    return Path(output_dir) / "status" / f"epoch_{int(epoch)}.{phase}_complete.json"


def write_phase_marker(
    output_dir: str | Path,
    epoch: int,
    phase: str,
    payload: Mapping[str, Any] | None = None,
) -> Path:
    marker = {"epoch": int(epoch), "phase": phase}
    marker.update(dict(payload or {}))
    return atomic_write_json(phase_marker_path(output_dir, epoch, phase), marker)


@dataclass(frozen=True)
class EpochPhasePlan:
    run_training: bool
    run_evaluation: bool
    run_metrics: bool


def plan_epoch_phases(
    output_dir: str | Path,
    epoch: int,
    skip_evaluation: bool = False,
    skip_metrics: bool = False,
    resume: bool = True,
) -> EpochPhasePlan:
    if not resume:
        return EpochPhasePlan(True, not skip_evaluation, not skip_evaluation and not skip_metrics)
    training_done = phase_marker_path(output_dir, epoch, "training").is_file()
    evaluation_done = phase_marker_path(output_dir, epoch, "evaluation").is_file()
    metrics_done = phase_marker_path(output_dir, epoch, "metrics").is_file()
    evaluation_current = training_done and evaluation_done
    metrics_current = evaluation_current and metrics_done
    return EpochPhasePlan(
        run_training=not training_done,
        run_evaluation=not skip_evaluation and not evaluation_current,
        run_metrics=not skip_evaluation and not skip_metrics and not metrics_current,
    )


def build_checkpoint_metadata(
    base_metadata: Mapping[str, Any],
    spec: AlgorithmSpec | str,
    epoch: int,
    stage2_checkpoint: str | Path,
    parent_checkpoint: str | Path,
    args: Any,
    fingerprints: Mapping[str, Any],
) -> dict[str, Any]:
    spec = get_algorithm_spec(spec) if isinstance(spec, str) else spec
    metadata = dict(base_metadata)
    hyperparameters = fixed_experiment_args(args)
    hyperparameters.update(
        {
            "loss_level": spec.loss_level,
            "advantage_alpha": spec.advantage_alpha,
            "rollout_mode": spec.rollout_mode,
            "reward_mode": spec.reward_mode,
            "component_alphas": _json_value(spec.component_alphas),
        }
    )
    if spec.is_icd:
        hyperparameters.update(
            {
                "identity_threshold": float(args.identity_threshold),
                "identity_margin": float(args.identity_margin),
                "icd_soft_clip": float(args.icd_soft_clip),
                "icd_conflict_threshold": float(args.icd_conflict_threshold),
            }
        )
    if spec.is_identity_gated:
        hyperparameters.update(
            {
                "ig_identity_margin": float(args.ig_identity_margin),
                "ig_soft_clip": float(args.ig_soft_clip),
            }
        )
    if spec.is_constraint_anchored:
        hyperparameters.update(
            {
                "ca_semantic_threshold": float(args.ca_semantic_threshold),
                "ca_informative_margin": float(args.ca_informative_margin),
                "ca_sft_weight": float(args.ca_sft_weight),
                "ca_fallback_sft_weight": float(args.ca_fallback_sft_weight),
            }
        )
    if spec.is_buffered:
        hyperparameters["rollout_buffer_size"] = int(args.rollout_buffer_size)
    metadata.update(
        {
            "stage": 3,
            "algorithm": spec.name,
            "epoch": int(epoch),
            "grpo_epoch": int(epoch),
            "common_stage2_checkpoint": str(Path(stage2_checkpoint).expanduser().resolve()),
            "parent_checkpoint": str(Path(parent_checkpoint).expanduser().resolve()),
            "input_fingerprints": _json_value(fingerprints),
            "hyperparameters": hyperparameters,
        }
    )
    return metadata


def rollout_decision(
    spec: AlgorithmSpec | str,
    rewards: Sequence[float],
    component_rows: Sequence[Mapping[str, float]] | None = None,
    qa_type: str = "open",
    is_positive: bool = True,
    identity_threshold: float = 0.5,
    identity_margin: float = 1.0,
    soft_clip: float = 0.5,
    conflict_threshold: float = 0.25,
    ca_semantic_threshold: float = 0.60,
    ca_informative_margin: float = 0.10,
) -> DynamicRolloutDecision:
    spec = get_algorithm_spec(spec) if isinstance(spec, str) else spec
    rewards = [float(value) for value in rewards]
    if spec.is_constraint_anchored:
        if component_rows is None:
            raise ValueError("constraint-anchored rollout decision requires component_rows")
        return decide_constraint_anchored_rollout(
            component_rows,
            semantic_threshold=ca_semantic_threshold,
            informative_margin=ca_informative_margin,
        )
    if spec.is_identity_gated:
        if component_rows is None:
            raise ValueError("identity-gated rollout decision requires component_rows")
        return decide_identity_gated_rollout(rewards, _identity_gate_states(component_rows))
    if spec.is_icd:
        if component_rows is None:
            raise ValueError("ICD rollout decision requires component_rows")
        if spec.is_vector_dynamic:
            return decide_vector_icd_rollout(
                component_rows,
                qa_type,
                is_positive,
                identity_threshold=identity_threshold,
                identity_margin=identity_margin,
                soft_clip=soft_clip,
                conflict_threshold=conflict_threshold,
            )
        return decide_fixed_icd_rollout(
            component_rows,
            qa_type,
            is_positive,
            identity_threshold=identity_threshold,
            identity_margin=identity_margin,
            soft_clip=soft_clip,
        )
    if spec.is_dynamic:
        return decide_dynamic_rollout(rewards)
    if len(rewards) != 4:
        raise ValueError("fixed4 rollout requires exactly four rewards")
    return DynamicRolloutDecision(4, True, "fixed4")


def compute_group_advantages(
    spec: AlgorithmSpec | str,
    rewards: Sequence[float],
    component_rows: Sequence[Mapping[str, float]],
    qa_type: str,
    is_positive: bool,
    category_multiplier: float = 1.0,
    eps: float = 1e-6,
    identity_threshold: float = 0.5,
    identity_margin: float = 1.0,
    soft_clip: float = 0.5,
    ig_identity_margin: float = 1.0,
    ig_soft_clip: float = 0.25,
) -> list[float]:
    spec = get_algorithm_spec(spec) if isinstance(spec, str) else spec
    if spec.is_constraint_anchored:
        advantages = constraint_anchored_advantages(component_rows, eps=eps)
    elif spec.is_identity_gated:
        advantages = identity_gated_advantages(
            rewards,
            _identity_gate_states(component_rows),
            identity_margin=ig_identity_margin,
            soft_clip=ig_soft_clip,
            eps=eps,
        )
    elif spec.is_icd:
        advantages = identity_constrained_advantages(
            component_rows,
            qa_type,
            is_positive,
            identity_threshold=identity_threshold,
            identity_margin=identity_margin,
            soft_clip=soft_clip,
        )
    elif spec.is_pa:
        alpha = spec.component_alphas.get(qa_type, 0.5)
        advantages = decoupled_component_advantages(
            component_rows,
            pa_component_weights(qa_type, bool(is_positive)),
            alpha=alpha,
            eps=eps,
        )
    else:
        advantages = normalize_group_advantages(
            rewards,
            alpha=float(spec.advantage_alpha),
            eps=eps,
        )
    multiplier = float(category_multiplier)
    return [float(value) * multiplier for value in advantages]


def parent_checkpoint_for_epoch(
    output_dir: str | Path,
    stage2_checkpoint: str | Path,
    epoch: int,
) -> Path:
    epoch = int(epoch)
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    if epoch == 1:
        return Path(stage2_checkpoint).expanduser()
    return Path(output_dir).expanduser() / "checkpoints" / f"epoch_{epoch - 1}"


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            for row in rows:
                handle.write(json.dumps(_json_value(row), ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_name = handle.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name and Path(temporary_name).exists():
            Path(temporary_name).unlink()
    return path


def merge_rollout_shards(
    output_dir: str | Path,
    epoch: int,
    world_size: int,
    expected_count: int,
    expected_indices: set[int] | None = None,
) -> Path:
    shard_dir = Path(output_dir) / "rollouts" / f"epoch_{int(epoch)}"
    rows = []
    seen = set()
    for rank in range(int(world_size)):
        shard = shard_dir / f"rank_{rank}.jsonl"
        if not shard.is_file():
            raise FileNotFoundError(f"missing rollout shard: {shard}")
        for line_number, line in enumerate(shard.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            group_id = item.get("group_id")
            if not group_id:
                raise ValueError(f"missing group_id in {shard}:{line_number}")
            if group_id in seen:
                raise ValueError(f"duplicate group_id in rollout shards: {group_id}")
            seen.add(group_id)
            rows.append(item)
    if len(rows) != int(expected_count):
        raise ValueError(
            f"merged real group count mismatch: expected {int(expected_count)}, got {len(rows)}"
        )
    if expected_indices is not None:
        actual_indices = [int(row.get("flat_index", -1)) for row in rows]
        if len(actual_indices) != len(set(actual_indices)) or set(actual_indices) != set(expected_indices):
            raise ValueError(
                "flat_index coverage mismatch: "
                f"expected={sorted(expected_indices)}, actual={sorted(actual_indices)}"
            )
    return _atomic_write_jsonl(shard_dir / "rollouts.jsonl", rows)


def manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    encoded = json.dumps(_json_value(manifest), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def aggregate_epoch_stats(
    rank_stats: Sequence[Mapping[str, Any]],
    spec: AlgorithmSpec | str,
    epoch: int,
    input_fingerprint: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    spec = get_algorithm_spec(spec) if isinstance(spec, str) else spec
    integer_fields = (
        "real_groups",
        "updates",
        "skipped",
        "zero_dispersion",
        "candidate_total",
        "reward_count",
        "identity_feasible_candidates",
        "identity_candidate_count",
        "expanded_groups",
        "ig_identity_correct_candidates",
        "ig_identity_wrong_candidates",
        "ig_identity_unknown_candidates",
        "ig_gated_groups",
        "ig_identity_expanded_groups",
        "ca_fallback_groups",
        "ca_anchored_groups",
        "ca_degenerate_candidates",
        "ca_identity_wrong_candidates",
    )
    totals = {
        field: sum(int(shard.get(field, 0)) for shard in rank_stats)
        for field in integer_fields
    }
    reward_sum = sum(float(shard.get("reward_sum", 0.0)) for shard in rank_stats)
    real_groups = totals["real_groups"]
    reward_count = totals["reward_count"]
    result = {
        "algorithm": spec.name,
        "epoch": int(epoch),
        "input_fingerprint": str(input_fingerprint),
        "real_groups": real_groups,
        "updates": totals["updates"],
        "skipped": totals["skipped"],
        "zero_dispersion": totals["zero_dispersion"],
        "candidate_total": totals["candidate_total"],
        "candidate_mean": totals["candidate_total"] / real_groups if real_groups else 0.0,
        "reward_mean": reward_sum / reward_count if reward_count else 0.0,
        "elapsed_seconds": float(elapsed_seconds),
    }
    identity_count = totals["identity_candidate_count"]
    result["identity_feasibility_rate"] = (
        totals["identity_feasible_candidates"] / identity_count if identity_count else None
    )
    result["expansion_rate"] = (
        totals["expanded_groups"] / real_groups if real_groups and spec.is_vector_dynamic else None
    )
    if spec.is_identity_gated:
        result["identity_gate_counts"] = {
            "correct": totals["ig_identity_correct_candidates"],
            "wrong": totals["ig_identity_wrong_candidates"],
            "unknown": totals["ig_identity_unknown_candidates"],
        }
        result["identity_gate_rate"] = (
            totals["ig_gated_groups"] / real_groups if real_groups else 0.0
        )
        result["identity_expansion_rate"] = (
            totals["ig_identity_expanded_groups"] / real_groups if real_groups else 0.0
        )
    if spec.is_constraint_anchored:
        result.update(
            {
                "fallback_groups": totals["ca_fallback_groups"],
                "anchored_groups": totals["ca_anchored_groups"],
                "degenerate_candidates": totals["ca_degenerate_candidates"],
                "identity_wrong_candidates": totals["ca_identity_wrong_candidates"],
            }
        )
    return result


def _atomic_torch_save(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
        torch.save(dict(payload), temporary_name)
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name and Path(temporary_name).exists():
            Path(temporary_name).unlink()
    return path


def save_adaptive_state_artifact(
    path: str | Path,
    algorithm: str,
    epoch: int,
    pa_state: PersonalizedAdaptiveState | None,
) -> Path:
    """Save the epoch-level adaptive state independently from the checkpoint."""

    return _atomic_torch_save(
        path,
        {
            "algorithm": str(algorithm),
            "epoch": int(epoch),
            "pa_state": pa_state.state_dict() if pa_state is not None else None,
        },
    )


def load_adaptive_state_artifact(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    state = payload.get("pa_state")
    return {
        **payload,
        "pa_state": PersonalizedAdaptiveState.from_state_dict(state) if state is not None else None,
    }


def save_runtime_state(
    path: str | Path,
    remoh_loss: AdaptiveReMoHLoss,
    pa_state: PersonalizedAdaptiveState | None,
) -> Path:
    payload = {
        "remoh_loss": remoh_loss.state_dict(),
        "pa_state": pa_state.state_dict() if pa_state is not None else None,
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    return _atomic_torch_save(path, payload)


def load_runtime_state(
    path: str | Path,
    remoh_loss: AdaptiveReMoHLoss,
    use_pa: bool,
    restore_rng: bool = True,
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing Stage 3 runtime state: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    remoh_loss.load_state_dict(payload["remoh_loss"])
    pa_payload = payload.get("pa_state")
    pa_state = PersonalizedAdaptiveState.from_state_dict(pa_payload) if use_pa and pa_payload else None
    if use_pa and pa_state is None:
        raise ValueError(f"PA checkpoint has no adaptive state: {path}")
    if restore_rng:
        random.setstate(payload["python_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and payload.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    return {**payload, "pa_state": pa_state}


def _epoch_seed(seed: int, epoch: int) -> int:
    return int(seed) + int(epoch) - 1


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _repeat_prompt_feature(feature: Mapping[str, Any], repeats: int, pad_token_id: int) -> dict:
    batch = collate_qwen35_features([feature], pad_token_id)
    for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
        batch[key] = batch[key].repeat(int(repeats), 1)
    batch["pixel_values_videos"] = batch["pixel_values_videos"].repeat(int(repeats), 1)
    batch["video_grid_thw"] = batch["video_grid_thw"].repeat(int(repeats), 1)
    batch["records"] = batch["records"] * int(repeats)
    return batch


def _forward_inputs(batch: Mapping[str, Any], include_remoh_masks: bool = True) -> dict:
    values = {key: value for key, value in batch.items() if key not in ("records", "labels")}
    if include_remoh_masks:
        mm_types = values["mm_token_type_ids"]
        values["pvchat_video_token_mask"] = mm_types.eq(2)
        values["pvchat_text_token_mask"] = mm_types.eq(0)
    return values


def _stop_token_ids(processor) -> list[int]:
    values = {processor.tokenizer.eos_token_id}
    values.add(processor.tokenizer.convert_tokens_to_ids("<|im_end|>"))
    return [int(value) for value in values if value is not None and int(value) >= 0]


def _trim_sequence(
    sequence: torch.Tensor,
    prompt_length: int,
    stop_token_ids: Sequence[int],
    pad_token_id: int | None,
) -> torch.Tensor:
    stop_ids = {int(value) for value in stop_token_ids}
    end = int(sequence.shape[0])
    for index in range(int(prompt_length), end):
        token = int(sequence[index])
        if token in stop_ids:
            end = index + 1
            break
    else:
        if pad_token_id is not None:
            while end > prompt_length and int(sequence[end - 1]) == int(pad_token_id):
                end -= 1
    return sequence[:end]


def build_gold_completion_sequence(
    prompt_input_ids: torch.Tensor,
    tokenizer,
    gold_answer: str,
) -> torch.Tensor:
    """Append the non-thinking gold answer to an already encoded video prompt."""

    if prompt_input_ids.ndim != 1:
        raise ValueError("prompt_input_ids must be one-dimensional")
    completion_ids = tokenizer.encode(
        f"{clean_text(gold_answer)}<|im_end|>\n",
        add_special_tokens=False,
    )
    if not completion_ids:
        raise ValueError("gold answer produced no completion tokens")
    completion = torch.tensor(
        completion_ids,
        dtype=prompt_input_ids.dtype,
        device=prompt_input_ids.device,
    )
    return torch.cat([prompt_input_ids, completion], dim=0)


def anchor_weight_for_decision(
    decision,
    regular_weight: float,
    fallback_weight: float,
) -> float:
    """Choose zero, regular, or fallback GT-anchor weight for one group."""

    if not bool(decision.should_update):
        return 0.0
    if bool(getattr(decision, "use_sft_fallback", False)):
        return float(fallback_weight)
    return float(regular_weight)


@torch.no_grad()
def _generation_sampling_kwargs(args, greedy: bool) -> dict:
    """greedy=True时用确定性解码(锚候选), 否则按args的采样配置。"""

    if greedy:
        return {"do_sample": False}
    return {
        "do_sample": True,
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
    }


def _generate_candidate_chunk(
    base_model,
    processor,
    feature: Mapping[str, Any],
    chunk_size: int,
    args,
    context,
    greedy: bool = False,
) -> tuple[list[torch.Tensor], list[str]]:
    batch = _repeat_prompt_feature(feature, chunk_size, processor.tokenizer.pad_token_id)
    batch = move_batch_to_device(batch, context.device)
    prompt_length = int(batch["input_ids"].shape[1])
    mm_types = batch["mm_token_type_ids"]
    with remoh_generation_masks(base_model, mm_types.eq(2), mm_types.eq(0)):
        generated = base_model.generate(
            **_forward_inputs(batch, include_remoh_masks=False),
            max_new_tokens=int(args.max_new_tokens),
            **_generation_sampling_kwargs(args, greedy),
            num_beams=1,
            use_cache=True,
        )

    stop_ids = _stop_token_ids(processor)
    sequences = [
        _trim_sequence(row, prompt_length, stop_ids, processor.tokenizer.pad_token_id)
        for row in generated
    ]
    completion_ids = [row[prompt_length:] for row in sequences]
    decoded = processor.batch_decode(
        completion_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    answers = [
        strip_thinking(text.replace("<|im_end|>", "").replace("<|endoftext|>", ""))
        for text in decoded
    ]
    return sequences, answers


def _pad_candidate_sequences(
    sequences: Sequence[torch.Tensor],
    pad_token_id: int,
) -> tuple[torch.Tensor, list[int]]:
    if not sequences:
        raise ValueError("at least one generated candidate is required")
    lengths = [int(sequence.shape[0]) for sequence in sequences]
    padded = sequences[0].new_full((len(sequences), max(lengths)), int(pad_token_id))
    for row, sequence in enumerate(sequences):
        padded[row, : sequence.shape[0]] = sequence
    return padded, lengths


@torch.no_grad()
def _build_score_batch_and_old_logprobs(
    base_model,
    processor,
    feature: Mapping[str, Any],
    sequences: Sequence[torch.Tensor],
    context,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    count = len(sequences)
    batch = _repeat_prompt_feature(feature, count, processor.tokenizer.pad_token_id)
    batch = move_batch_to_device(batch, context.device)
    prompt_length = int(batch["input_ids"].shape[1])
    padded, lengths = _pad_candidate_sequences(sequences, processor.tokenizer.pad_token_id)
    padded = padded.to(context.device)
    sequence_length = int(padded.shape[1])

    labels = build_completion_labels(padded, prompt_length, _stop_token_ids(processor))
    attention_mask = padded.new_zeros((count, sequence_length), dtype=torch.long)
    attention_mask[:, :prompt_length] = batch["attention_mask"]
    mm_types = padded.new_zeros((count, sequence_length), dtype=batch["mm_token_type_ids"].dtype)
    mm_types[:, :prompt_length] = batch["mm_token_type_ids"]
    for row, length in enumerate(lengths):
        attention_mask[row, prompt_length:length] = 1
        labels[row, length:] = -100

    score_batch = {
        "input_ids": padded,
        "attention_mask": attention_mask,
        "mm_token_type_ids": mm_types,
        "pixel_values_videos": batch["pixel_values_videos"],
        "video_grid_thw": batch["video_grid_thw"],
        "labels": labels,
        "records": batch["records"],
    }
    positions, targets = select_causal_positions(labels)
    old_inputs = _forward_inputs(score_batch)
    old_inputs["logits_to_keep"] = positions
    old_outputs = base_model(**old_inputs)
    old_logprobs, token_mask = selected_token_logprobs(old_outputs.logits, targets)
    return score_batch, old_logprobs.detach(), token_mask


def _adaptive_category(qa_type: str) -> str:
    return qa_type if qa_type in CATEGORIES else "action"


@dataclass
class SampledGroup:
    score_batch: dict
    old_logprobs: torch.Tensor
    token_mask: torch.Tensor
    answers: list[str]
    scored: list[Any]
    advantages: list[float]
    decision: Any
    qa_type: str
    adaptive_category: str
    category_multiplier: float
    algorithm: str
    decision_trace: tuple[str, ...]
    identity_gated: bool
    identity_expanded: bool
    candidate_count: int
    anchor_weight: float


def execute_rollout_schedule(
    spec: AlgorithmSpec | str,
    slots,
    rollout_buffer_size: int,
    collect_slot,
    update_slot,
    collect_slots=None,
) -> None:
    """Collect a fixed-policy rollout window before buffered policy updates.

    When collect_slots is given, each buffered window is collected through it in
    one call (enabling cross-group batched generation); otherwise every slot is
    collected individually. Updates always run per slot, in order — the window
    is collected entirely under the pre-update policy either way.
    """

    spec = get_algorithm_spec(spec) if isinstance(spec, str) else spec
    if not spec.is_buffered:
        for slot in slots:
            update_slot(collect_slot(slot))
        return

    rollout_buffer_size = int(rollout_buffer_size)
    if rollout_buffer_size <= 1:
        raise ValueError("buffered rollout schedule requires rollout_buffer_size greater than one")

    def flush(window) -> None:
        if collect_slots is not None:
            prepared = collect_slots(list(window))
            if len(prepared) != len(window):
                raise RuntimeError("collect_slots must return one prepared item per slot")
        else:
            prepared = [collect_slot(slot) for slot in window]
        for item in prepared:
            update_slot(item)

    window = []
    for slot in slots:
        window.append(slot)
        if len(window) == rollout_buffer_size:
            flush(window)
            window = []
    if window:
        flush(window)


@torch.no_grad()
def _scorer_for_spec(spec: AlgorithmSpec):
    if spec.is_constraint_anchored:
        return score_constraint_anchored_answer
    if spec.is_identity_gated:
        return score_identity_gated_answer
    if spec.is_icd:
        return score_icd_answer
    return score_pa_answer if spec.is_pa else score_answer


def _next_chunk_size(spec: AlgorithmSpec, scored, qa_type: str, is_positive: bool, args) -> int:
    return candidate_chunk_size(
        spec,
        [item.reward for item in scored],
        component_rows=[item.components for item in scored] if scored else None,
        qa_type=qa_type,
        is_positive=is_positive,
        identity_threshold=float(args.identity_threshold),
        identity_margin=float(args.identity_margin),
        soft_clip=float(args.icd_soft_clip),
        conflict_threshold=float(args.icd_conflict_threshold),
        ca_semantic_threshold=float(getattr(args, "ca_semantic_threshold", 0.60)),
        ca_informative_margin=float(getattr(args, "ca_informative_margin", 0.10)),
    )


def _new_group_state(feature: Mapping[str, Any], record, personalized_token: str, spec: AlgorithmSpec, pa_state) -> dict:
    qa_type = classify_qa_type(record.question, record.is_special)
    adaptive_category = _adaptive_category(qa_type)
    category_multiplier = (
        pa_state.multiplier(personalized_token, adaptive_category)
        if spec.is_pa and pa_state is not None
        else 1.0
    )
    return {
        "feature": feature,
        "record": record,
        "personalized_token": personalized_token,
        "qa_type": qa_type,
        "adaptive_category": adaptive_category,
        "category_multiplier": category_multiplier,
        "sequences": [],
        "answers": [],
        "scored": [],
        "decision_trace": [],
    }


def _absorb_candidate_chunk(state: dict, chunk_sequences, chunk_answers, spec: AlgorithmSpec, args) -> None:
    record = state["record"]
    scorer = _scorer_for_spec(spec)
    state["sequences"].extend(chunk_sequences)
    state["answers"].extend(chunk_answers)
    state["scored"].extend(
        scorer(
            record.question,
            record.answer,
            answer,
            state["personalized_token"],
            record.is_positive,
            state["qa_type"],
        )
        for answer in chunk_answers
    )
    if spec.is_identity_gated or spec.is_constraint_anchored:
        current_decision = rollout_decision(
            spec,
            [item.reward for item in state["scored"]],
            component_rows=[item.components for item in state["scored"]],
            ca_semantic_threshold=float(getattr(args, "ca_semantic_threshold", 0.60)),
            ca_informative_margin=float(getattr(args, "ca_informative_margin", 0.10)),
        )
        state["decision_trace"].append(current_decision.reason)


def _finalize_group(base_model, processor, state: dict, spec: AlgorithmSpec, args, context) -> SampledGroup:
    record = state["record"]
    feature = state["feature"]
    scored = state["scored"]
    sequences = state["sequences"]
    decision_trace = state["decision_trace"]
    decision = rollout_decision(
        spec,
        [item.reward for item in scored],
        component_rows=(
            [item.components for item in scored]
            if spec.is_icd or spec.is_identity_gated or spec.is_constraint_anchored
            else None
        ),
        qa_type=state["qa_type"],
        is_positive=record.is_positive,
        identity_threshold=float(args.identity_threshold),
        identity_margin=float(args.identity_margin),
        soft_clip=float(args.icd_soft_clip),
        conflict_threshold=float(args.icd_conflict_threshold),
        ca_semantic_threshold=float(getattr(args, "ca_semantic_threshold", 0.60)),
        ca_informative_margin=float(getattr(args, "ca_informative_margin", 0.10)),
    )
    candidate_count = len(sequences)
    anchor_weight = (
        anchor_weight_for_decision(
            decision,
            float(getattr(args, "ca_sft_weight", 0.05)),
            float(getattr(args, "ca_fallback_sft_weight", 0.20)),
        )
        if spec.is_constraint_anchored
        else 0.0
    )
    if anchor_weight > 0.0:
        sequences.append(
            build_gold_completion_sequence(
                feature["input_ids"],
                processor.tokenizer,
                record.answer,
            )
        )
    score_batch, all_old_logprobs, all_token_mask = _build_score_batch_and_old_logprobs(
        base_model,
        processor,
        feature,
        sequences,
        context,
    )
    old_logprobs = all_old_logprobs[:candidate_count]
    token_mask = all_token_mask[:candidate_count]

    advantages = compute_group_advantages(
        spec,
        [item.reward for item in scored],
        [item.components for item in scored],
        qa_type=state["qa_type"],
        is_positive=record.is_positive,
        category_multiplier=state["category_multiplier"],
        eps=float(args.advantage_eps),
        identity_threshold=float(args.identity_threshold),
        identity_margin=float(args.identity_margin),
        soft_clip=float(args.icd_soft_clip),
        ig_identity_margin=float(getattr(args, "ig_identity_margin", 1.0)),
        ig_soft_clip=float(getattr(args, "ig_soft_clip", 0.25)),
    )
    # 负样本组加权: gold为"不在/拒答"的组按negative_group_weight放大advantage,
    # 对冲正样本组更新次数上的压倒性优势, 保住拒答行为。组内标准化已完成,
    # 此处缩放等价于放大该组的policy loss权重。
    negative_weight = float(getattr(args, "negative_group_weight", 1.0))
    if negative_weight != 1.0 and detect_identity_presence(
        record.answer, state["personalized_token"]
    ) == ABSENT:
        advantages = [value * negative_weight for value in advantages]
    return SampledGroup(
        score_batch=score_batch,
        old_logprobs=old_logprobs,
        token_mask=token_mask,
        answers=state["answers"],
        scored=scored,
        advantages=advantages,
        decision=decision,
        qa_type=state["qa_type"],
        adaptive_category=state["adaptive_category"],
        category_multiplier=state["category_multiplier"],
        algorithm=spec.name,
        decision_trace=tuple(decision_trace),
        identity_gated=any(reason.startswith("identity_") for reason in decision_trace),
        identity_expanded="identity_expand_no_correct" in decision_trace,
        candidate_count=candidate_count,
        anchor_weight=anchor_weight,
    )


def sample_experiment_group(
    model,
    processor,
    feature: Mapping[str, Any],
    record,
    personalized_token: str,
    spec: AlgorithmSpec | str,
    pa_state: PersonalizedAdaptiveState | None,
    args,
    context,
) -> SampledGroup:
    """Generate chunks first, then pad and score the final candidate set once."""

    spec = get_algorithm_spec(spec) if isinstance(spec, str) else spec
    state = _new_group_state(feature, record, personalized_token, spec, pa_state)
    with generation_runtime(model) as base_model:
        while True:
            chunk_size = _next_chunk_size(spec, state["scored"], state["qa_type"], record.is_positive, args)
            if chunk_size == 0:
                break
            chunk_sequences: list = []
            chunk_answers: list = []
            # greedy锚候选: 每组第一个chunk的第一个候选用确定性解码
            # (= stage2的greedy行为), 保证组内始终存在可强化的高质量样本。
            if bool(getattr(args, "greedy_anchor_candidate", False)) and not state["scored"]:
                anchor_sequences, anchor_answers = _generate_candidate_chunk(
                    base_model, processor, feature, 1, args, context, greedy=True
                )
                chunk_sequences.extend(anchor_sequences)
                chunk_answers.extend(anchor_answers)
                chunk_size -= 1
            if chunk_size > 0:
                sampled_sequences, sampled_answers = _generate_candidate_chunk(
                    base_model,
                    processor,
                    feature,
                    chunk_size,
                    args,
                    context,
                )
                chunk_sequences.extend(sampled_sequences)
                chunk_answers.extend(sampled_answers)
            _absorb_candidate_chunk(state, chunk_sequences, chunk_answers, spec, args)
        return _finalize_group(base_model, processor, state, spec, args, context)


@torch.no_grad()
def _generate_candidate_chunks_multi(base_model, processor, requests, args, context, greedy: bool = False) -> dict:
    """Generate several groups' next chunks in one left-padded generate call.

    requests: sequence of (owner, feature, chunk_size). Rows for every owner are
    repeated chunk_size times and fused into a single batch; prompts of unequal
    length are left-padded (mm pad value -1 keeps ReMoH masks off the padding,
    same convention the batched evaluation path already relies on). Returned
    sequences are rebuilt as [true prompt, completion] so downstream scoring
    sees exactly what the per-group generation path would produce.
    """

    pad_token_id = processor.tokenizer.pad_token_id
    row_features = []
    row_owners = []
    for owner, feature, chunk_size in requests:
        for _ in range(int(chunk_size)):
            row_features.append(feature)
            row_owners.append(owner)
    batch = collate_qwen35_features(list(row_features), pad_token_id, padding_side="left")
    batch = move_batch_to_device(batch, context.device)
    padded_length = int(batch["input_ids"].shape[1])
    mm_types = batch["mm_token_type_ids"]
    with remoh_generation_masks(base_model, mm_types.eq(2), mm_types.eq(0)):
        generated = base_model.generate(
            **_forward_inputs(batch, include_remoh_masks=False),
            max_new_tokens=int(args.max_new_tokens),
            **_generation_sampling_kwargs(args, greedy),
            num_beams=1,
            use_cache=True,
        )

    stop_ids = _stop_token_ids(processor)
    prompts = {owner: feature["input_ids"] for owner, feature, _ in requests}
    sequences_by_owner: dict = {owner: [] for owner, _, _ in requests}
    completion_ids = []
    for row, owner in enumerate(row_owners):
        prompt_ids = prompts[owner].to(generated.device)
        prompt_length = int(prompt_ids.shape[0])
        full = torch.cat([prompt_ids, generated[row, padded_length:]])
        trimmed = _trim_sequence(full, prompt_length, stop_ids, pad_token_id)
        sequences_by_owner[owner].append(trimmed)
        completion_ids.append(trimmed[prompt_length:])
    decoded = processor.batch_decode(
        completion_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    answers_by_owner: dict = {owner: [] for owner, _, _ in requests}
    for owner, text in zip(row_owners, decoded):
        answers_by_owner[owner].append(
            strip_thinking(text.replace("<|im_end|>", "").replace("<|endoftext|>", ""))
        )
    return {owner: (sequences_by_owner[owner], answers_by_owner[owner]) for owner, _, _ in requests}


def sample_experiment_groups_batched(
    model,
    processor,
    items,
    spec: AlgorithmSpec | str,
    personalized_token: str,
    pa_state: PersonalizedAdaptiveState | None,
    args,
    context,
) -> list[SampledGroup]:
    """Sample several groups together, fusing each generation wavefront.

    Scoring, dynamic-rollout decisions and finalization reuse the exact same
    helpers as sample_experiment_group; only autoregressive generation is
    batched across groups (one wavefront per round, so dynamic expansions of
    different groups are fused too). Group updates stay per-group elsewhere —
    this changes wall-clock behaviour, not algorithm semantics.
    """

    spec = get_algorithm_spec(spec) if isinstance(spec, str) else spec
    states = [
        _new_group_state(feature, record, personalized_token, spec, pa_state)
        for feature, record in items
    ]
    use_anchor = bool(getattr(args, "greedy_anchor_candidate", False))
    with generation_runtime(model) as base_model:
        while True:
            anchor_requests = []
            sampled_requests = []
            for index, state in enumerate(states):
                chunk_size = _next_chunk_size(
                    spec, state["scored"], state["qa_type"], state["record"].is_positive, args
                )
                if chunk_size <= 0:
                    continue
                # 组的第一个chunk: 1个greedy锚 + 其余采样(见sample_experiment_group)。
                if use_anchor and not state["scored"]:
                    anchor_requests.append((index, state["feature"], 1))
                    chunk_size -= 1
                if chunk_size > 0:
                    sampled_requests.append((index, state["feature"], chunk_size))
            if not anchor_requests and not sampled_requests:
                break
            produced_anchor = (
                _generate_candidate_chunks_multi(base_model, processor, anchor_requests, args, context, greedy=True)
                if anchor_requests
                else {}
            )
            produced_sampled = (
                _generate_candidate_chunks_multi(base_model, processor, sampled_requests, args, context)
                if sampled_requests
                else {}
            )
            for index in sorted(set(produced_anchor) | set(produced_sampled)):
                chunk_sequences: list = []
                chunk_answers: list = []
                if index in produced_anchor:
                    chunk_sequences.extend(produced_anchor[index][0])
                    chunk_answers.extend(produced_anchor[index][1])
                if index in produced_sampled:
                    chunk_sequences.extend(produced_sampled[index][0])
                    chunk_answers.extend(produced_sampled[index][1])
                _absorb_candidate_chunk(states[index], chunk_sequences, chunk_answers, spec, args)
        return [_finalize_group(base_model, processor, state, spec, args, context) for state in states]


def _all_reduce_active_count(context, is_active: bool) -> int:
    value = torch.tensor([1 if is_active else 0], dtype=torch.int64, device=context.device)
    if context.distributed:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return int(value.item())


def _synchronize_spr_weight(
    remoh_loss: AdaptiveReMoHLoss,
    context,
    is_active: bool,
    active_count: int,
) -> None:
    if active_count == 0:
        return
    value = remoh_loss.spr_weight.detach().clone() if is_active else remoh_loss.spr_weight.detach().clone().zero_()
    if context.distributed:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value.div_(active_count)
    remoh_loss.spr_weight.copy_(value.to(remoh_loss.spr_weight.device))


def _synchronize_pa_observations(
    pa_state: PersonalizedAdaptiveState,
    local_observation,
    context,
) -> None:
    if context.distributed:
        observations = [None for _ in range(context.world_size)]
        dist.all_gather_object(observations, local_observation)
    else:
        observations = [local_observation]
    pa_state.observe_many(observation for observation in observations if observation is not None)


def train_distributed_step(
    model,
    sampled: SampledGroup,
    optimizer,
    remoh_loss: AdaptiveReMoHLoss,
    spec: AlgorithmSpec | str,
    args,
    context,
    is_active: bool,
    ref_model=None,
) -> dict[str, float | int]:
    spec = get_algorithm_spec(spec) if isinstance(spec, str) else spec
    active_count = _all_reduce_active_count(context, is_active)
    optimizer.zero_grad(set_to_none=True)
    model.train()

    positions, targets = select_causal_positions(sampled.score_batch["labels"])
    forward_inputs = _forward_inputs(sampled.score_batch)
    forward_inputs["logits_to_keep"] = positions
    outputs = model(**forward_inputs)
    new_logprobs, new_mask = selected_token_logprobs(outputs.logits, targets)
    candidate_logprobs = new_logprobs[: sampled.candidate_count]
    candidate_mask = new_mask[: sampled.candidate_count]
    mask = sampled.token_mask & candidate_mask

    if is_active:
        advantages = torch.tensor(
            sampled.advantages,
            dtype=torch.float32,
            device=context.device,
        )
        objective = token_policy_loss if spec.loss_level == "token" else sequence_policy_loss
        policy_total, policy_stats = objective(
            candidate_logprobs,
            sampled.old_logprobs,
            advantages,
            mask,
            clip_range=float(args.clip_range),
            drift_beta=float(args.drift_beta),
        )
        if sampled.anchor_weight > 0.0:
            anchor_logprobs = new_logprobs[sampled.candidate_count :]
            anchor_mask = new_mask[sampled.candidate_count :]
            if anchor_logprobs.shape[0] != 1:
                raise RuntimeError("an anchored CA group must contain exactly one gold row")
            anchor_loss = supervised_anchor_loss(anchor_logprobs, anchor_mask)
        else:
            anchor_loss = connected_zero_loss(new_logprobs)
        auxiliary = remoh_loss.from_model(unwrap_model(model))
        ref_kl_beta = float(getattr(args, "ref_kl_beta", 0.0))
        if ref_model is not None and ref_kl_beta > 0.0:
            # 标准GRPO的参照KL: 锚向冻结的Stage 2参照模型(绝对参照系),
            # 与drift(锚向滚动行为快照)互补, 防退化跑飞与身份慢性漂移。
            with torch.no_grad():
                ref_outputs = ref_model(**forward_inputs)
                ref_all, _ = selected_token_logprobs(ref_outputs.logits, targets)
            ref_kl = reference_kl_loss(
                candidate_logprobs, ref_all[: sampled.candidate_count].detach(), mask
            )
        else:
            ref_kl = connected_zero_loss(new_logprobs).detach()
        unscaled_total = (
            policy_total
            + float(sampled.anchor_weight) * anchor_loss
            + auxiliary.total_loss
            + ref_kl_beta * ref_kl
        )
        active_ratio = float(auxiliary.active_ratio.detach().cpu())
    else:
        unscaled_total = connected_zero_loss(new_logprobs)
        zero = unscaled_total.detach()
        policy_stats = {"policy_loss": zero, "drift_loss": zero}
        anchor_loss = zero
        active_ratio = 0.0

    scale = active_loss_scale(context.world_size, active_count, is_active)
    total = unscaled_total * scale if is_active else unscaled_total
    total.backward()
    if active_count > 0:
        torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), float(args.max_grad_norm))
        optimizer.step()
    _synchronize_spr_weight(remoh_loss, context, is_active, active_count)
    return {
        "loss": float(unscaled_total.detach().cpu()) if is_active else 0.0,
        "policy_loss": float(policy_stats["policy_loss"].cpu()) if is_active else 0.0,
        "drift_loss": float(policy_stats["drift_loss"].cpu()) if is_active else 0.0,
        "anchor_loss": float(anchor_loss.detach().cpu()) if is_active else 0.0,
        "ref_kl": float(ref_kl.detach().cpu()) if is_active else 0.0,
        "active_ratio": active_ratio,
        "active_count": active_count,
        "loss_scale": scale,
    }


def _main_call(context, function):
    if not context.distributed:
        return function()
    message = [None]
    if context.is_main:
        try:
            message[0] = {"ok": True, "result": function()}
        except Exception as error:  # noqa: BLE001 - the error must reach every rank.
            message[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(message, src=0)
    status = message[0]
    if not status["ok"]:
        raise RuntimeError(f"rank 0 {status['error_type']}: {status['error']}")
    return status.get("result")


def _load_bundle_for_checkpoint(args, context, checkpoint_path: Path):
    bundle = load_qwen35_pvchat_model(
        model_path=args.model_path,
        person_token=args.sks_name,
        device=context.device,
        checkpoint_path=str(checkpoint_path),
        num_detail_tokens=args.num_detail_tokens,
        remoh_layers=args.remoh_layers,
        routed_heads=args.routed_heads,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        attn_implementation=args.attn_implementation,
        gradient_checkpointing=not args.disable_gradient_checkpointing,
    )
    set_lora_dropout(bundle.model, probability=0.0)
    if context.is_main:
        print("[Qwen3.5 Stage3 Experiment] trainable parameters:", trainable_parameter_summary(bundle.model), flush=True)
    return bundle


def _wrap_ddp(model, context):
    if not context.distributed:
        return model
    return DistributedDataParallel(
        model,
        device_ids=[context.local_rank],
        output_device=context.local_rank,
        find_unused_parameters=False,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )


def _build_optimizer_from_args(model, args):
    return build_optimizer(
        model,
        token_lr=args.token_lr,
        remoh_lr=args.remoh_lr,
        lora_lr=args.lora_lr,
        weight_decay=args.weight_decay,
    )


def _build_remoh_loss(args, device) -> AdaptiveReMoHLoss:
    return AdaptiveReMoHLoss(
        target_active_ratio=args.target_active_ratio,
        initial_spr_weight=args.initial_spr_weight,
        hae_weight=args.hae_weight,
    ).to(device)


def _validate_parent_checkpoint(
    checkpoint: Path,
    spec: AlgorithmSpec,
    expected_epoch: int,
    stage2_checkpoint: Path,
    fingerprints: Mapping[str, Any],
) -> None:
    metadata = load_checkpoint_metadata(checkpoint)
    if metadata.get("algorithm") != spec.name:
        raise ValueError(f"parent checkpoint belongs to another algorithm: {metadata.get('algorithm')}")
    if int(metadata.get("epoch", -1)) != int(expected_epoch):
        raise ValueError(f"parent checkpoint epoch mismatch: expected {expected_epoch}")
    parent_fingerprints = dict(metadata.get("input_fingerprints") or {})
    current_fingerprints = dict(_json_value(fingerprints))
    # A parent policy checkpoint is independent of the held-out test set.
    parent_fingerprints.pop("test_json", None)
    current_fingerprints.pop("test_json", None)
    for fingerprint in (*parent_fingerprints.values(), *current_fingerprints.values()):
        if isinstance(fingerprint, dict):
            fingerprint.pop("path", None)
    if parent_fingerprints != current_fingerprints:
        raise ValueError("parent checkpoint input fingerprints mismatch")


def _rank_stats_template() -> dict[str, float | int]:
    return {
        "real_groups": 0,
        "updates": 0,
        "skipped": 0,
        "zero_dispersion": 0,
        "candidate_total": 0,
        "reward_sum": 0.0,
        "reward_count": 0,
        "loss_sum": 0.0,
        "policy_loss_sum": 0.0,
        "drift_loss_sum": 0.0,
        "anchor_loss_sum": 0.0,
        "active_ratio_sum": 0.0,
        "identity_feasible_candidates": 0,
        "identity_candidate_count": 0,
        "expanded_groups": 0,
        "ig_identity_correct_candidates": 0,
        "ig_identity_wrong_candidates": 0,
        "ig_identity_unknown_candidates": 0,
        "ig_gated_groups": 0,
        "ig_identity_expanded_groups": 0,
        "ca_fallback_groups": 0,
        "ca_anchored_groups": 0,
        "ca_degenerate_candidates": 0,
        "ca_identity_wrong_candidates": 0,
    }


def _update_rank_stats(rank_stats: dict, sampled: SampledGroup, step_stats: Mapping[str, Any]) -> None:
    rewards = [float(item.reward) for item in sampled.scored]
    rank_stats["real_groups"] += 1
    rank_stats["updates"] += int(sampled.decision.should_update)
    rank_stats["skipped"] += int(not sampled.decision.should_update)
    rank_stats["zero_dispersion"] += int(max(rewards) - min(rewards) == 0.0)
    rank_stats["candidate_total"] += len(rewards)
    rank_stats["reward_sum"] += sum(rewards)
    rank_stats["reward_count"] += len(rewards)
    if sampled.scored and all("identity" in item.components for item in sampled.scored):
        rank_stats["identity_candidate_count"] += len(sampled.scored)
        rank_stats["identity_feasible_candidates"] += sum(
            float(item.components["identity"]) >= 0.5 for item in sampled.scored
        )
        rank_stats["expanded_groups"] += int(
            sampled.algorithm == "icd_gspo_v1" and len(sampled.scored) == 8
        )
    if sampled.algorithm == "ig_dynamic_gspo":
        states = [float(item.components["identity_gate"]) for item in sampled.scored]
        rank_stats["ig_identity_correct_candidates"] += sum(
            value == IDENTITY_CORRECT for value in states
        )
        rank_stats["ig_identity_wrong_candidates"] += sum(
            value == IDENTITY_WRONG for value in states
        )
        rank_stats["ig_identity_unknown_candidates"] += sum(
            value == IDENTITY_UNKNOWN for value in states
        )
        rank_stats["ig_gated_groups"] += int(sampled.identity_gated)
        rank_stats["ig_identity_expanded_groups"] += int(sampled.identity_expanded)
    if get_algorithm_spec(sampled.algorithm).is_constraint_anchored:
        rank_stats["ca_fallback_groups"] += int(
            bool(getattr(sampled.decision, "use_sft_fallback", False))
        )
        rank_stats["ca_anchored_groups"] += int(sampled.anchor_weight > 0.0)
        rank_stats["ca_degenerate_candidates"] += sum(
            float(item.components.get("degeneration", 0.0)) > 0.0
            for item in sampled.scored
        )
        rank_stats["ca_identity_wrong_candidates"] += sum(
            float(item.components.get("identity_gate", 0.0)) == IDENTITY_WRONG
            for item in sampled.scored
        )
    if sampled.decision.should_update:
        rank_stats["loss_sum"] += float(step_stats["loss"])
        rank_stats["policy_loss_sum"] += float(step_stats["policy_loss"])
        rank_stats["drift_loss_sum"] += float(step_stats["drift_loss"])
        rank_stats["anchor_loss_sum"] += float(step_stats["anchor_loss"])
        rank_stats["active_ratio_sum"] += float(step_stats["active_ratio"])


def _rollout_row(record, sampled: SampledGroup) -> dict[str, Any]:
    row = {
        "group_id": f"{record.video_index}:{record.qa_index}",
        "flat_index": int(record.flat_index),
        "video_path": record.video_path,
        "question": record.question,
        "gold_answer": record.answer,
        "qa_type": sampled.qa_type,
        "candidate_count": len(sampled.answers),
        "should_update": sampled.decision.should_update,
        "decision_reason": sampled.decision.reason,
        "decision_diagnostics": dict(getattr(sampled.decision, "diagnostics", {})),
        "category_multiplier": sampled.category_multiplier,
        "candidates": [
            {
                "answer": answer,
                "reward": result.reward,
                "components": result.components,
                "advantage": advantage,
            }
            for answer, result, advantage in zip(sampled.answers, sampled.scored, sampled.advantages)
        ],
    }
    if sampled.algorithm == "ig_dynamic_gspo":
        row["identity_decision_trace"] = list(sampled.decision_trace)
        row["identity_gated"] = sampled.identity_gated
        row["identity_expanded"] = sampled.identity_expanded
    if get_algorithm_spec(sampled.algorithm).is_constraint_anchored:
        row["anchor_weight"] = sampled.anchor_weight
        row["sft_fallback"] = bool(
            getattr(sampled.decision, "use_sft_fallback", False)
        )
    return row


def _gather_objects(context, local_value):
    if not context.distributed:
        return [local_value]
    gathered = [None for _ in range(context.world_size)]
    dist.all_gather_object(gathered, local_value)
    return gathered


def _train_epoch(
    args,
    context,
    spec: AlgorithmSpec,
    epoch: int,
    records,
    slots: Sequence[int | None],
    expected_count: int,
    expected_indices: set[int],
    stage2_checkpoint: Path,
    parent_checkpoint: Path,
    fingerprints: Mapping[str, Any],
    input_fingerprint: str,
    video_overrides: Any,
):
    if epoch > 1:
        _validate_parent_checkpoint(
            parent_checkpoint,
            spec,
            expected_epoch=epoch - 1,
            stage2_checkpoint=stage2_checkpoint,
            fingerprints=fingerprints,
        )
    bundle = _load_bundle_for_checkpoint(args, context, parent_checkpoint)
    optimizer = _build_optimizer_from_args(bundle.model, args)
    remoh_loss = _build_remoh_loss(args, context.device)
    pa_state = PersonalizedAdaptiveState() if spec.is_pa else None
    if epoch > 1:
        load_optimizer_state(optimizer, parent_checkpoint)
        restored = load_runtime_state(
            parent_checkpoint / RUNTIME_STATE_NAME,
            remoh_loss,
            use_pa=spec.is_pa,
            restore_rng=False,
        )
        if spec.is_pa:
            pa_state = restored["pa_state"]
    ref_model = None
    if float(getattr(args, "ref_kl_beta", 0.0)) > 0.0:
        if context.is_main:
            print("[Qwen3.5 Stage3 Experiment] loading frozen Stage 2 reference for ref-KL", flush=True)
        ref_bundle = _load_bundle_for_checkpoint(args, context, stage2_checkpoint)
        ref_model = ref_bundle.model
        ref_model.eval()
        ref_model.requires_grad_(False)
    _seed_everything(_epoch_seed(args.seed, epoch))
    model = _wrap_ddp(bundle.model, context)

    output_root = Path(args.output_dir)
    shard_path = output_root / "rollouts" / f"epoch_{epoch}" / f"rank_{context.rank}.jsonl"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    rank_stats = _rank_stats_template()
    cached_sample: SampledGroup | None = None
    started = time.monotonic()
    progress = tqdm(slots, desc=f"{spec.name} epoch {epoch}", disable=not context.is_main)
    with shard_path.open("w", encoding="utf-8") as rollout_file:
        def collect_slot(index):
            nonlocal cached_sample
            record = None
            if index is not None:
                record = records[index]
                feature = encode_record(
                    bundle.processor,
                    record,
                    bundle.personalized_tokens,
                    stage=2,
                    answer=None,
                    video_overrides=video_overrides,
                )
                sampled = sample_experiment_group(
                    model,
                    bundle.processor,
                    feature,
                    record,
                    bundle.personalized_tokens[0],
                    spec,
                    pa_state,
                    args,
                    context,
                )
                cached_sample = sampled
                is_active = sampled.decision.should_update
            else:
                if cached_sample is None:
                    raise RuntimeError("dummy DDP slot has no previous real score batch")
                sampled = cached_sample
                is_active = False

            return record, sampled, is_active

        def collect_slots(window):
            nonlocal cached_sample
            real_positions = [pos for pos, index in enumerate(window) if index is not None]
            if getattr(args, "disable_batched_rollout", False) or len(real_positions) <= 1:
                return [collect_slot(index) for index in window]
            items = []
            for pos in real_positions:
                window_record = records[window[pos]]
                window_feature = encode_record(
                    bundle.processor,
                    window_record,
                    bundle.personalized_tokens,
                    stage=2,
                    answer=None,
                    video_overrides=video_overrides,
                )
                items.append((window_feature, window_record))
            sampled_groups = sample_experiment_groups_batched(
                model,
                bundle.processor,
                items,
                spec,
                bundle.personalized_tokens[0],
                pa_state,
                args,
                context,
            )
            by_position = dict(zip(real_positions, sampled_groups))
            prepared = []
            for pos, index in enumerate(window):
                if index is not None:
                    sampled = by_position[pos]
                    cached_sample = sampled
                    prepared.append((records[index], sampled, sampled.decision.should_update))
                else:
                    if cached_sample is None:
                        raise RuntimeError("dummy DDP slot has no previous real score batch")
                    prepared.append((None, cached_sample, False))
            return prepared

        def update_slot(prepared_slot):
            record, sampled, is_active = prepared_slot
            step_stats = train_distributed_step(
                model,
                sampled,
                optimizer,
                remoh_loss,
                spec,
                args,
                context,
                is_active=is_active,
                ref_model=ref_model,
            )
            if spec.is_pa:
                local_observation = None
                if record is not None:
                    mean_reward = sum(item.reward for item in sampled.scored) / len(sampled.scored)
                    local_observation = (
                        bundle.personalized_tokens[0],
                        sampled.adaptive_category,
                        mean_reward,
                    )
                _synchronize_pa_observations(pa_state, local_observation, context)

            if record is not None:
                rollout_file.write(json.dumps(_rollout_row(record, sampled), ensure_ascii=False) + "\n")
                rollout_file.flush()
                _update_rank_stats(rank_stats, sampled, step_stats)
                if context.is_main:
                    progress.set_postfix(
                        loss=f"{step_stats['loss']:.4f}",
                        candidates=len(sampled.answers),
                    )

        execute_rollout_schedule(
            spec,
            progress,
            rollout_buffer_size=int(getattr(args, "rollout_buffer_size", 4)),
            collect_slot=collect_slot,
            update_slot=update_slot,
            collect_slots=collect_slots,
        )

    elapsed = time.monotonic() - started
    rank_stats["elapsed_seconds"] = elapsed
    gathered_stats = _gather_objects(context, rank_stats)
    checkpoint_dir = output_root / "checkpoints" / f"epoch_{epoch}"
    stats_path = output_root / "training_stats" / f"epoch_{epoch}.json"

    def finish_training():
        merged_path = merge_rollout_shards(
            output_root,
            epoch,
            world_size=context.world_size,
            expected_count=expected_count,
            expected_indices=expected_indices,
        )
        metadata = build_checkpoint_metadata(
            bundle.metadata,
            spec,
            epoch,
            stage2_checkpoint,
            parent_checkpoint,
            args,
            fingerprints,
        )
        save_checkpoint(
            checkpoint_dir,
            unwrap_model(model),
            bundle.processor,
            metadata,
            optimizer=optimizer,
        )
        save_runtime_state(checkpoint_dir / RUNTIME_STATE_NAME, remoh_loss, pa_state)
        adaptive_state_path = save_adaptive_state_artifact(
            output_root / "adaptive_state" / f"epoch_{epoch}.pt",
            spec.name,
            epoch,
            pa_state,
        )
        stats = aggregate_epoch_stats(
            gathered_stats,
            spec,
            epoch,
            input_fingerprint,
            max(float(item.get("elapsed_seconds", elapsed)) for item in gathered_stats),
        )
        update_count = stats["updates"]
        if update_count:
            stats.update(
                {
                    "loss_mean": sum(float(item["loss_sum"]) for item in gathered_stats) / update_count,
                    "policy_loss_mean": sum(float(item["policy_loss_sum"]) for item in gathered_stats) / update_count,
                    "drift_loss_mean": sum(float(item["drift_loss_sum"]) for item in gathered_stats) / update_count,
                    "anchor_loss_mean": sum(float(item["anchor_loss_sum"]) for item in gathered_stats) / update_count,
                    "active_ratio_mean": sum(float(item["active_ratio_sum"]) for item in gathered_stats) / update_count,
                }
            )
        else:
            stats.update(
                {
                    "loss_mean": 0.0,
                    "policy_loss_mean": 0.0,
                    "drift_loss_mean": 0.0,
                    "anchor_loss_mean": 0.0,
                    "active_ratio_mean": 0.0,
                }
            )
        atomic_write_json(stats_path, stats)
        write_phase_marker(
            output_root,
            epoch,
            "training",
            {
                "algorithm": spec.name,
                "checkpoint": str(checkpoint_dir.resolve()),
                "runtime_state": str((checkpoint_dir / RUNTIME_STATE_NAME).resolve()),
                "adaptive_state": str(adaptive_state_path.resolve()),
                "rollouts": str(merged_path.resolve()),
                "training_stats": str(stats_path.resolve()),
                "real_groups": expected_count,
                "input_fingerprint": input_fingerprint,
            },
        )
        return str(checkpoint_dir)

    _main_call(context, finish_training)
    if ref_model is not None:
        del ref_model, ref_bundle
        torch.cuda.empty_cache()
    return bundle, model, checkpoint_dir


def _load_evaluation_model(args, context, checkpoint_dir: Path):
    bundle = _load_bundle_for_checkpoint(args, context, checkpoint_dir)
    return bundle, _wrap_ddp(bundle.model, context)


def metrics_completion_payload(summary_path: Path, spec: AlgorithmSpec | str) -> dict[str, Any]:
    spec = get_algorithm_spec(spec) if isinstance(spec, str) else spec
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    usage = summary.get("judge", {}).get("model_usage", {})
    used_models = set()
    for name, counts in usage.items():
        if isinstance(counts, Mapping):
            used_models.update(model for model, count in counts.items() if count)
        elif counts:
            used_models.add(name)
    used_models = sorted(used_models)
    return {
        "algorithm": spec.name,
        "summary": str(summary_path.resolve()),
        "judge_model_usage": usage,
        "judge_models_used": used_models,
        "mixed_judges": len(used_models) > 1,
    }


def _validate_experiment_args(args, spec: AlgorithmSpec) -> None:
    if not getattr(args, "checkpoint_path", None):
        raise ValueError("Stage 3 experiment requires the common Stage 2 --checkpoint_path")
    if int(args.num_epochs) <= 0:
        raise ValueError("num_epochs must be positive")
    expected_samples = 8 if spec.rollout_mode == "fixed8" else 4
    if int(args.num_samples) != expected_samples:
        raise ValueError(
            f"{spec.name} requires --num_samples {expected_samples}; got {args.num_samples}"
        )
    if float(args.advantage_eps) <= 0:
        raise ValueError("advantage_eps must be positive")
    if spec.is_icd:
        identity_margin = float(getattr(args, "identity_margin", 1.0))
        soft_clip = float(getattr(args, "icd_soft_clip", 0.5))
        conflict_threshold = float(getattr(args, "icd_conflict_threshold", 0.25))
        if identity_margin < soft_clip:
            raise ValueError("identity_margin must be at least icd_soft_clip")
        if conflict_threshold < 0:
            raise ValueError("icd_conflict_threshold must be non-negative")
    if spec.is_identity_gated:
        identity_margin = float(getattr(args, "ig_identity_margin", 1.0))
        soft_clip = float(getattr(args, "ig_soft_clip", 0.25))
        if not math.isfinite(identity_margin) or not math.isfinite(soft_clip):
            raise ValueError("IG identity margin and soft clip must be finite")
        if soft_clip < 0 or identity_margin <= 2.0 * soft_clip:
            raise ValueError("ig_identity_margin must be greater than 2 * ig_soft_clip")
    if spec.is_constraint_anchored:
        semantic_threshold = float(args.ca_semantic_threshold)
        informative_margin = float(args.ca_informative_margin)
        sft_weight = float(args.ca_sft_weight)
        fallback_weight = float(args.ca_fallback_sft_weight)
        values = (semantic_threshold, informative_margin, sft_weight, fallback_weight)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("CA thresholds and anchor weights must be finite")
        if not 0.0 <= semantic_threshold <= 1.0:
            raise ValueError("ca_semantic_threshold must be between zero and one")
        if informative_margin < 0.0:
            raise ValueError("ca_informative_margin must be non-negative")
        if sft_weight < 0.0:
            raise ValueError("ca_sft_weight must be non-negative")
        if fallback_weight < sft_weight:
            raise ValueError("ca_fallback_sft_weight must be at least ca_sft_weight")
    if spec.is_buffered and int(getattr(args, "rollout_buffer_size", 4)) <= 1:
        raise ValueError("CA-Dynamic-GSPO requires rollout_buffer_size greater than one")


def run_stage3_experiment(args, context, algorithm: str | None = None):
    """Run one isolated Stage 3 algorithm with per-phase epoch resume."""

    spec = get_algorithm_spec(algorithm or args.algorithm)
    _validate_experiment_args(args, spec)
    output_root = Path(args.output_dir).expanduser().resolve()
    stage2_checkpoint = Path(args.checkpoint_path).expanduser().resolve()

    def initialize_manifest():
        fingerprints = collect_input_fingerprints(stage2_checkpoint, args.train_json, args.test_json)
        manifest = build_experiment_manifest(spec.name, args, stage2_checkpoint, fingerprints)
        validate_or_write_manifest(output_root / MANIFEST_NAME, manifest)
        if args.no_resume:
            for epoch in range(1, int(args.num_epochs) + 1):
                for phase in PHASES:
                    phase_marker_path(output_root, epoch, phase).unlink(missing_ok=True)
        return {"fingerprints": fingerprints, "manifest": manifest}

    initialized = _main_call(context, initialize_manifest)
    fingerprints = initialized["fingerprints"]
    input_fingerprint = manifest_fingerprint(initialized["manifest"])
    budget_per_10s = getattr(args, "video_budget_per_10_seconds", None)
    if budget_per_10s:
        # 逐视频按实际时长缩放（新版Stage 2规则），rollout、SFT anchor和测试三条
        # 输入路径都经过encode_record，因此在这里统一替换即可。
        video_overrides = DynamicVideoBudget(int(budget_per_10s), min_tokens=int(args.video_min_tokens))
        if context.is_main:
            print(f"[stage3] 逐视频动态视觉预算 {video_overrides.describe()}（video_max_tokens 不再作为固定预算）", flush=True)
    else:
        video_overrides = video_overrides_from_token_budget(args.video_min_tokens, args.video_max_tokens)
    data_summary = validate_training_json_classes(args.train_json, f"Qwen3.5 Stage 3 {spec.name}")
    if context.is_main:
        print(describe_training_summary(data_summary), flush=True)
    records = load_qa_records(args.train_json)

    for epoch in range(1, int(args.num_epochs) + 1):
        all_slots = build_epoch_slots(
            len(records),
            context.world_size,
            _epoch_seed(args.seed, epoch),
            max_steps_per_epoch=args.max_steps_per_epoch,
        )
        local_slots = all_slots[context.rank]
        expected_count = sum(value is not None for rank_slots in all_slots for value in rank_slots)
        expected_indices = {
            int(value)
            for rank_slots in all_slots
            for value in rank_slots
            if value is not None
        }
        plan = plan_epoch_phases(
            output_root,
            epoch,
            skip_evaluation=args.skip_evaluation,
            skip_metrics=args.skip_metrics,
            resume=not args.no_resume,
        )
        checkpoint_dir = output_root / "checkpoints" / f"epoch_{epoch}"
        bundle = None
        model = None

        if plan.run_training:
            parent_checkpoint = parent_checkpoint_for_epoch(output_root, stage2_checkpoint, epoch)
            bundle, model, checkpoint_dir = _train_epoch(
                args,
                context,
                spec,
                epoch,
                records,
                local_slots,
                expected_count,
                expected_indices,
                stage2_checkpoint,
                parent_checkpoint,
                fingerprints,
                input_fingerprint,
                video_overrides,
            )

        if plan.run_evaluation:
            if bundle is None:
                _validate_parent_checkpoint(
                    checkpoint_dir,
                    spec,
                    expected_epoch=epoch,
                    stage2_checkpoint=stage2_checkpoint,
                    fingerprints=fingerprints,
                )
                bundle, model = _load_evaluation_model(args, context, checkpoint_dir)
            result_json = output_root / "evaluations" / f"epoch_{epoch}" / "test_results.json"
            run_distributed_evaluation(
                model,
                bundle.processor,
                args.test_json,
                bundle.personalized_tokens,
                result_json,
                context,
                stage=2,
                max_new_tokens=args.eval_max_new_tokens,
                batch_size=args.eval_batch_size,
                video_overrides=video_overrides,
            )
            _main_call(
                context,
                lambda: str(
                    write_phase_marker(
                        output_root,
                        epoch,
                        "evaluation",
                        {
                            "algorithm": spec.name,
                            "result_json": str(result_json.resolve()),
                            "input_fingerprint": input_fingerprint,
                        },
                    )
                ),
            )

        if plan.run_metrics:
            result_json = output_root / "evaluations" / f"epoch_{epoch}" / "test_results.json"
            metrics_dir = output_root / "metrics" / f"epoch_{epoch}"

            # The 9B policy is no longer needed in this epoch. Drop every local
            # reference before the metrics subprocess loads the 67GB 35B judge.
            model = None
            bundle = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            barrier(context)

            def run_metrics_on_main():
                summary_path = run_metrics(
                    result_json,
                    metrics_dir,
                    args.sks_name,
                    judge_backend=args.judge_backend,
                    judge_model=args.judge_model,
                    judge_fallback_models=args.judge_fallback_models,
                    api_num_workers=args.api_num_workers,
                    bertscore_device=f"cuda:{context.local_rank}" if torch.cuda.is_available() else "cpu",
                    local_judge_model_path=args.local_judge_model_path,
                    local_judge_device=args.local_judge_device,
                    local_judge_batch_size=args.local_judge_batch_size,
                    local_judge_es_items_per_prompt=args.local_judge_es_items_per_prompt,
                    local_judge_dc_items_per_prompt=args.local_judge_dc_items_per_prompt,
                    local_judge_max_new_tokens=args.local_judge_max_new_tokens,
                )
                payload = metrics_completion_payload(summary_path, spec)
                payload["input_fingerprint"] = input_fingerprint
                write_phase_marker(output_root, epoch, "metrics", payload)
                return str(summary_path)

            _main_call(context, run_metrics_on_main)

        del model
        del bundle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        barrier(context)
    return output_root / "checkpoints" / f"epoch_{int(args.num_epochs)}"


run_stage3 = run_stage3_experiment
