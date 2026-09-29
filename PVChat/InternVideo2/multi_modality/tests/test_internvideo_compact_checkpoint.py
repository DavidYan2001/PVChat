import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from internvideo_compact_checkpoint import (
    DELTA_FILENAME,
    DELTA_FORMAT,
    apply_compact_delta,
    load_checkpoint_state,
    save_compact_checkpoint,
)


class _TinyModel(nn.Module):
    """键名风格与真实delta一致的迷你模型（spec驱动，命名本身不敏感）。"""

    def __init__(self, vocab_rows=10, hidden=4):
        super().__init__()
        self.personal_query_tokens = nn.Parameter(torch.zeros(2, hidden))
        self.qformer = nn.Module()
        self.qformer.proj = nn.Linear(hidden, hidden, bias=False)
        self.lm = nn.Module()
        self.lm.embed = nn.Embedding(vocab_rows, hidden)
        self.lm.lm_head = nn.Linear(hidden, vocab_rows, bias=False)


def _payload(vocab_rows=10, hidden=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "format": DELTA_FORMAT,
        "vocab_size": vocab_rows,
        "dense_state": {
            "personal_query_tokens": torch.randn(2, hidden, generator=g),
            "qformer.proj.weight": torch.randn(hidden, hidden, generator=g),
        },
        "row_state": {
            "lm.embed.weight": {
                "indices": torch.tensor([vocab_rows - 2, vocab_rows - 1]),
                "values": torch.randn(2, hidden, generator=g),
            },
            "lm.lm_head.weight": {
                "indices": torch.tensor([vocab_rows - 1]),
                "values": torch.randn(1, hidden, generator=g),
            },
        },
    }


class CompactCheckpointTest(unittest.TestCase):
    def test_apply_writes_dense_and_rows_in_place(self):
        model = _TinyModel()
        payload = _payload()
        spec = apply_compact_delta(model, payload)
        self.assertTrue(torch.equal(model.personal_query_tokens.data, payload["dense_state"]["personal_query_tokens"]))
        self.assertTrue(torch.equal(model.lm.embed.weight.data[8:10], payload["row_state"]["lm.embed.weight"]["values"]))
        # 未覆盖的行保持原状
        self.assertTrue(torch.equal(model.lm.embed.weight.data[0], torch.zeros(4) + model.lm.embed.weight.data[0]))
        self.assertEqual(spec.vocab_size, 10)
        self.assertEqual(sorted(spec.row_indices), ["lm.embed.weight", "lm.lm_head.weight"])

    def test_save_then_reload_round_trips_trained_values(self):
        model = _TinyModel()
        spec = apply_compact_delta(model, _payload())
        # 模拟Stage 3训练：改动delta覆盖的参数
        with torch.no_grad():
            model.personal_query_tokens.add_(1.0)
            model.qformer.proj.weight.add_(2.0)
            model.lm.embed.weight[9].fill_(7.0)
        with tempfile.TemporaryDirectory() as tmp:
            save_compact_checkpoint(model, spec, tmp, training_info={"sks_tokens": ["<X>"]})
            saved = torch.load(Path(tmp) / DELTA_FILENAME, weights_only=False)
            self.assertEqual(saved["format"], DELTA_FORMAT)
            fresh = _TinyModel()
            load_checkpoint_state(fresh, tmp)
            self.assertTrue(torch.equal(fresh.personal_query_tokens.data, model.personal_query_tokens.data))
            self.assertTrue(torch.equal(fresh.qformer.proj.weight.data, model.qformer.proj.weight.data))
            self.assertTrue(torch.equal(fresh.lm.embed.weight.data[9], model.lm.embed.weight.data[9]))
            info = torch.load(Path(tmp) / "training_info.bin", weights_only=False)
            self.assertEqual(info["sks_tokens"], ["<X>"])
            self.assertEqual(info["checkpoint_format"], DELTA_FORMAT)

    def test_saved_file_contains_only_delta_keys(self):
        model = _TinyModel()
        spec = apply_compact_delta(model, _payload())
        with tempfile.TemporaryDirectory() as tmp:
            save_compact_checkpoint(model, spec, tmp)
            saved = torch.load(Path(tmp) / DELTA_FILENAME, weights_only=False)
            self.assertEqual(sorted(saved["dense_state"]), ["personal_query_tokens", "qformer.proj.weight"])
            self.assertEqual(saved["row_state"]["lm.embed.weight"]["values"].shape, (2, 4))
            self.assertFalse((Path(tmp) / "pytorch_model.bin").exists())
            self.assertFalse((Path(tmp) / "optimizer.pt").exists())

    def test_vocab_not_resized_raises(self):
        model = _TinyModel(vocab_rows=8)  # 小于delta要求的10行
        with self.assertRaises(ValueError):
            apply_compact_delta(model, _payload(vocab_rows=10))

    def test_unknown_key_raises(self):
        model = _TinyModel()
        payload = _payload()
        payload["dense_state"]["not.a.real.key"] = torch.zeros(1)
        with self.assertRaises(KeyError):
            apply_compact_delta(model, payload)

    def test_wrong_format_raises(self):
        with self.assertRaises(ValueError):
            apply_compact_delta(_TinyModel(), {"format": "something_else"})

    def test_redundant_lm_head_alias_is_skipped(self):
        """旧机器delta带 lm.lm_head.weight 别名(与peft包装路径逐位相同)时跳过而非报错。"""

        class _PeftLikeModel(nn.Module):
            def __init__(self, vocab_rows=10, hidden=4):
                super().__init__()
                self.personal_query_tokens = nn.Parameter(torch.zeros(2, hidden))
                self.qformer = nn.Module()
                self.qformer.proj = nn.Linear(hidden, hidden, bias=False)
                # 只有包装路径 lm.base_model.model.lm_head, 没有 lm.lm_head
                self.lm = nn.Module()
                self.lm.base_model = nn.Module()
                self.lm.base_model.model = nn.Module()
                self.lm.base_model.model.lm_head = nn.Linear(hidden, vocab_rows, bias=False)

        model = _PeftLikeModel()
        g = torch.Generator().manual_seed(1)
        shared_rows = {
            "indices": torch.tensor([9]),
            "values": torch.randn(1, 4, generator=g),
        }
        payload = {
            "format": DELTA_FORMAT,
            "vocab_size": 10,
            "dense_state": {
                "personal_query_tokens": torch.randn(2, 4, generator=g),
                "qformer.proj.weight": torch.randn(4, 4, generator=g),
            },
            "row_state": {
                "lm.base_model.model.lm_head.weight": shared_rows,
                "lm.lm_head.weight": {
                    "indices": shared_rows["indices"].clone(),
                    "values": shared_rows["values"].clone(),
                },
            },
        }
        spec = apply_compact_delta(model, payload)
        self.assertEqual(sorted(spec.row_indices), ["lm.base_model.model.lm_head.weight"])
        self.assertTrue(
            torch.equal(
                model.lm.base_model.model.lm_head.weight.data[9], shared_rows["values"][0]
            )
        )
        # 别名与包装路径数值不同时仍必须报错
        payload["row_state"]["lm.lm_head.weight"]["values"] = shared_rows["values"] + 1.0
        with self.assertRaises(KeyError):
            apply_compact_delta(_PeftLikeModel(), payload)

    def test_full_state_fallback_and_missing_dir(self):
        model = _TinyModel()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_checkpoint_state(model, tmp)
            reference = _TinyModel()
            with torch.no_grad():
                reference.qformer.proj.weight.fill_(3.0)
            torch.save(reference.state_dict(), Path(tmp) / "pytorch_model.bin")
            spec = load_checkpoint_state(model, tmp)
            self.assertIsNone(spec)  # 旧式完整格式
            self.assertTrue(torch.equal(model.qformer.proj.weight.data, reference.qformer.proj.weight.data))


if __name__ == "__main__":
    unittest.main()
