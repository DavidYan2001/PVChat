import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from qwen35_pvchat.data import (
    DEFAULT_VIDEO_CACHE_ENTRIES,
    VideoDecodeCache,
    install_video_decode_cache,
)


class _FakeVideoProcessor:
    """模拟transformers BaseVideoProcessor.fetch_videos的行为。"""

    def __init__(self):
        self.decode_calls = []

    def fetch_videos(self, video_url_or_urls, sample_indices_fn=None):
        if isinstance(video_url_or_urls, list):
            # 与真实实现一致：列表递归走self.fetch_videos，
            # 因此打补丁后每个元素都会经过缓存包装。
            return list(
                zip(*[self.fetch_videos(x, sample_indices_fn=sample_indices_fn) for x in video_url_or_urls])
            )
        self.decode_calls.append(video_url_or_urls)
        video = torch.full((2, 4, 4, 3), fill_value=len(self.decode_calls), dtype=torch.uint8)
        metadata = SimpleNamespace(total_num_frames=2, fps=2.0, frames_indices=[0, 1])
        return video, metadata


def _make_processor():
    return SimpleNamespace(video_processor=_FakeVideoProcessor())


class VideoDecodeCacheTest(unittest.TestCase):
    def test_repeat_path_decodes_once(self):
        processor = _make_processor()
        cache = install_video_decode_cache(processor)
        first, _ = processor.video_processor.fetch_videos("/videos/a.mp4")
        second, _ = processor.video_processor.fetch_videos("/videos/a.mp4")
        self.assertEqual(processor.video_processor.decode_calls, ["/videos/a.mp4"])
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cache.misses, 1)

    def test_distinct_paths_decode_separately(self):
        processor = _make_processor()
        install_video_decode_cache(processor)
        processor.video_processor.fetch_videos("/videos/a.mp4")
        processor.video_processor.fetch_videos("/videos/b.mp4")
        self.assertEqual(
            processor.video_processor.decode_calls, ["/videos/a.mp4", "/videos/b.mp4"]
        )

    def test_hit_returns_defensive_copy(self):
        processor = _make_processor()
        install_video_decode_cache(processor)
        first, first_meta = processor.video_processor.fetch_videos("/videos/a.mp4")
        first.zero_()
        first_meta.frames_indices = None
        second, second_meta = processor.video_processor.fetch_videos("/videos/a.mp4")
        self.assertEqual(int(second.max()), 1)
        self.assertEqual(second_meta.frames_indices, [0, 1])

    def test_list_input_reuses_cache_per_element(self):
        processor = _make_processor()
        install_video_decode_cache(processor)
        processor.video_processor.fetch_videos(["/videos/a.mp4", "/videos/a.mp4"])
        self.assertEqual(processor.video_processor.decode_calls, ["/videos/a.mp4"])

    def test_lru_eviction_respects_max_entries(self):
        processor = _make_processor()
        install_video_decode_cache(processor, max_entries=1)
        processor.video_processor.fetch_videos("/videos/a.mp4")
        processor.video_processor.fetch_videos("/videos/b.mp4")
        processor.video_processor.fetch_videos("/videos/a.mp4")
        self.assertEqual(
            processor.video_processor.decode_calls,
            ["/videos/a.mp4", "/videos/b.mp4", "/videos/a.mp4"],
        )

    def test_install_is_idempotent(self):
        processor = _make_processor()
        first_cache = install_video_decode_cache(processor)
        second_cache = install_video_decode_cache(processor)
        self.assertIs(first_cache, second_cache)
        processor.video_processor.fetch_videos("/videos/a.mp4")
        processor.video_processor.fetch_videos("/videos/a.mp4")
        self.assertEqual(processor.video_processor.decode_calls, ["/videos/a.mp4"])

    def test_env_kill_switch_disables_cache(self):
        processor = _make_processor()
        with patch.dict(os.environ, {"PVCHAT_VIDEO_CACHE": "0"}):
            cache = install_video_decode_cache(processor)
        self.assertIsNone(cache)
        processor.video_processor.fetch_videos("/videos/a.mp4")
        processor.video_processor.fetch_videos("/videos/a.mp4")
        self.assertEqual(len(processor.video_processor.decode_calls), 2)

    def test_clear_forces_redecode(self):
        processor = _make_processor()
        cache = install_video_decode_cache(processor)
        self.assertIsInstance(cache, VideoDecodeCache)
        processor.video_processor.fetch_videos("/videos/a.mp4")
        cache.clear()
        self.assertEqual(len(cache), 0)
        processor.video_processor.fetch_videos("/videos/a.mp4")
        self.assertEqual(len(processor.video_processor.decode_calls), 2)

    def test_install_keeps_processor_instance_json_serializable(self):
        # 回归：缓存曾以实例属性形式挂在video_processor上，导致
        # processor.save_pretrained在训练收尾序列化配置时崩溃。
        # 缓存安装与使用都不得改变实例__dict__。
        processor = _make_processor()
        keys_before = set(vars(processor.video_processor))
        install_video_decode_cache(processor)
        processor.video_processor.fetch_videos("/videos/a.mp4")
        processor.video_processor.fetch_videos("/videos/a.mp4")
        instance_dict = vars(processor.video_processor)
        self.assertEqual(set(instance_dict), keys_before)
        json.dumps(instance_dict)

    def test_missing_video_processor_is_noop(self):
        processor = SimpleNamespace()
        self.assertIsNone(install_video_decode_cache(processor))

    def test_default_capacity_is_positive(self):
        self.assertGreater(DEFAULT_VIDEO_CACHE_ENTRIES, 0)


if __name__ == "__main__":
    unittest.main()
