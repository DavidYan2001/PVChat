import unittest

from qwen_video_qa import (
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIDEO_MAX_PIXELS,
    build_video_messages,
    unpack_video_inputs,
)


class BuildVideoMessagesTest(unittest.TestCase):
    def test_build_video_messages_uses_local_file_uri_and_defaults(self):
        messages = build_video_messages("/tmp/demo.mp4", "Describe this video.")

        self.assertEqual(len(messages), 1)
        content = messages[0]["content"]
        self.assertEqual(content[0]["type"], "video")
        self.assertEqual(content[0]["video"], "file:///tmp/demo.mp4")
        self.assertEqual(content[0]["fps"], DEFAULT_VIDEO_FPS)
        self.assertEqual(content[0]["max_pixels"], DEFAULT_VIDEO_MAX_PIXELS)
        self.assertEqual(content[1], {"type": "text", "text": "Describe this video."})


class UnpackVideoInputsTest(unittest.TestCase):
    def test_unpack_video_inputs_splits_metadata(self):
        videos, metadata = unpack_video_inputs([
            ("video_tensor_1", {"fps": 1.0}),
            ("video_tensor_2", {"fps": 2.0}),
        ])

        self.assertEqual(videos, ["video_tensor_1", "video_tensor_2"])
        self.assertEqual(metadata, [{"fps": 1.0}, {"fps": 2.0}])

    def test_unpack_video_inputs_handles_none(self):
        videos, metadata = unpack_video_inputs(None)

        self.assertIsNone(videos)
        self.assertIsNone(metadata)


if __name__ == "__main__":
    unittest.main()
