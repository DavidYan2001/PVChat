import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from importlib.machinery import ModuleSpec
from unittest.mock import patch


MULTI_MODALITY_DIR = Path(__file__).resolve().parents[1]
if str(MULTI_MODALITY_DIR) not in sys.path:
    sys.path.insert(0, str(MULTI_MODALITY_DIR))


def import_module_with_stubs(module_name: str):
    sys.modules.pop(module_name, None)

    decord_stub = types.ModuleType("decord")
    decord_stub.__spec__ = ModuleSpec("decord", loader=None)
    decord_stub.VideoReader = object
    decord_stub.cpu = lambda *_args, **_kwargs: None
    decord_stub.bridge = types.SimpleNamespace(set_bridge=lambda *_args, **_kwargs: None)

    openai_stub = types.ModuleType("openai")
    openai_stub.__spec__ = ModuleSpec("openai", loader=None)

    class DummyOpenAI:
        def __init__(self, *_args, **_kwargs):
            pass

    openai_stub.OpenAI = DummyOpenAI

    cv2_stub = types.ModuleType("cv2")
    cv2_stub.__spec__ = ModuleSpec("cv2", loader=None)
    cv2_stub.VideoCapture = object
    cv2_stub.VideoWriter = object
    cv2_stub.VideoWriter_fourcc = lambda *_args: 0
    cv2_stub.CAP_PROP_FPS = 5

    with patch.dict(
        sys.modules,
        {
            "decord": decord_stub,
            "openai": openai_stub,
            "cv2": cv2_stub,
        },
        clear=False,
    ):
        return importlib.import_module(module_name)


class TwoPersonPromptRoutingTest(unittest.TestCase):
    def test_person2_branch_targets_the_woman_in_source(self):
        source_path = MULTI_MODALITY_DIR / "video_qa_generation_all_video_2person.py"
        source_text = source_path.read_text(encoding="utf-8")

        self.assertIn('elif sample_type == "person2":', source_text)
        self.assertIn('question_in_intern = q.replace("<sks>", "the woman")', source_text)


class ShortVideoFilteringTest(unittest.TestCase):
    def _run_short_filter(self, module_name: str, *process_args):
        module = import_module_with_stubs(module_name)
        payload = {
            "videos": [
                {
                    "video_path": "/tmp/demo.mp4",
                    "qa_pairs": [
                        {
                            "question": "How would you describe <Aa>'s appearance and attire?",
                            "answer": "Clothing answer",
                            "is_special": False,
                        },
                        {
                            "question": "How would you describe <Aa>'s facial expression or emotional state in this footage?",
                            "answer": "Emotion answer",
                            "is_special": False,
                        },
                        {
                            "question": "Is <Aa> in this video?",
                            "answer": "Yes.",
                            "is_special": True,
                        },
                    ],
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch.object(module, "create_short_video", return_value=True), patch.object(
                module.os.path, "exists", return_value=True
            ), patch.object(module.os, "makedirs"):
                new_videos = module.process_videos(str(input_path), *process_args)

        return new_videos[0]["qa_pairs"]

    def test_single_person_short_filter_excludes_emotion_questions(self):
        qa_pairs = self._run_short_filter("short_video_generation", "<Aa>")
        questions = [qa["question"] for qa in qa_pairs]

        self.assertIn("How would you describe <Aa>'s appearance and attire?", questions)
        self.assertIn("Is <Aa> in this video?", questions)
        self.assertNotIn(
            "How would you describe <Aa>'s facial expression or emotional state in this footage?",
            questions,
        )

    def test_two_person_short_filter_excludes_emotion_questions(self):
        qa_pairs = self._run_short_filter("short_video_generation_2people", "Aa", "Bb")
        questions = [qa["question"] for qa in qa_pairs]

        self.assertIn("How would you describe <Aa>'s appearance and attire?", questions)
        self.assertNotIn(
            "How would you describe <Aa>'s facial expression or emotional state in this footage?",
            questions,
        )

    def test_three_person_short_filter_excludes_emotion_questions(self):
        qa_pairs = self._run_short_filter("short_video_generation_3people", "Aa", "Bb", "Cc")
        questions = [qa["question"] for qa in qa_pairs]

        self.assertIn("How would you describe <Aa>'s appearance and attire?", questions)
        self.assertNotIn(
            "How would you describe <Aa>'s facial expression or emotional state in this footage?",
            questions,
        )


class NegativeSampleExpansionTest(unittest.TestCase):
    def test_negative_samples_receive_absence_style_emotion_answers(self):
        module = import_module_with_stubs("video_qa_generation_all_video")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.json"
            output_path = Path(tmpdir) / "output.json"
            input_path.write_text(
                json.dumps(
                    {
                        "data": [
                            {
                                "is_positive": False,
                                "video_name": "negative_demo",
                                "video_path": "/tmp/demo.mp4",
                                "qa_pairs": [
                                    {
                                        "question": "Is <sks> in this video?",
                                        "answer": "No.",
                                        "is_special": True,
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(
                module, "ask_qwen_video_qa", side_effect=AssertionError("negative samples should not call the model")
            ), patch.object(module.random, "choice", side_effect=lambda seq: seq[0]):
                module.process_json_file(str(input_path), str(output_path))

            processed = json.loads(output_path.read_text(encoding="utf-8"))

        qa_pairs = processed["data"][0]["qa_pairs"]
        self.assertEqual(len(qa_pairs), 21)
        emotion_pair = next(
            qa for qa in qa_pairs if "visible emotion" in qa["question"]
        )
        self.assertIn("not present", emotion_pair["answer"])


if __name__ == "__main__":
    unittest.main()
