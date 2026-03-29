import os
from pathlib import Path


DEFAULT_QWEN3_VL_MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_VIDEO_MAX_PIXELS = 360 * 420
DEFAULT_VIDEO_FPS = 1.0
DEFAULT_MAX_NEW_TOKENS = 128


def resolve_qwen_model_path(model_path=None):
    return model_path or os.getenv("QWEN3_VL_MODEL_PATH", DEFAULT_QWEN3_VL_MODEL_PATH)


def to_video_uri(video_path):
    if video_path.startswith(("file://", "http://", "https://")):
        return video_path
    return Path(video_path).expanduser().resolve().as_uri()


def build_video_messages(video_path, question, fps=DEFAULT_VIDEO_FPS, max_pixels=DEFAULT_VIDEO_MAX_PIXELS):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": to_video_uri(video_path),
                    "fps": fps,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": question},
            ],
        }
    ]


def unpack_video_inputs(videos):
    if videos is None:
        return None, None
    unpacked_videos, metadata = zip(*videos)
    return list(unpacked_videos), list(metadata)


def load_qwen_model_and_processor(model_path=None, hf_token=None):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import torch

    resolved_model_path = resolve_qwen_model_path(model_path)
    model_kwargs = {"device_map": "auto", "token": hf_token}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForImageTextToText.from_pretrained(
        resolved_model_path,
        **model_kwargs,
    )
    processor = AutoProcessor.from_pretrained(resolved_model_path, token=hf_token)
    return model, processor


def prepare_qwen_inputs(processor, messages, device, process_vision_info_fn=None):
    if process_vision_info_fn is None:
        from qwen_vl_utils import process_vision_info as process_vision_info_fn

    image_patch_size = getattr(getattr(processor, "image_processor", None), "patch_size", 16)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info_fn(
        messages,
        image_patch_size=image_patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    videos, video_metadata = unpack_video_inputs(videos)

    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        video_metadata=video_metadata,
        return_tensors="pt",
        do_resize=False,
        **video_kwargs,
    )
    return inputs.to(device)


def ask_qwen3_vl(
    video_path,
    question,
    model=None,
    processor=None,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    process_vision_info_fn=None,
):
    if model is None or processor is None:
        raise ValueError("Both model and processor must be provided for Qwen3-VL inference.")

    messages = build_video_messages(video_path, question)
    inputs = prepare_qwen_inputs(
        processor,
        messages,
        model.device,
        process_vision_info_fn=process_vision_info_fn,
    )

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()
