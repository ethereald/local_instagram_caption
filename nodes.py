import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import comfy.model_management as mm
import folder_paths


DEFAULT_INSTRUCTIONS = """Write one Instagram caption from the supplied image description.

Rules:
- Sound like a real person sharing a genuine moment, not an advertisement or an AI.
- Use a warm, conversational tone with natural sentence rhythm.
- Mention only facts present in the image description or account context.
- Do not invent a place, event, identity, relationship, product, or backstory.
- Write 1 to 3 short sentences.
- Use the requested number of natural emojis.
- Add a blank line, then the requested number of relevant hashtags.
- Prefer specific hashtags; avoid #instagood, #viral, #fyp, and #photo.
- Return only the finished caption. Do not add a heading, notes, or quotation marks."""


def _local_text_models():
    llm_root = os.path.realpath(os.path.join(folder_paths.models_dir, "LLM"))
    models = {}
    if not os.path.isdir(llm_root):
        return models

    for root, dirs, files in os.walk(llm_root):
        dirs[:] = [name for name in dirs if name not in {"blobs", ".locks", ".no_exist"}]
        if "config.json" not in files:
            continue

        config_path = os.path.join(root, "config.json")
        try:
            with open(config_path, "r", encoding="utf-8") as config_file:
                architectures = json.load(config_file).get("architectures", [])
        except (OSError, json.JSONDecodeError):
            continue

        if not any(name.endswith("ForCausalLM") for name in architectures):
            continue

        model_path = os.path.realpath(root)
        if os.path.commonpath([llm_root, model_path]) != llm_root:
            continue
        models[os.path.relpath(model_path, llm_root)] = model_path

    return dict(sorted(models.items()))


class LocalInstagramCaption:
    @classmethod
    def INPUT_TYPES(cls):
        cls.model_paths = _local_text_models()
        model_names = list(cls.model_paths)
        if not model_names:
            model_names = ["No local causal language model found in models/LLM"]

        return {
            "required": {
                "image_description": ("STRING", {"multiline": True, "forceInput": True}),
                "model": (model_names,),
                "tone": (["warm and natural", "playful", "thoughtful", "minimal", "enthusiastic"],),
                "language": (["English", "Traditional Chinese (繁體中文)", "Japanese"],),
                "emoji_count": ("INT", {"default": 2, "min": 0, "max": 5}),
                "hashtag_count": ("INT", {"default": 8, "min": 0, "max": 20}),
                "seed": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "control_after_generate": True,
                }),
            },
            "optional": {
                "account_context": ("STRING", {"default": "", "multiline": True}),
                "instructions": ("STRING", {"default": DEFAULT_INSTRUCTIONS, "multiline": True}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_new_tokens": ("INT", {"default": 256, "min": 32, "max": 1024}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("instagram_caption",)
    OUTPUT_NODE = True
    FUNCTION = "generate"
    CATEGORY = "Local AI/Text"

    def generate(
        self,
        image_description,
        model,
        tone,
        language,
        emoji_count,
        hashtag_count,
        seed,
        account_context="",
        instructions=DEFAULT_INSTRUCTIONS,
        temperature=0.7,
        max_new_tokens=256,
    ):
        model_path = self.model_paths.get(model)
        if model_path is None:
            raise ValueError("No usable local text model was found in ComfyUI/models/LLM.")

        description = image_description.strip()
        if not description:
            raise ValueError("image_description is empty. Connect Florence2Run's caption output.")

        system_prompt = instructions.strip() or DEFAULT_INSTRUCTIONS
        user_prompt = (
            f"Image description:\n{description}\n\n"
            f"Account context:\n{account_context.strip() or 'None provided.'}\n\n"
            f"Output language: {language.strip() or 'English'}\n"
            f"Tone: {tone}\n"
            f"Emoji count: {emoji_count}\n"
            f"Hashtag count: {hashtag_count}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")

        device = mm.get_torch_device()
        dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
            torch.float16 if device.type == "cuda" else torch.float32
        )
        language_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).eval().to(device)
        inputs = {name: value.to(device) for name, value in inputs.items()}

        torch.manual_seed(seed)
        do_sample = temperature > 0
        generate_args = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": 1.08,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generate_args["temperature"] = temperature
            generate_args["top_p"] = 0.9

        output_ids = language_model.generate(**generate_args)
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        caption = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        del output_ids, inputs, language_model, tokenizer
        mm.soft_empty_cache()

        return {"ui": {"text": [caption]}, "result": (caption,)}


NODE_CLASS_MAPPINGS = {
    "LocalInstagramCaption": LocalInstagramCaption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LocalInstagramCaption": "Local Instagram Caption",
}
