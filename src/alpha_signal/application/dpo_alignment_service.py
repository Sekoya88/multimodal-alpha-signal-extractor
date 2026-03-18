"""dpo_alignment_service.py — DPO Alignment Service implementing DPOAlignmentPort.

Builds chosen/rejected pairs from training data using model inference vs oracle (forward_return),
runs DPOTrainer, and computes calibration metrics.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from alpha_signal.application.ports import DPOAlignmentPort

# Unsloth import is lazy in _run_dpo_trainer to allow DPO_USE_UNSLOTH=0 fallback (avoids RecursionError on Colab)

logger = logging.getLogger(__name__)


class DPOAlignmentService(DPOAlignmentPort):
    """Service to build DPO preference pairs and run alignment training."""

    def build_preference_pairs(
        self,
        jsonl_path: str | Path,
        model: Any,
        processor: Any,
        device: str = "cuda",
        max_samples: int | None = None,
    ) -> list[dict[str, Any]]:
        """Build chosen/rejected pairs: oracle (from label) vs model prediction.

        Reward oracle: correct prediction (predicted action == oracle action) = chosen,
        incorrect = rejected.
        """
        import torch
        from PIL import Image
        import base64
        import io

        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        samples: list[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        if max_samples:
            samples = samples[:max_samples]

        pairs: list[dict[str, Any]] = []
        model.eval()

        for i, sample in enumerate(samples):
            messages = sample["messages"]
            user_msg = next(m for m in messages if m["role"] == "user")
            assistant_msg = next(m for m in messages if m["role"] == "assistant")

            # Extract image
            img_block = next(
                (c for c in user_msg["content"] if c.get("type") == "image"),
                None,
            )
            if not img_block:
                continue
            img_data = img_block.get("image", "")
            if img_data.startswith("data:image"):
                img_data = img_data.split(",", 1)[1]
            img_bytes = base64.b64decode(img_data)
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # Extract prompt (user text only, no image in prompt for collator)
            user_text = next(
                (c["text"] for c in user_msg["content"] if c.get("type") == "text"),
                "",
            )
            system_msg = next(m for m in messages if m["role"] == "system")
            system_text = system_msg["content"][0]["text"] if system_msg["content"] else ""
            # content as list everywhere so PyArrow schema is consistent (avoids "cannot mix list and non-list")
            prompt = [
                {"role": "system", "content": [{"type": "text", "text": system_text}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
            ]

            # Oracle (chosen) - from assistant = ground truth from forward_return
            oracle_text = assistant_msg["content"][0]["text"]
            try:
                oracle_obj = json.loads(oracle_text)
                oracle_action = oracle_obj.get("action", "HOLD")
            except json.JSONDecodeError:
                oracle_action = "HOLD"

            # Model prediction (rejected when wrong)
            # Format messages with image for processor
            infer_messages = [
                {"role": "system", "content": [{"type": "text", "text": system_text}]},
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_text}]},
            ]
            with torch.no_grad():
                full_input = processor.apply_chat_template(
                    infer_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                full_input = {k: v.to(device) if hasattr(v, "to") else v for k, v in full_input.items()}
                out = model.generate(
                    **full_input,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id
                    or processor.tokenizer.eos_token_id,
                )
                prompt_len = full_input["input_ids"].shape[1]
                gen_text = processor.decode(
                    out[0][prompt_len:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                pred_obj = _parse_signal_json(gen_text)
                pred_action = pred_obj.get("action", "HOLD")

            # Build pair: chosen = oracle, rejected = model if wrong else synthetic wrong
            chosen = oracle_text
            if pred_action != oracle_action:
                rejected = gen_text
            else:
                rejected = _synthetic_rejected(oracle_obj, oracle_action)

            pairs.append({
                "images": [image],
                "prompt": prompt,
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
            })
            if (i + 1) % 10 == 0:
                logger.info(f"  Built {i + 1}/{len(samples)} pairs")

        return pairs

    def train(
        self,
        pairs: list[dict[str, Any]],
        output_dir: str | Path,
    ) -> dict[str, float]:
        """Run DPOTrainer and return metrics."""
        from datasets import Dataset

        ds = Dataset.from_list(pairs)
        return _run_dpo_trainer(ds, Path(output_dir))


def _parse_signal_json(raw: str) -> dict[str, Any]:
    """Extract JSON signal from raw model output."""
    raw = raw.strip()
    for start in ["{", "```json"]:
        if start in raw:
            idx = raw.find(start)
            if start == "```json":
                idx += 7
            rest = raw[idx:]
            end = rest.find("}") + 1
            if end > 0:
                try:
                    return json.loads(rest[:end])
                except json.JSONDecodeError:
                    pass
    return {"action": "HOLD", "confidence": 0.5}


def _synthetic_rejected(oracle_obj: dict[str, Any], oracle_action: str) -> str:
    """Create a synthetic rejected response with wrong action."""
    flip = {"BUY": "SELL", "SELL": "BUY", "HOLD": "BUY"}
    wrong_action = flip.get(oracle_action, "HOLD")
    rejected_obj = oracle_obj.copy()
    rejected_obj["action"] = wrong_action
    rejected_obj["reasoning"] = f"Signal incorrect: {wrong_action} au lieu de {oracle_action}."
    return json.dumps(rejected_obj, ensure_ascii=False)


from dataclasses import dataclass

# Shim for transformers 5: TRL 0.24 expects MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
def _ensure_trl_vision_shim() -> None:
    import transformers.models.auto.modeling_auto as auto
    if not hasattr(auto, "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES"):
        from collections import OrderedDict
        # Empty mapping so TRL import succeeds; DPO uses explicit Qwen2_5_VLForConditionalGeneration
        auto.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = OrderedDict()


def _import_trl_collator():
    _ensure_trl_vision_shim()
    from trl.trainer.dpo_trainer import DataCollatorForPreference
    return DataCollatorForPreference


DataCollatorForPreference = _import_trl_collator()

@dataclass
class VisionDPODataCollator(DataCollatorForPreference):
    """Custom collator to handle Qwen2.5-VL vision tensors correctly for DPO.
    TRL's default pads pixel_values (making them 3D), but Qwen expects 2D concats.
    """
    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        pixel_values = []
        image_grid_thw = []
        for example in examples:
            if "pixel_values" in example:
                pixel_values.append(torch.tensor(example.pop("pixel_values")))
            if "image_grid_thw" in example:
                image_grid_thw.append(torch.tensor(example.pop("image_grid_thw")))
                
        # Let TRL pad the text inputs
        batch = super().torch_call(examples)
        
        if pixel_values:
            batch["pixel_values"] = torch.cat(pixel_values, dim=0)
        if image_grid_thw:
            # We rename to image_sizes so TRL's DPOTrainer automatically duplicates 
            # it for chosen/rejected along with pixel_values.
            batch["image_sizes"] = torch.cat(image_grid_thw, dim=0)
            
        return batch


def _patched_process_row(
    features,
    processing_class,
    max_prompt_length=None,
    max_completion_length=None,
    add_special_tokens=True,
):
    """Standalone patched process_row to avoid multiprocessing pickling timeouts.
    Ensures <|image_pad|> in prompt via processor.apply_chat_template before processor().
    Extracts chosen/rejected text from list format for tokenizer.
    """
    import torch
    processor, tokenizer = processing_class, processing_class.tokenizer

    # Build prompt messages with actual images so processor produces correct <|image_pad|>
    prompt_msgs = []
    for msg in features["prompt"]:
        if msg.get("role") == "user":
            content = []
            for c in msg.get("content", []):
                if isinstance(c, dict) and c.get("type") == "image":
                    content.append({"type": "image", "image": features["images"][0]})
                else:
                    content.append(c)
            prompt_msgs.append({"role": "user", "content": content})
        else:
            prompt_msgs.append(msg)

    # Apply template to get string with <|image_pad|> (fixes tokens:0 vs features:1820)
    prompt_str = processor.apply_chat_template(
        prompt_msgs,
        tokenize=False,
        add_generation_prompt=True,
    )
    processed_features = processor(images=features["images"], text=prompt_str, add_special_tokens=False)

    prompt_input_ids = processed_features["input_ids"][0]
    if isinstance(prompt_input_ids, torch.Tensor):
        prompt_input_ids = prompt_input_ids.tolist()

    # Qwen2.5-VL pixel_values is [num_patches, channels], no batch dim.
    pixel_values = processed_features["pixel_values"]
    if isinstance(pixel_values, torch.Tensor):
        if pixel_values.dim() == 3 and pixel_values.shape[0] == 1:
            pixel_values = pixel_values[0]
        pixel_values = pixel_values.tolist()

    # chosen/rejected can be [{"role":"assistant","content":"..."}] or plain string
    chosen = features["chosen"]
    if isinstance(chosen, list) and chosen and isinstance(chosen[0], dict):
        chosen = chosen[0].get("content", chosen[0].get("text", ""))
    rejected = features["rejected"]
    if isinstance(rejected, list) and rejected and isinstance(rejected[0], dict):
        rejected = rejected[0].get("content", rejected[0].get("text", ""))

    chosen_input_ids = tokenizer(str(chosen), add_special_tokens=False)["input_ids"]
    rejected_input_ids = tokenizer(str(rejected), add_special_tokens=False)["input_ids"]

    if add_special_tokens:
        if tokenizer.bos_token_id is not None:
            prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
        if tokenizer.eos_token_id is not None:
            prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
    chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
    rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

    if max_prompt_length is not None:
        prompt_input_ids = prompt_input_ids[-max_prompt_length:]
    if max_completion_length is not None:
        chosen_input_ids = chosen_input_ids[:max_completion_length]
        rejected_input_ids = rejected_input_ids[:max_completion_length]

    output = {
        "prompt_input_ids": prompt_input_ids,
        "pixel_values": pixel_values,
        "chosen_input_ids": chosen_input_ids,
        "rejected_input_ids": rejected_input_ids,
    }

    if "pixel_attention_mask" in processed_features:
        mask = processed_features["pixel_attention_mask"]
        if isinstance(mask, torch.Tensor):
            mask = mask[0].tolist() if mask.dim() == 2 else mask.tolist()
        output["pixel_attention_mask"] = mask
        
    if "image_grid_thw" in processed_features:
        grid = processed_features["image_grid_thw"]
        if isinstance(grid, torch.Tensor):
            if grid.dim() == 3 and grid.shape[0] == 1:
                grid = grid[0]
            grid = grid.tolist()
        output["image_grid_thw"] = grid

    return output


def _run_dpo_trainer_standard(dataset: Any, output_dir: Path) -> dict[str, float]:
    """DPO training with standard Transformers+PEFT+TRL (no Unsloth).
    Use when DPO_USE_UNSLOTH=0 to avoid RecursionError with bitsandbytes on Colab T4.
    Slower but stable. See: https://github.com/unslothai/unsloth/issues/1921
    """
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import DPOTrainer, DPOConfig

    from config import dpo_cfg

    if not torch.cuda.is_available():
        raise RuntimeError("DPO training requires CUDA")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=dpo_cfg.load_in_4bit,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=dpo_cfg.lora_r,
        lora_alpha=dpo_cfg.lora_alpha,
        lora_dropout=dpo_cfg.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if not hasattr(processor, "pad"):
        processor.pad = processor.tokenizer.pad

    from trl import DPOTrainer as TRL_DPOTrainer
    TRL_DPOTrainer.process_row = staticmethod(_patched_process_row)
    original_forward = model.forward

    def dpo_vision_forward(*f_args, **f_kwargs):
        if "image_sizes" in f_kwargs and "image_grid_thw" not in f_kwargs:
            f_kwargs["image_grid_thw"] = f_kwargs.pop("image_sizes")
        return original_forward(*f_args, **f_kwargs)

    model.forward = dpo_vision_forward

    from datasets import Dataset
    _orig_map = Dataset.map

    def _no_mp_map(self, *a, **kw):
        kw.pop("num_proc", None)
        return _orig_map(self, *a, **kw)

    Dataset.map = _no_mp_map

    args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=dpo_cfg.num_train_epochs,
        per_device_train_batch_size=dpo_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=dpo_cfg.gradient_accumulation_steps,
        learning_rate=dpo_cfg.learning_rate,
        beta=dpo_cfg.beta,
        max_prompt_length=dpo_cfg.max_prompt_length,
        max_length=dpo_cfg.max_length,
        seed=dpo_cfg.seed,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        dataset_num_proc=None,
        torch_compile=False,
    )

    trainer = TRL_DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=processor,
        data_collator=VisionDPODataCollator(
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        ),
    )
    result = trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    m = result.metrics
    state = getattr(trainer, "state", None)
    log_history = getattr(state, "log_history", []) if state else []
    last_log = log_history[-1] if log_history else {}
    rc = last_log.get("rewards/chosen", 0.0)
    rr = last_log.get("rewards/rejected", 0.0)
    return {
        "train_loss": m.get("train_loss", 0.0),
        "calibration_improvement": float(rc - rr),
        "rewards_chosen": rc,
        "rewards_rejected": rr,
    }


def _run_dpo_trainer(dataset: Any, output_dir: Path) -> dict[str, float]:
    """Run TRL DPOTrainer and compute calibration metric."""
    use_unsloth = os.environ.get("DPO_USE_UNSLOTH", "1").lower() not in ("0", "false")
    if not use_unsloth:
        logger.info("DPO_USE_UNSLOTH=0: using standard Transformers path (avoids RecursionError on Colab)")
        return _run_dpo_trainer_standard(dataset, output_dir)

    import torch
    import unsloth  # noqa: F401 - must be before FastVisionModel
    from unsloth import FastVisionModel
    from trl import DPOTrainer, DPOConfig

    from config import dpo_cfg

    if not torch.cuda.is_available():
        raise RuntimeError("DPO training requires CUDA")

    model, processor = FastVisionModel.from_pretrained(
        model_name=dpo_cfg.base_model,
        max_seq_length=dpo_cfg.max_seq_length,
        load_in_4bit=dpo_cfg.load_in_4bit,
        dtype=None,
    )
    
    # Bypass UnslothZoo's overzealous data collator replacement for Vision DPO
    if not hasattr(processor, "pad"):
        processor.pad = processor.tokenizer.pad

    model = FastVisionModel.get_peft_model(
        model,
        r=dpo_cfg.lora_r,
        lora_alpha=dpo_cfg.lora_alpha,
        lora_dropout=dpo_cfg.lora_dropout,
        target_modules="all-linear",
        use_gradient_checkpointing="unsloth",
        random_state=dpo_cfg.seed,
    )
    FastVisionModel.for_training(model)

    # Monkey-patch DPOTrainer.process_row so it preserves image_grid_thw and pixel_values
    # TRL 0.24 incorrectly assumes pixel_values has a batch dimension (like Llava [1, C, H, W])
    # and does [0], which destroys Qwen2.5-VL's [num_patches, channels] tensor.
    DPOTrainer.process_row = staticmethod(_patched_process_row)

    # Wrap model.forward to accept 'image_sizes' from TRL and pass it as 'image_grid_thw' to Qwen
    original_forward = model.forward
    def dpo_vision_forward(*f_args, **f_kwargs):
        if "image_sizes" in f_kwargs and "image_grid_thw" not in f_kwargs:
            f_kwargs["image_grid_thw"] = f_kwargs.pop("image_sizes")
        return original_forward(*f_args, **f_kwargs)
    model.forward = dpo_vision_forward

    # Force disable multiprocessing in datasets.map to avoid Colab OOM and dying subprocesses
    from datasets import Dataset
    original_map = Dataset.map
    def no_mp_map(self, *m_args, **m_kwargs):
        m_kwargs.pop("num_proc", None)
        return original_map(self, *m_args, **m_kwargs)
    Dataset.map = no_mp_map

    args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=dpo_cfg.num_train_epochs,
        per_device_train_batch_size=dpo_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=dpo_cfg.gradient_accumulation_steps,
        learning_rate=dpo_cfg.learning_rate,
        beta=dpo_cfg.beta,
        max_prompt_length=dpo_cfg.max_prompt_length,
        max_length=dpo_cfg.max_length,
        seed=dpo_cfg.seed,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        # Required to run sequentially. If set to 1, TRL uses multiprocessing which OOMs on T4 Colab GPUs.
        dataset_num_proc=None,
        # CRITICAL: Disable torch.compile to avoid RecursionError with bitsandbytes + Unsloth AOT on Colab T4.
        # See: https://github.com/unslothai/unsloth/issues/1921, #1925
        torch_compile=False,
    )

    # For Unsloth/TRL compatibility with Vision DPO
    # Pass our custom collator to prevent UnslothZoo from replacing it with DataCollatorForLanguageModeling
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=processor,
        data_collator=VisionDPODataCollator(pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id),
    )

    result = trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    m = result.metrics
    train_loss = m.get("train_loss", 0.0)
    # Calibration proxy: DPO reward margin (chosen - rejected) indicates
    # how much the model prefers correct over incorrect responses
    state = getattr(trainer, "state", None)
    log_history = getattr(state, "log_history", []) if state else []
    last_log = log_history[-1] if log_history else {}
    rewards_chosen = last_log.get("rewards/chosen", 0.0)
    rewards_rejected = last_log.get("rewards/rejected", 0.0)
    calibration_improvement = float(rewards_chosen - rewards_rejected)

    return {
        "train_loss": train_loss,
        "calibration_improvement": calibration_improvement,
        "rewards_chosen": rewards_chosen,
        "rewards_rejected": rewards_rejected,
    }
