"""dpo_alignment_service.py — DPO Alignment Service implementing DPOAlignmentPort.

Builds chosen/rejected pairs from training data using model inference vs oracle (forward_return),
runs DPOTrainer, and computes calibration metrics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from alpha_signal.application.ports import DPOAlignmentPort

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
            prompt = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
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


    def _run_dpo_trainer(dataset: Any, output_dir: Path) -> dict[str, float]:
        """Run TRL DPOTrainer and compute calibration metric."""
        import torch
        from trl import DPOTrainer, DPOConfig
        from unsloth import FastVisionModel

        from config import dpo_cfg

        if not torch.cuda.is_available():
            raise RuntimeError("DPO training requires CUDA")

        model, processor = FastVisionModel.from_pretrained(
            model_name=dpo_cfg.base_model,
            max_seq_length=dpo_cfg.max_seq_length,
            load_in_4bit=dpo_cfg.load_in_4bit,
            dtype=None,
        )
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
            bf16=True,
            logging_steps=1,
            save_steps=50,
            save_total_limit=2,
            remove_unused_columns=False,
            # Required for newer TRL to process chat templates correctly
            dataset_num_proc=1,
        )

        # For Unsloth/TRL compatibility with Vision DPO
        # Passing None to data_collator lets DPOTrainer use its default DataCollatorForPreference
        trainer = DPOTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            processing_class=processor,
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
