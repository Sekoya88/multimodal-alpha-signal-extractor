"""reward_scorer_adapter.py — Adapter for Visual Reward Model.

Architecture: frozen Qwen2.5-VL backbone + 2-layer MLP head → scalar [0,1].
Training data: past predictions labeled by realized return
(return > 2% for BUY = reward 1.0, else 0.0).

Compatible with Apple Silicon M4 (CPU/MPS) and Colab T4 (CUDA).
"""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from alpha_signal.application.ports import RewardScorerPort

logger = logging.getLogger(__name__)


# ============================================================================
# MLP Reward Head
# ============================================================================

class RewardMLP(nn.Module):
    """2-layer MLP head mapping VLM hidden states → scalar reward [0, 1].

    Input: concatenation of [pooled_visual_features, action_embedding, confidence]
    Output: sigmoid-activated scalar reward.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        # Action embedding: 3 actions (BUY=0, SELL=1, HOLD=2) → 16-dim
        self.action_embedding = nn.Embedding(3, 16)
        # input_dim (visual features) + 16 (action emb) + 1 (confidence scalar)
        mlp_input = input_dim + 16 + 1
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        action_ids: torch.Tensor,
        confidences: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            visual_features: (B, D) pooled visual features from frozen VLM.
            action_ids: (B,) integer action IDs (0=BUY, 1=SELL, 2=HOLD).
            confidences: (B,) float confidence values.

        Returns:
            (B,) reward scores in [0, 1].
        """
        action_emb = self.action_embedding(action_ids)       # (B, 16)
        conf = confidences.unsqueeze(-1)                      # (B, 1)
        x = torch.cat([visual_features, action_emb, conf], dim=-1)
        return torch.sigmoid(self.mlp(x)).squeeze(-1)


# ============================================================================
# Action mapping
# ============================================================================

ACTION_TO_ID = {"BUY": 0, "SELL": 1, "HOLD": 2}


# ============================================================================
# Adapter
# ============================================================================

class RewardScorerAdapter(RewardScorerPort):
    """Reward scorer: frozen Qwen2.5-VL backbone + trainable 2-layer MLP head.

    The VLM backbone is loaded once and frozen. Only the MLP head is trained.
    Compatible with CPU (Apple Silicon) and CUDA (Colab T4).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        device: str | None = None,
    ):
        self._hidden_dim = hidden_dim
        self._dropout = dropout
        self._device = device or self._detect_device()
        self._backbone = None
        self._processor = None
        self._visual_dim: int | None = None
        self._mlp: RewardMLP | None = None

    @staticmethod
    def _detect_device() -> str:
        """Detect best available device."""
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_backbone(self) -> None:
        """Lazy-load and freeze VLM backbone."""
        if self._backbone is not None:
            return

        logger.info("Loading frozen Qwen2.5-VL backbone for reward scoring...")
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            trust_remote_code=True,
        )
        self._backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            device_map=self._device if self._device == "cuda" else None,
            trust_remote_code=True,
        )
        # Freeze all backbone parameters
        for param in self._backbone.parameters():
            param.requires_grad = False
        self._backbone.eval()

        if self._device != "cuda":
            self._backbone = self._backbone.to(self._device)

        # Detect visual feature dimension from the model config
        hidden_size = self._backbone.config.hidden_size
        self._visual_dim = hidden_size
        logger.info(f"  Backbone loaded (hidden_size={hidden_size}, device={self._device})")

        # Initialize MLP head
        self._mlp = RewardMLP(
            input_dim=self._visual_dim,
            hidden_dim=self._hidden_dim,
            dropout=self._dropout,
        ).to(self._device)
        logger.info(f"  MLP head initialized ({self._hidden_dim} hidden dim)")

    def _extract_visual_features(self, image_path: Path) -> torch.Tensor:
        """Extract pooled visual features from an image using the frozen backbone.

        Returns:
            (1, D) tensor of pooled visual features.
        """
        self._load_backbone()

        image = Image.open(image_path).convert("RGB")

        # Use processor to prepare inputs for the model
        text_prompt = "Describe the trading chart."
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text_prompt},
            ]},
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._backbone(**inputs, output_hidden_states=True)
            # Pool the last hidden state (mean over sequence)
            last_hidden = outputs.hidden_states[-1]  # (1, seq_len, D)
            pooled = last_hidden.mean(dim=1)          # (1, D)

        return pooled.float()

    def score(
        self,
        image_path: Path,
        predicted_action: str,
        predicted_confidence: float,
    ) -> float:
        """Score a (chart, prediction) pair using frozen backbone + MLP head."""
        self._load_backbone()
        if self._mlp is None:
            raise RuntimeError("MLP head not initialized. Call load_weights() first.")

        self._mlp.eval()
        visual_features = self._extract_visual_features(image_path)

        action_id = ACTION_TO_ID.get(predicted_action.upper(), 2)
        action_tensor = torch.tensor([action_id], device=self._device)
        conf_tensor = torch.tensor([predicted_confidence], dtype=torch.float32, device=self._device)

        with torch.no_grad():
            reward = self._mlp(visual_features, action_tensor, conf_tensor)

        return float(reward.item())

    def train(self, data_path: Path) -> dict[str, float]:
        """Train the MLP reward head on labeled data.

        Data format (JSONL): each line has:
          - image_b64: base64-encoded chart PNG
          - predicted_action: BUY/SELL/HOLD
          - predicted_confidence: float
          - reward: float (0.0 or 1.0)
        """
        self._load_backbone()
        if self._mlp is None:
            raise RuntimeError("MLP head not initialized. Load backbone first.")

        from config import reward_model_cfg

        # Load training data
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Reward training data not found: {data_path}")

        records: list[dict[str, Any]] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        if not records:
            raise ValueError(f"No training records in {data_path}")

        logger.info(f"Training reward model MLP on {len(records)} samples...")

        # Pre-extract visual features (frozen backbone, no grad)
        features_list: list[torch.Tensor] = []
        action_ids: list[int] = []
        confidences: list[float] = []
        rewards: list[float] = []

        for i, rec in enumerate(records):
            # Decode image from base64
            img_bytes = base64.b64decode(rec["image_b64"])
            tmp_path = Path(f"/tmp/reward_train_{i}.png")
            tmp_path.write_bytes(img_bytes)

            feats = self._extract_visual_features(tmp_path)
            features_list.append(feats.cpu())

            action_ids.append(ACTION_TO_ID.get(rec["predicted_action"].upper(), 2))
            confidences.append(float(rec["predicted_confidence"]))
            rewards.append(float(rec["reward"]))

            tmp_path.unlink(missing_ok=True)

            if (i + 1) % 10 == 0:
                logger.info(f"  Extracted features for {i + 1}/{len(records)} samples")

        # Stack into tensors
        all_features = torch.cat(features_list, dim=0).to(self._device)   # (N, D)
        all_actions = torch.tensor(action_ids, device=self._device)        # (N,)
        all_confs = torch.tensor(confidences, dtype=torch.float32, device=self._device)
        all_rewards = torch.tensor(rewards, dtype=torch.float32, device=self._device)

        # Train MLP head
        self._mlp.train()
        optimizer = torch.optim.AdamW(
            self._mlp.parameters(),
            lr=reward_model_cfg.learning_rate,
            weight_decay=reward_model_cfg.weight_decay,
        )
        criterion = nn.BCELoss()

        n = len(records)
        batch_size = reward_model_cfg.per_device_train_batch_size
        epoch_losses: list[float] = []

        for epoch in range(reward_model_cfg.num_train_epochs):
            # Shuffle indices
            perm = torch.randperm(n)
            running_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                pred = self._mlp(
                    all_features[idx],
                    all_actions[idx],
                    all_confs[idx],
                )
                loss = criterion(pred, all_rewards[idx])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                n_batches += 1

            avg_loss = running_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)
            logger.info(f"  Epoch {epoch + 1}/{reward_model_cfg.num_train_epochs} — loss: {avg_loss:.4f}")

        # Compute training accuracy
        self._mlp.eval()
        with torch.no_grad():
            all_pred = self._mlp(all_features, all_actions, all_confs)
            pred_binary = (all_pred > 0.5).float()
            accuracy = float((pred_binary == all_rewards).float().mean().item())

        metrics = {
            "final_loss": epoch_losses[-1] if epoch_losses else 0.0,
            "accuracy": accuracy,
            "num_samples": n,
            "epoch_losses": epoch_losses,
        }

        # Save MLP weights
        reward_model_cfg.output_dir.mkdir(parents=True, exist_ok=True)
        weights_path = reward_model_cfg.output_dir / "mlp_head.pt"
        torch.save(self._mlp.state_dict(), weights_path)
        logger.info(f"  MLP weights saved → {weights_path}")

        return metrics

    def load_weights(self, weights_path: Path) -> None:
        """Load previously trained MLP head weights."""
        self._load_backbone()
        if self._mlp is None:
            raise RuntimeError("MLP head not initialized. Load backbone first.")

        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"MLP weights not found: {weights_path}")

        state_dict = torch.load(weights_path, map_location=self._device, weights_only=True)
        self._mlp.load_state_dict(state_dict)
        self._mlp.eval()
        logger.info(f"  MLP weights loaded from {weights_path}")
