r"""
Simple Temperature Scaling for GPT-2 Instruct Model

This module provides simple temperature scaling for calibrating
model confidence in text generation and Best-of-N candidate scoring.

Key Equations:
- Temperature Scaling: $\text{softmax}(z_i / T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$
  - Higher T (>1) softens probabilities (more uniform, higher uncertainty)
  - Lower T (<1) sharpens probabilities (more peaked, higher confidence)

- Length-Normalized Log-Likelihood Score (Best-of-N):
  $s(x, y) = \frac{1}{|y|} \sum_{t=1}^{|y|} \log p_\theta(y_t | x, y_{<t}; T_{\text{calib}})$
  - Computes log of geometric mean of token probabilities
  - Normalizes by sequence length |y| to eliminate short-response bias
"""

import torch
import torch.nn.functional as F
from typing import Dict
from config import ChatConfig


class SimpleTemperatureScaler:
    """
    Simple temperature scaling for confidence calibration and candidate scoring.

    Temperature is a configurable parameter that scales the logits
    before applying softmax to adjust the confidence distribution.
    """

    def __init__(self, config: ChatConfig):
        """
        Initialize temperature scaler from config.

        Args:
            config: ChatConfig containing calibration_temperature
        """
        self.temperature = config.calibration_temperature

    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        r"""
        Apply temperature scaling to logits.

        Equation: $\text{scaled\_logits} = \frac{\text{logits}}{T}$

        Args:
            logits: Raw model outputs [B, seq_len, V] or [seq_len, V]

        Returns:
            Scaled logits
        """
        return logits / self.temperature

    def score_candidate(
            self,
            logits: torch.Tensor,
            generated_tokens: torch.Tensor,
            pad_token_id: int
    ) -> Dict[str, float]:
        r"""
        Compute length-normalized log-likelihood score for a single candidate response:

        $s(x, y) = \frac{1}{|y|} \sum_{t=1}^{|y|} \log p(y_t | x, y_{<t}; T_{\text{calib}})$

        Args:
            logits: Logit tensor for generated tokens [seq_len, vocab_size]
            generated_tokens: Token IDs for generated sequence [seq_len]
            pad_token_id: ID used for padding tokens

        Returns:
            Dictionary containing length-normalized score, geometric mean prob, and token count.
        """
        if generated_tokens.dim() == 1:
            generated_tokens = generated_tokens.unsqueeze(0)
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)

        # Create valid mask (exclude pad tokens)
        valid_mask = (generated_tokens != pad_token_id)
        valid_len = valid_mask.sum().item()

        if valid_len == 0:
            return {'score': -float('inf'), 'geom_mean_prob': 0.0, 'token_count': 0}

        # Apply temperature scaling
        scaled_logits = self.scale_logits(logits)  # [1, seq_len, V]

        # Log-softmax over vocabulary
        log_probs = F.log_softmax(scaled_logits, dim=-1)  # [1, seq_len, V]

        # Extract log_prob of actual generated tokens
        seq_len = generated_tokens.shape[1]
        token_log_probs = log_probs[0, torch.arange(seq_len), generated_tokens[0]]  # [seq_len]

        # Mask pad tokens and average log-probabilities
        valid_log_probs = token_log_probs[valid_mask[0]]
        score = valid_log_probs.sum().item() / valid_len
        geom_mean_prob = torch.exp(torch.tensor(score)).item()

        return {
            'score': score,
            'geom_mean_prob': geom_mean_prob,
            'token_count': valid_len
        }

    def compute_confidence_metrics(
            self,
            logits: torch.Tensor,
            tokens: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute simple confidence metrics from logits.

        Args:
            logits: Model logits [B, seq_len, V]
            tokens: Generated token IDs [B, seq_len]

        Returns:
            Dictionary with confidence metrics
        """
        # Apply temperature scaling
        scaled_logits = self.scale_logits(logits)

        # Compute softmax probabilities
        probs = F.softmax(scaled_logits, dim=-1)

        # Max probability per position (confidence)
        max_probs = torch.max(probs, dim=-1)[0]

        # Probability of actual generated tokens
        B, seq_len, V = probs.shape
        token_probs = probs[torch.arange(B)[:, None],
        torch.arange(seq_len)[None, :],
        tokens]

        # Aggregate metrics
        metrics = {
            'avg_max_prob': max_probs.mean().item(),
            'avg_token_prob': token_probs.mean().item(),
            'min_token_prob': token_probs.min().item(),
            'temperature': self.temperature
        }

        return metrics

    def set_temperature(self, temperature: float):
        """
        Update temperature parameter.

        Args:
            temperature: New temperature value
        """
        self.temperature = temperature