"""
Deep Knowledge Tracing (DKT) baseline.

DKT (Piech et al. 2015) uses a single LSTM over skill-question interactions,
predicting the probability of correct response for each skill.

This is a standard KT baseline, not a novel contribution.
Used for external comparison against UMiKT-GAT on mastery prediction.

Architecture:
  Input: one-hot(skill_id, correctness) → LSTM → output projection per skill
  Input dim = n_skills * 2 (one-hot over skill+correctness pairs)
  Output dim = n_skills (probability per skill)

Training: BCE loss on next-question correctness prediction.
"""

import torch
import torch.nn as nn
from torch import Tensor


class DKTModel(nn.Module):
    """
    Deep Knowledge Tracing LSTM model (Piech et al. 2015).

    Args:
        n_skills: Number of skills/concepts.
        hidden_dim: LSTM hidden dimension.
        dropout_rate: Dropout between LSTM layers.
        n_layers: Number of LSTM layers.
    """

    def __init__(
        self,
        n_skills: int = 7,
        hidden_dim: int = 64,
        dropout_rate: float = 0.3,
        n_layers: int = 2,
    ):
        super().__init__()
        self.n_skills = n_skills

        # DKT input: (skill_id, correctness) encoded as n_skills*2 one-hot
        input_dim = n_skills * 2
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_rate if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.out_proj = nn.Linear(hidden_dim, n_skills)

    def _encode_input(self, skill_id: int, correct: int) -> Tensor:
        """
        Encode (skill_id, correct) as a one-hot vector of dim n_skills * 2.

        For correct=1: set bit at skill_id.
        For correct=0: set bit at n_skills + skill_id.
        """
        x = torch.zeros(self.n_skills * 2)
        if correct:
            x[skill_id] = 1.0
        else:
            x[self.n_skills + skill_id] = 1.0
        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input sequence. Shape: (batch, seq_len, n_skills*2)

        Returns:
            Mastery logits for all skills at each timestep.
            Shape: (batch, seq_len, n_skills)
        """
        out, _ = self.lstm(x)
        out = self.dropout(out)
        return self.out_proj(out)

    def predict_sequence(
        self,
        skill_ids: list[int],
        correctness: list[int],
        device: str = "cpu",
    ) -> list[float]:
        """
        Predict mastery for a sequence of interactions.

        Args:
            skill_ids: List of skill IDs for each interaction.
            correctness: List of correctness values (0 or 1).

        Returns:
            List of mastery predictions for the primary skill at each step.
        """
        if not skill_ids:
            return []

        # Build input sequence
        inputs = [
            self._encode_input(s, c)
            for s, c in zip(skill_ids, correctness)
        ]
        x = torch.stack(inputs, dim=0).unsqueeze(0).to(device)  # (1, T, n_skills*2)

        with torch.no_grad():
            logits = self.forward(x)  # (1, T, n_skills)
            probs = torch.sigmoid(logits)  # (1, T, n_skills)

        # Return mastery probability for the queried skill at each step
        preds = []
        for t, s in enumerate(skill_ids):
            preds.append(float(probs[0, t, s].item()))

        return preds


class DKTTrainer:
    """Trains a DKTModel using next-response prediction."""

    def __init__(
        self,
        n_skills: int = 7,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        n_epochs: int = 30,
        device: str = "cpu",
    ):
        self.n_skills = n_skills
        self.device = device
        self.n_epochs = n_epochs

        self.model = DKTModel(n_skills=n_skills, hidden_dim=hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def _make_sequence_tensor(
        self, records: list[dict]
    ) -> tuple[Tensor, Tensor, list[int]]:
        """
        Build (X, y_skill_ids, y_correct) for a learner's sequence.

        Returns:
            x: Input tensor. Shape: (T-1, n_skills*2)
            y: Target correctness. Shape: (T-1,)
            y_skills: Skill IDs for next-step prediction.
        """
        T = len(records)
        if T < 2:
            return None, None, []

        inputs = []
        targets = []
        target_skills = []

        for i in range(T - 1):
            r = records[i]
            r_next = records[i + 1]
            s = int(r.get("concept_id", 0)) % self.n_skills
            c = int(float(r.get("correctness", 0)))
            s_next = int(r_next.get("concept_id", 0)) % self.n_skills
            c_next = float(r_next.get("correctness", 0))

            inp = self.model._encode_input(s, c)
            inputs.append(inp)
            targets.append(c_next)
            target_skills.append(s_next)

        x = torch.stack(inputs).to(self.device)
        y = torch.tensor(targets, dtype=torch.float32).to(self.device)
        return x, y, target_skills

    def train(self, records: list[dict]):
        """Train DKT on a list of interaction records."""
        from collections import defaultdict

        # Group by learner
        by_learner = defaultdict(list)
        for r in records:
            by_learner[r.get("learner_id", "unknown")].append(r)
        for lid in by_learner:
            by_learner[lid].sort(key=lambda r: r.get("interaction_id", 0))

        self.model.train()

        for epoch in range(self.n_epochs):
            total_loss = 0.0
            n_batches = 0

            for lid, seq in by_learner.items():
                x, y, y_skills = self._make_sequence_tensor(seq)
                if x is None or len(x) < 1:
                    continue

                x = x.unsqueeze(0)  # (1, T-1, n_skills*2)
                logits = self.model(x)  # (1, T-1, n_skills)

                # Gather predicted logit for each next skill
                skill_tensor = torch.tensor(y_skills, dtype=torch.long).to(self.device)
                pred_logits = logits[0, range(len(y_skills)), skill_tensor]  # (T-1,)

                loss = self.loss_fn(pred_logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 10 == 0:
                avg = total_loss / max(n_batches, 1)
                print(f"  DKT epoch {epoch+1}/{self.n_epochs}, loss={avg:.4f}")

    @torch.no_grad()
    def predict_all(self, records: list[dict], skill_col: str = "concept_id") -> list[float]:
        """Predict mastery for all records."""
        from collections import defaultdict
        by_learner = defaultdict(list)
        for r in records:
            by_learner[r.get("learner_id", "unknown")].append(r)
        for lid in by_learner:
            by_learner[lid].sort(key=lambda r: r.get("interaction_id", 0))

        self.model.eval()
        # Build a mapping from record to prediction
        pred_map = {}

        for lid, seq in by_learner.items():
            skill_ids = [int(r.get("concept_id", 0)) % self.n_skills for r in seq]
            correctness = [int(float(r.get("correctness", 0))) for r in seq]
            preds = self.model.predict_sequence(skill_ids, correctness, self.device)
            for i, r in enumerate(seq):
                key = (r.get("learner_id"), r.get("interaction_id", i))
                pred_map[key] = preds[i] if i < len(preds) else 0.3

        predictions = []
        for r in records:
            key = (r.get("learner_id"), r.get("interaction_id", 0))
            predictions.append(pred_map.get(key, 0.3))

        return predictions
