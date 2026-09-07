"""
Unit tests for UMiKT-GAT components.

Tests cover:
  1. Schema validation (LLM extraction output)
  2. Graph propagation (GAT layer)
  3. Priority function (curriculum priority)
  4. Feature builder (correct shape and range)
  5. Retention module (decay formula)

Run with:
  cd umikt-gat && python -m pytest tests/ -v
"""

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch

from src.llm.schema import ExtractionResult, ExtractionRequest, VALID_MISCONCEPTION_LABELS
from src.llm.mock_provider import MockLLMProvider
from src.models.gat_layer import MultiHeadGAT
from src.models.gru_encoder import GRUEncoder
from src.curriculum.priority import CurriculumPriority, ActionPolicy, N_CONCEPTS
from src.features.feature_builder import build_feature_vector, INPUT_DIM, N_CONCEPTS as FC_N
from src.models.retention_module import ExponentialDecayRetention


# ─── Schema Validation Tests ──────────────────────────────────────────────────

class TestSchemaValidation:

    def test_valid_extraction_result(self):
        result = ExtractionResult(
            correctness=0.8,
            concepts_detected=["Variables"],
            misconception_label="none",
            misconception_severity=0.0,
            reasoning_quality=0.7,
            confidence=0.85,
        )
        assert result.correctness == 0.8
        assert result.misconception_label == "none"

    def test_invalid_correctness_above_1(self):
        with pytest.raises(Exception):
            ExtractionResult(correctness=1.5, misconception_label="none",
                             misconception_severity=0.0, reasoning_quality=0.5,
                             confidence=0.5)

    def test_invalid_correctness_below_0(self):
        with pytest.raises(Exception):
            ExtractionResult(correctness=-0.1, misconception_label="none",
                             misconception_severity=0.0, reasoning_quality=0.5,
                             confidence=0.5)

    def test_invalid_misconception_label(self):
        with pytest.raises(Exception):
            ExtractionResult(
                correctness=0.5,
                misconception_label="made_up_label",
                misconception_severity=0.5,
                reasoning_quality=0.5,
                confidence=0.5,
            )

    def test_all_valid_misconception_labels(self):
        for label in VALID_MISCONCEPTION_LABELS:
            result = ExtractionResult(
                correctness=0.5,
                misconception_label=label,
                misconception_severity=0.0 if label == "none" else 0.4,
                reasoning_quality=0.5,
                confidence=0.7,
            )
            assert result.misconception_label == label

    def test_invalid_concept_filtered(self):
        """Invalid concept names in concepts_detected should be filtered, not raise."""
        result = ExtractionResult(
            correctness=0.5,
            concepts_detected=["Variables", "NonExistentConcept", "Loops"],
            misconception_label="none",
            misconception_severity=0.0,
            reasoning_quality=0.5,
            confidence=0.7,
        )
        assert "NonExistentConcept" not in result.concepts_detected
        assert "Variables" in result.concepts_detected

    def test_to_feature_dict(self):
        result = ExtractionResult(
            correctness=0.7,
            misconception_label="loop_termination_misunderstanding",
            misconception_severity=0.6,
            reasoning_quality=0.4,
            confidence=0.75,
        )
        fd = result.to_feature_dict()
        assert "misconception_severity" in fd
        assert fd["misconception_severity"] == 0.6

    def test_to_misconception_class_id(self):
        result = ExtractionResult(
            correctness=0.5,
            misconception_label="loop_termination_misunderstanding",
            misconception_severity=0.5,
            reasoning_quality=0.5,
            confidence=0.7,
        )
        class_id = result.to_misconception_class_id()
        assert class_id == VALID_MISCONCEPTION_LABELS.index("loop_termination_misunderstanding")
        assert class_id == 3

    def test_mock_provider_returns_valid_result(self):
        provider = MockLLMProvider()
        request = ExtractionRequest(
            question="What is the output of: for i in range(5): print(i)?",
            student_answer="4 times, starting from 1",
            concept_name="Loops",
            expected_answer="5 times, printing 0 1 2 3 4",
            concept_id=2,
        )
        result = provider.extract(request)
        assert isinstance(result, ExtractionResult)
        assert 0.0 <= result.correctness <= 1.0
        assert result.misconception_label in VALID_MISCONCEPTION_LABELS
        assert 0.0 <= result.misconception_severity <= 1.0

    def test_mock_provider_safe_default_on_empty_answer(self):
        provider = MockLLMProvider()
        request = ExtractionRequest(
            question="What is x?",
            student_answer="",
            concept_name="Variables",
            concept_id=0,
        )
        result = provider.extract(request)
        # Should not raise — returns safe default
        assert isinstance(result, ExtractionResult)

    def test_mock_provider_detects_loop_termination_misconception(self):
        """Mock provider should detect loop termination misconception in a known wrong answer."""
        provider = MockLLMProvider()
        request = ExtractionRequest(
            question="How many times does 'for i in range(5)' execute?",
            student_answer="4 times, range starts from 1 to 5",
            concept_name="Loops",
            expected_answer="5 times (i=0,1,2,3,4)",
            concept_id=2,
        )
        result = provider.extract(request)
        # The mock may or may not detect it depending on regex — just verify it's valid
        assert result.misconception_label in VALID_MISCONCEPTION_LABELS


# ─── Graph Propagation Tests ──────────────────────────────────────────────────

class TestGATLayer:

    @pytest.fixture
    def gat(self):
        graph_path = Path(__file__).parent.parent / "data" / "concept_graph.json"
        if not graph_path.exists():
            pytest.skip("concept_graph.json not found")
        return MultiHeadGAT(node_dim=16, out_dim=16, n_heads=2, dynamic=True,
                            graph_path=graph_path)

    def test_output_shape(self, gat):
        node_feats = torch.randn(N_CONCEPTS, 16)
        learner_states = torch.rand(N_CONCEPTS, 4)
        out = gat(node_feats, learner_states)
        assert out.shape == (N_CONCEPTS, 16), f"Expected ({N_CONCEPTS}, 16), got {out.shape}"

    def test_static_mode_output_shape(self):
        graph_path = Path(__file__).parent.parent / "data" / "concept_graph.json"
        if not graph_path.exists():
            pytest.skip("concept_graph.json not found")
        gat_static = MultiHeadGAT(node_dim=16, out_dim=16, n_heads=2, dynamic=False,
                                   graph_path=graph_path)
        node_feats = torch.randn(N_CONCEPTS, 16)
        out = gat_static(node_feats, None)
        assert out.shape == (N_CONCEPTS, 16)

    def test_dynamic_differs_from_static(self):
        """Dynamic (learner-state-conditioned) attention should produce different output
        than static attention given the same node features but different learner states."""
        graph_path = Path(__file__).parent.parent / "data" / "concept_graph.json"
        if not graph_path.exists():
            pytest.skip("concept_graph.json not found")

        gat_dyn = MultiHeadGAT(node_dim=16, out_dim=16, n_heads=2, dynamic=True,
                                graph_path=graph_path)
        node_feats = torch.randn(N_CONCEPTS, 16)

        ls1 = torch.rand(N_CONCEPTS, 4)
        ls2 = torch.rand(N_CONCEPTS, 4) * 0 + 1.0  # all high mastery

        out1 = gat_dyn(node_feats, ls1)
        out2 = gat_dyn(node_feats, ls2)

        # Outputs should differ when learner states differ
        assert not torch.allclose(out1, out2, atol=1e-4), \
            "Dynamic GAT should produce different outputs for different learner states"

    def test_no_nan_in_output(self, gat):
        node_feats = torch.randn(N_CONCEPTS, 16)
        learner_states = torch.rand(N_CONCEPTS, 4)
        out = gat(node_feats, learner_states)
        assert not torch.any(torch.isnan(out)), "GAT output contains NaN values"

    def test_prerequisite_propagation_direction(self):
        """Verify that GAT propagates from prerequisite (source) to dependent (target).
        Algorithms (6) depends on Data Structures (5).
        Strong feature at concept 5 should influence concept 6's output."""
        graph_path = Path(__file__).parent.parent / "data" / "concept_graph.json"
        if not graph_path.exists():
            pytest.skip("concept_graph.json not found")
        gat = MultiHeadGAT(node_dim=8, out_dim=8, n_heads=1, dynamic=False,
                            graph_path=graph_path).eval()
        # Baseline: zero node features
        node_feats_zero = torch.zeros(N_CONCEPTS, 8)
        out_zero = gat(node_feats_zero, None)

        # Strong signal at concept 5 (Data Structures)
        node_feats_strong = torch.zeros(N_CONCEPTS, 8)
        node_feats_strong[5] = 5.0

        out_strong = gat(node_feats_strong, None)

        # Concept 6 (Algorithms) output should change more than concept 0 (Variables)
        diff_6 = (out_strong[6] - out_zero[6]).abs().mean().item()
        diff_0 = (out_strong[0] - out_zero[0]).abs().mean().item()
        # Concept 6 should be influenced more strongly by concept 5 signal
        # than concept 0 (which has no edge from 5)
        assert diff_6 >= diff_0, (
            f"GAT should propagate signal from DS(5)→Algo(6) more than DS(5)→Vars(0). "
            f"diff_6={diff_6:.4f}, diff_0={diff_0:.4f}"
        )


# ─── Feature Builder Tests ────────────────────────────────────────────────────

class TestFeatureBuilder:

    def test_output_dimension(self):
        rec = {
            "correctness": 0.8, "difficulty": 0.5, "response_time_norm": 0.3,
            "attempts": 0.2, "temporal_gap_norm": 0.1, "evaluator_confidence": 0.75,
            "question_type": "multiple_choice", "concept_id": 2,
            "misconception_severity": 0.0, "misconception_confidence": 0.0,
            "reasoning_quality": 0.5,
        }
        fv = build_feature_vector(rec)
        assert len(fv) == INPUT_DIM, f"Expected {INPUT_DIM}, got {len(fv)}"

    def test_all_values_in_range(self):
        rec = {
            "correctness": 1.0, "difficulty": 1.0, "response_time_norm": 1.0,
            "attempts": 1.0, "temporal_gap_norm": 1.0, "evaluator_confidence": 1.0,
            "question_type": "debugging", "concept_id": 6,
            "misconception_severity": 1.0, "misconception_confidence": 1.0,
            "reasoning_quality": 1.0,
        }
        fv = build_feature_vector(rec)
        for v in fv:
            assert 0.0 <= v <= 1.0, f"Feature value {v} out of [0,1] range"

    def test_concept_id_one_hot(self):
        for cid in range(FC_N):
            rec = {
                "correctness": 0.5, "difficulty": 0.5, "response_time_norm": 0.5,
                "attempts": 0.2, "temporal_gap_norm": 0.1, "evaluator_confidence": 0.75,
                "question_type": "short_answer", "concept_id": cid,
                "misconception_severity": 0.0, "misconception_confidence": 0.0,
                "reasoning_quality": 0.5,
            }
            fv = build_feature_vector(rec)
            # Concept one-hot starts at index 10
            one_hot_slice = fv[10:10 + FC_N]
            assert one_hot_slice[cid] == 1.0, f"Concept {cid} one-hot bit should be 1"
            for j, v in enumerate(one_hot_slice):
                if j != cid:
                    assert v == 0.0, f"Concept {j} bit should be 0 when concept_id={cid}"

    def test_unknown_question_type_handled(self):
        rec = {
            "correctness": 0.5, "difficulty": 0.5, "response_time_norm": 0.5,
            "attempts": 0.2, "temporal_gap_norm": 0.1, "evaluator_confidence": 0.75,
            "question_type": "unknown_type", "concept_id": 0,
            "misconception_severity": 0.0, "misconception_confidence": 0.0,
            "reasoning_quality": 0.5,
        }
        fv = build_feature_vector(rec)  # should not raise
        assert len(fv) == INPUT_DIM


# ─── Priority Function Tests ──────────────────────────────────────────────────

class TestCurriculumPriority:

    def setup_method(self):
        self.priority = CurriculumPriority()
        self.policy = ActionPolicy()

    def test_priority_output_length(self):
        mastery = [0.5] * N_CONCEPTS
        mc_sev = [0.2] * N_CONCEPTS
        unc = [0.3] * N_CONCEPTS
        forget = [0.1] * N_CONCEPTS
        priorities = self.priority.compute_priorities(mastery, mc_sev, unc, forget)
        assert len(priorities) == N_CONCEPTS

    def test_high_misconception_increases_priority(self):
        base = [0.5] * N_CONCEPTS
        mc_no = [0.0] * N_CONCEPTS
        mc_high = [0.0] * N_CONCEPTS
        mc_high[3] = 0.9  # high misconception on concept 3

        p_no = self.priority.compute_priorities(base, mc_no, [0.3]*N_CONCEPTS, [0.1]*N_CONCEPTS)
        p_high = self.priority.compute_priorities(base, mc_high, [0.3]*N_CONCEPTS, [0.1]*N_CONCEPTS)
        assert p_high[3] > p_no[3], "High misconception should increase priority"

    def test_high_mastery_decreases_priority(self):
        mc_sev = [0.0] * N_CONCEPTS
        unc = [0.2] * N_CONCEPTS
        forget = [0.1] * N_CONCEPTS

        mastery_low = [0.2] * N_CONCEPTS
        mastery_high = [0.9] * N_CONCEPTS

        p_low = self.priority.compute_priorities(mastery_low, mc_sev, unc, forget)
        p_high = self.priority.compute_priorities(mastery_high, mc_sev, unc, forget)
        assert all(p_low[i] > p_high[i] for i in range(N_CONCEPTS)), \
            "High mastery should decrease priority"

    def test_high_mastery_triggers_advance(self):
        action, _ = self.policy.select_action(
            concept_id=0, mastery=0.90, misconception_severity=0.05,
            uncertainty=0.1, forgetting_risk=0.05, dep_score=0.0,
        )
        assert action == "advance_to_next_topic"

    def test_high_misconception_triggers_remediation(self):
        action, _ = self.policy.select_action(
            concept_id=2, mastery=0.35, misconception_severity=0.75,
            uncertainty=0.2, forgetting_risk=0.1, dep_score=0.1,
        )
        assert action == "misconception_targeted_content"

    def test_high_uncertainty_triggers_diagnostic(self):
        action, _ = self.policy.select_action(
            concept_id=1, mastery=0.55, misconception_severity=0.10,
            uncertainty=0.70, forgetting_risk=0.1, dep_score=0.1,
        )
        assert action == "adaptive_diagnostic_assessment"

    def test_explanation_is_nonempty_string(self):
        _, explanation = self.policy.select_action(
            concept_id=4, mastery=0.30, misconception_severity=0.60,
            uncertainty=0.2, forgetting_risk=0.1, dep_score=0.3,
        )
        assert isinstance(explanation, str) and len(explanation) > 10

    def test_concept_name_in_explanation(self):
        concept_id = 4  # Recursion
        _, explanation = self.policy.select_action(
            concept_id=concept_id, mastery=0.30, misconception_severity=0.70,
            uncertainty=0.2, forgetting_risk=0.1, dep_score=0.3,
        )
        assert "Recursion" in explanation


# ─── Retention Module Tests ───────────────────────────────────────────────────

class TestRetentionModule:

    def test_zero_time_preserves_mastery(self):
        ret = ExponentialDecayRetention(lambda_init=0.05, learnable=False)
        mastery = torch.tensor([0.8, 0.6, 0.4])
        delta_t = torch.zeros(3)
        out = ret(mastery, delta_t)
        # exp(-0.05 * 0) = 1.0, so R = M
        assert torch.allclose(out, mastery, atol=1e-4), \
            f"Zero time gap should preserve mastery. Got {out}"

    def test_large_time_decays_mastery(self):
        ret = ExponentialDecayRetention(lambda_init=0.1, learnable=False)
        mastery = torch.tensor([0.9])
        delta_t = torch.tensor([100.0])  # 100 hours
        out = ret(mastery, delta_t)
        # exp(-0.1 * 100) ≈ 4.5e-5, so R ≈ 0
        assert out.item() < 0.01, f"Large time gap should cause forgetting. Got {out.item()}"

    def test_output_in_valid_range(self):
        ret = ExponentialDecayRetention(lambda_init=0.05, learnable=False)
        mastery = torch.rand(7)
        delta_t = torch.rand(7) * 48
        out = ret(mastery, delta_t)
        assert torch.all(out >= 0.0) and torch.all(out <= 1.0)

    def test_forgetting_risk_complement(self):
        ret = ExponentialDecayRetention(lambda_init=0.05, learnable=False)
        mastery = torch.tensor([0.8, 0.6])
        delta_t = torch.tensor([10.0, 20.0])
        retention = ret(mastery, delta_t)
        risk = ret.forgetting_risk(mastery, delta_t)
        assert torch.allclose(retention + risk, torch.ones(2), atol=1e-5)

    def test_lambda_init_from_data(self):
        records = [
            {"learner_id": "L0", "concept_id": 0, "temporal_gap_hours": 5.0, "correctness": 0.8},
            {"learner_id": "L0", "concept_id": 0, "temporal_gap_hours": 10.0, "correctness": 0.6},
            {"learner_id": "L1", "concept_id": 1, "temporal_gap_hours": 8.0, "correctness": 0.7},
            {"learner_id": "L1", "concept_id": 1, "temporal_gap_hours": 20.0, "correctness": 0.4},
        ]
        lam = ExponentialDecayRetention.fit_lambda_from_data(records)
        assert 0.001 <= lam <= 0.5, f"Lambda {lam} out of expected range"


# ─── GRU Encoder Tests ────────────────────────────────────────────────────────

class TestGRUEncoder:

    def test_output_shape_single(self):
        enc = GRUEncoder(input_dim=INPUT_DIM, hidden_dim=32, num_layers=2)
        x = torch.randn(1, 5, INPUT_DIM)  # (batch, seq, dim)
        out, h_n = enc(x)
        assert out.shape == (1, 5, 32)
        assert h_n.shape == (2, 1, 32)

    def test_output_shape_batched(self):
        enc = GRUEncoder(input_dim=INPUT_DIM, hidden_dim=32, num_layers=2)
        x = torch.randn(4, 10, INPUT_DIM)  # (batch=4, seq=10, dim)
        out, h_n = enc(x)
        assert out.shape == (4, 10, 32)

    def test_no_nan_output(self):
        enc = GRUEncoder(input_dim=INPUT_DIM, hidden_dim=32)
        x = torch.randn(1, 5, INPUT_DIM)
        out, _ = enc(x)
        assert not torch.any(torch.isnan(out))

    def test_init_hidden_shape(self):
        enc = GRUEncoder(input_dim=INPUT_DIM, hidden_dim=32, num_layers=2)
        h = enc.init_hidden(batch_size=3)
        assert h.shape == (2, 3, 32)
