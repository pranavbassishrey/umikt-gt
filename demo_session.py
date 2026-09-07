"""
Demo session script — runs one complete closed-loop adaptive session.

Demonstrates the full UMiKT-GAT pipeline end-to-end:
  1. Loads the trained full model (must run run_ablation.py first)
  2. Runs a 5-interaction adaptive session for a synthetic demo learner
  3. Prints per-step learner state, decision, and human-readable explanation
  4. Saves system cost metrics to experiments/results/system_cost.json

All data is SYNTHETIC. The student answers are simulated, not real.

Usage:
  python demo_session.py
  python demo_session.py --learner demo_student_001 --steps 8
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from src.models.umikt_gat import UMiKTGATModel, make_ablation_config
from src.agents.orchestrator import AgentOrchestrator


def main():
    parser = argparse.ArgumentParser(description="UMiKT-GAT Demo Session")
    parser.add_argument("--learner", type=str, default="demo_learner_001")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained full model
    config = make_ablation_config("full_umikt_gat")
    model = UMiKTGATModel(config).to(device)
    model.eval()

    checkpoint = Path("experiments/results/full_umikt_gat/best_model.pt")
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print("[WARNING] No trained checkpoint found. Using random weights.")
        print("Run 'python src/run_ablation.py' first for meaningful results.")

    # Run session
    log_dir = Path("experiments/results/agent_logs")
    orchestrator = AgentOrchestrator(model=model, device=str(device), log_dir=log_dir)

    session_result = orchestrator.run_session(
        learner_id=args.learner,
        n_interactions=args.steps,
        simulate_wrong_answers=True,
        rng_seed=args.seed,
    )

    # Save system cost
    cost_path = Path("experiments/results/system_cost.json")
    cost_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cost_path, "w") as f:
        json.dump(session_result["system_cost"], f, indent=2)
    print(f"\nSystem cost saved: {cost_path}")

    # Save full session log
    log_path = Path("experiments/results/demo_session_log.json")
    with open(log_path, "w") as f:
        json.dump(session_result, f, indent=2, default=str)
    print(f"Session log saved:  {log_path}")

    print("\nFinal Learner State:")
    for concept, state in session_result["final_state"].items():
        print(f"  {concept:<18}: M={state['M']:.3f}  MC={state['MC_sev']:.3f}  "
              f"U={state['U']:.3f}  R={state['R']:.3f}")


if __name__ == "__main__":
    main()
