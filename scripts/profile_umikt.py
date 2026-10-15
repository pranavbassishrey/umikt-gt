from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


import time
import torch
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.umikt_gat import UMiKTGATModel, make_default_config
from src.features.feature_builder import N_CONCEPTS, INPUT_DIM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = make_default_config()
config.update({
    "hidden_dim": 64,
    "use_misconception_head": False,
    "use_retention": False,
    "graph_path": "data/foundational_assist_concept_graph.json",
})

model = UMiKTGATModel(config).to(device)
x = torch.randn(N_CONCEPTS, 15, INPUT_DIM, device=device)
states = torch.rand(N_CONCEPTS, 4, device=device)

for _ in range(5):
    out = model(x, states)
    loss = out["mastery"].mean()
    loss.backward()
    model.zero_grad(set_to_none=True)

if device.type == "cuda":
    torch.cuda.synchronize()

n = 30
start = time.perf_counter()
for _ in range(n):
    model.zero_grad(set_to_none=True)
    out = model(x, states)
    out["mastery"].mean().backward()

if device.type == "cuda":
    torch.cuda.synchronize()

print(f"Device: {device}")
print(f"Forward + backward: {(time.perf_counter() - start) / n * 1000:.2f} ms/step")
print(f"GPU allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MiB"
      if device.type == "cuda" else "GPU not used")
