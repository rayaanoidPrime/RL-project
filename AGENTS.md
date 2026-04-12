# AGENTS.md — Instructions for OpenAI Codex Agent

## Project overview

You are helping reproduce the experiments from the NeurIPS 2025 paper:
**"Robust LLM Alignment via Distributionally Robust Direct Preference Optimization"**
by Xu et al. (arXiv:2502.01930).

The paper proposes two algorithms, WDPO and KLDPO, that make DPO robust to
preference distribution shift. The official code is at:
https://github.com/TheBlackCat22/distributionally_robust_dpo

**Your primary directive:** Use the paper's existing code wherever possible.
Do NOT rewrite their algorithms. Wire, configure, and debug — don't replace.

---

## Repository layout

After running `bash setup_local.sh` the project looks like this:

```
.
├── distributionally_robust_dpo/   # paper's repo (do not modify core src/)
│   ├── src/
│   │   ├── setup.py               # downloads models + datasets
│   │   ├── generate_completions.py
│   │   ├── generate_preferences.py
│   │   └── train_preference.py    # main training script (DeepSpeed)
│   ├── generate_completions.sh
│   ├── train_alignment.sh
│   ├── leaderboard_eval.sh
│   └── setup.sh                   # original conda-based setup (reference only)
├── modal_jobs.py                  # Modal wrappers around the paper's scripts
├── setup_local.sh                 # uv-based local dev setup
├── .env.example                   # env vars template
└── AGENTS.md                      # this file
```

---

## Environment

- Local dev: Python 3.11 venv managed by `uv`, activated via `source .venv/bin/activate`
- Remote GPU: Modal (cloud GPU provider). All heavy jobs run there.
- GPU training uses DeepSpeed ZeRO-2 + bfloat16 exactly as in the paper.
- The paper's original code uses conda + SLURM. We replace both with Modal.

---

## Completed work (do not redo)

1. `setup_local.sh` — uv environment + repo clone
2. `modal_jobs.py` — Modal functions wrapping every paper experiment
3. `.env.example` — token template

---

## Your tasks (in priority order)

### TASK 1 — Create a Modal secret for HuggingFace

Run this once to register the HF token with Modal:
```bash
modal secret create huggingface-secret HF_TOKEN=<the_token_from_.env>
```
If this is already done, skip it.

---

### TASK 2 — Verify the Modal image builds

Run the smoke test to make sure the Docker image builds correctly and all
imports work:
```bash
modal run modal_jobs.py::smoke_test
```

If it fails, debug the image build in `modal_jobs.py`. Common issues:
- `flash-attn` build failures: pin to a wheel that matches torch 2.5.1 + cu124.
  Try: `flash-attn==2.7.2.post1` or fetch the correct wheel from
  https://github.com/Dao-AILab/flash-attention/releases
- `openrlhf[vllm]==0.6.1.post1` may have transitive version conflicts.
  If so, install without vllm first: `openrlhf==0.6.1.post1`, then `vllm`
  separately at a compatible version.
- DeepSpeed build: if `DS_BUILD_FUSED_ADAM=1` fails in the image, try
  `DS_BUILD_FUSED_ADAM=0` as a fallback (fused Adam is an optimization,
  not required for correctness).

Do NOT change the paper's `src/` files to fix image issues.

---

### TASK 3 — Download assets

```bash
modal run modal_jobs.py::download_assets
```

This calls the paper's `src/setup.py`. If that file doesn't exist in the
cloned repo (it may not be public), create a minimal replacement at
`distributionally_robust_dpo/src/setup.py` that does the following and
nothing else:

```python
# Minimal setup.py — only create this if the paper's version is missing.
from huggingface_hub import snapshot_download
import os

MODELS = {
    "models/Llama1b":  "meta-llama/Llama-3.2-1B-Instruct",
    "models/Llama3b":  "meta-llama/Llama-3.2-3B-Instruct",
    "models/Llama8b":  "meta-llama/Llama-3.1-8B-Instruct",
    "models/ArmoRM":   "RLHFlow/ArmoRM-Llama3-8B-v0.1",
}
DATASETS = {
    "datasets/helpsteer2_prompts": ("Nvidia/HelpSteer2", None),
}

os.makedirs("models", exist_ok=True)
os.makedirs("datasets", exist_ok=True)

for local_path, hf_id in MODELS.items():
    if not os.path.exists(local_path):
        print(f"Downloading {hf_id} -> {local_path}")
        snapshot_download(repo_id=hf_id, local_dir=local_path)

from datasets import load_dataset
for local_path, (hf_id, cfg) in DATASETS.items():
    if not os.path.exists(local_path):
        print(f"Downloading dataset {hf_id} -> {local_path}")
        ds = load_dataset(hf_id, cfg) if cfg else load_dataset(hf_id)
        ds.save_to_disk(local_path)
```

LLaMA models require HuggingFace gated access. Make sure the HF token has
been granted access at https://huggingface.co/meta-llama

---

### TASK 4 — Run the ArmoRM multi-objective experiment (Figure 3)

Execute in order:
```bash
modal run modal_jobs.py::generate_completions_1b_armo
modal run modal_jobs.py::generate_preferences_armo_plot1
modal run modal_jobs.py::generate_preferences_armo_plot2
modal run modal_jobs.py::generate_preferences_armo_plot3
```

Then train all three methods on all three preference datasets.
The paper trains LLaMA-3.2-1B on each of the 3 plots' datasets, for 4 epochs,
with the default beta=0.01.

The `modal_jobs.py` `_train()` helper currently hardcodes `--max_epochs 8`
(from `train_alignment.sh`). For the ArmoRM experiment the paper uses **4 epochs**.
**Fix this**: add a `max_epochs` parameter to `_train()` and the Modal function
signatures, defaulting to 8 for leaderboard and 4 for ArmoRM.

For each of plot1 / plot2 / plot3:
```bash
# DPO
modal run modal_jobs.py::train_dpo_1b_armo --dataset-path datasets/helpsteer2_prefs_armo_plot1
# KLDPO (tau in {0.5, 0.75, 1.0} per paper)
modal run modal_jobs.py::train_kldpo_1b_armo --dataset-path datasets/helpsteer2_prefs_armo_plot1 --tau 0.5
# WDPO (rho in {50, 75, 100} per paper — note different scale than leaderboard)
modal run modal_jobs.py::train_wdpo_1b_armo --dataset-path datasets/helpsteer2_prefs_armo_plot1 --rho 50
```

**Note on hyperparameters:** The ArmoRM experiment uses different tau/rho ranges
than the leaderboard experiment. Inspect the paper's Figure 2 caption:
- KLDPO tau ∈ {0.5, 0.75, 1.0} for Emotion; tau ∈ {0.005, 0.01} for leaderboard
- WDPO rho ∈ {50, 75, 100} for Emotion; rho ∈ {0.005, 0.01} for leaderboard

This difference is real — double check these values before running expensive jobs.

---

### TASK 5 — Run the OpenLLM Leaderboard experiment (Table 1)

```bash
# Generate data (10 completions per prompt)
modal run modal_jobs.py::generate_completions_1b_leaderboard
modal run modal_jobs.py::generate_preferences_leaderboard

# Train 1B — all three methods
modal run modal_jobs.py::train_dpo_1b
modal run modal_jobs.py::train_kldpo_1b --tau 0.05
modal run modal_jobs.py::train_kldpo_1b --tau 0.1
modal run modal_jobs.py::train_wdpo_1b --rho 0.005
modal run modal_jobs.py::train_wdpo_1b --rho 0.01
```

The paper's DPO is evaluated at epoch 2 (early stopping) and epoch 4.
`train_alignment.sh` saves checkpoints every 158 steps (`--save_steps 158`).
After 8 epochs with batch_size=128 on ~10K preference pairs:
- Epoch 2 ≈ global_step_158
- Epoch 4 ≈ global_step_316

Evaluate:
```bash
modal run modal_jobs.py::eval_leaderboard --model-path outputs/llama1b_dpo/global_step_158
modal run modal_jobs.py::eval_leaderboard --model-path outputs/llama1b_kldpo_tau0.05/global_step_158
modal run modal_jobs.py::eval_leaderboard --model-path outputs/llama1b_wdpo_rho0.01/global_step_158
```

Repeat for 3B:
```bash
# Generate 3B completions (can reuse leaderboard dataset from 1B run
# IF the dataset is model-agnostic — check generate_completions.py)
modal run modal_jobs.py::train_dpo_3b
modal run modal_jobs.py::train_kldpo_3b --tau 0.005
modal run modal_jobs.py::eval_leaderboard --model-path outputs/llama3b_kldpo_tau0.005/global_step_158
```

8B (KLDPO only):
```bash
modal run modal_jobs.py::train_kldpo_8b --tau 0.005
modal run modal_jobs.py::eval_leaderboard --model-path outputs/llama8b_kldpo_tau0.005/global_step_158
```

---

### TASK 6 — Collect and plot results

Create `plot_results.py` (new file, not in the paper's repo) that:

1. Reads the leaderboard eval JSON outputs from `/vol/results/`
2. Produces a reproduction of Table 1 as a pandas DataFrame, printed and
   saved to `results/table1_repro.csv`
3. Reads the ArmoRM per-objective scores from the test.jsonl files
   produced by `eval_armo` and plots radar charts matching Figure 3,
   saved to `results/figure3_repro.png`

Use only standard libraries: `pandas`, `matplotlib`, `numpy`, `json`.
Do not use seaborn or plotly.

---

### TASK 7 — Debugging guide (known issues to watch for)

**WDPO gradient penalty on embeddings:**
The paper computes `∇_z l(z; θ)` w.r.t. the embedding layer output, not raw
token IDs. The paper's `src/train_preference.py` should handle this internally.
If you see errors like "element 0 of tensors does not require grad", the
embedding layer's `requires_grad` is not set. Do NOT modify `src/train_preference.py`
unless you have confirmed the bug; instead open a GitHub issue on the paper's
repo and document the workaround here.

**KLDPO all-gather in distributed training:**
The KLDPO worst-case kernel requires averaging loss across ALL workers, not
just the local micro-batch. This is done via a `dist.all_reduce` in the
paper's trainer. If you see KLDPO results matching DPO exactly (no improvement),
this all-gather may be silently failing. Add a debug print to confirm the
mean loss is computed globally, not per-worker.

**SLURM_GPUS_ON_NODE:**
`train_alignment.sh` uses `$SLURM_GPUS_ON_NODE` as the `--zpg` argument to
DeepSpeed. `modal_jobs.py` sets this env var via `os.environ` before calling
the script. Verify it is set correctly for each GPU count (4 for A100x4, 8
for H100x8).

**Volume symlinks:**
The paper's scripts use relative paths (`models/`, `datasets/`, `outputs/`).
`modal_jobs.py` symlinks these into `/vol/`. If a script fails with
"No such file or directory", check that the symlink was created before the
script ran and that the volume was not re-mounted at a different path.

**Flash attention:**
`flash-attn` must be compiled against the exact torch + CUDA version.
If the pip install fails, fetch the pre-built wheel directly:
```
https://github.com/Dao-AILab/flash-attention/releases
```
Find the wheel matching `torch2.5.1`, `cu124`, `cp311`. Add it to the
Modal image as `.pip_install("<wheel_url>")`.

---

### TASK 8 — Modal image caching

Modal builds images once and caches them. After the first successful build,
subsequent `modal run` calls start in seconds. If you need to change the image
(e.g. fix a dependency), bump any `run_commands` or `pip_install` call —
Modal detects the change by content hash and rebuilds only the affected layers.

To avoid expensive rebuilds during debugging, add new pip installs at the END
of the image definition, not in the middle (later layers cache independently).

---

### TASK 9 — Cost tracking

Rough cost estimates (Modal, April 2026 pricing):
| Job | GPU | Time | Est. cost |
|---|---|---|---|
| smoke_test | A10G ×1 | 5 min | ~$0.05 |
| download_assets | A10G ×1 | 60 min | ~$0.50 |
| generate_completions (×2 best_of_n=10) | A10G ×1 | 4 hr | ~$2 |
| train_*_1b | A100 80GB ×4 | 5 hr | ~$20 |
| train_*_3b | A100 80GB ×4 | 10 hr | ~$40 |
| train_kldpo_8b | H100 ×8 | 18 hr | ~$150 |
| eval_leaderboard | A100 80GB ×4 | 8 hr | ~$32 |

**Always run 1B experiments first to catch bugs cheaply before scaling up.**

---

### TASK 10 — Do NOT do these things

- Do NOT modify `distributionally_robust_dpo/src/*.py` unless there is a
  confirmed bug that cannot be worked around at the Modal layer.
- Do NOT rewrite WDPO or KLDPO loss functions. Use the paper's implementations.
- Do NOT change hyperparameters from the paper without documenting why.
- Do NOT commit `.env` or any file containing API tokens.
- Do NOT run 3B or 8B jobs until 1B results look correct.

---

## Quick reference: full run sequence

```bash
# One-time setup
bash setup_local.sh
source .venv/bin/activate
cp .env.example .env        # fill in HF_TOKEN
modal setup                  # authenticate Modal
modal secret create huggingface-secret HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2)

# Verify infra (cheap)
modal run modal_jobs.py::smoke_test

# Download assets (once)
modal run modal_jobs.py::download_assets

# ArmoRM experiment (Figure 3)
modal run modal_jobs.py::generate_completions_1b_armo
modal run modal_jobs.py::generate_preferences_armo_plot1
modal run modal_jobs.py::generate_preferences_armo_plot2
modal run modal_jobs.py::generate_preferences_armo_plot3
# ... train + eval (see TASK 4)

# Leaderboard experiment (Table 1) — start with 1B
modal run modal_jobs.py::generate_completions_1b_leaderboard
modal run modal_jobs.py::generate_preferences_leaderboard
modal run modal_jobs.py::train_dpo_1b
modal run modal_jobs.py::train_kldpo_1b
modal run modal_jobs.py::train_wdpo_1b
modal run modal_jobs.py::eval_leaderboard --model-path outputs/llama1b_dpo/global_step_158
modal run modal_jobs.py::eval_leaderboard --model-path outputs/llama1b_kldpo_tau0.05/global_step_158
modal run modal_jobs.py::eval_leaderboard --model-path outputs/llama1b_wdpo_rho0.01/global_step_158
```
