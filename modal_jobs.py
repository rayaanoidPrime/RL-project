"""
modal_jobs.py
=============
Modal wrappers around the paper's existing scripts.
We do NOT rewrite the paper's code — we call their scripts directly.

Repo layout assumed (cloned into the Modal image):
  distributionally_robust_dpo/
    src/
      setup.py
      generate_completions.py
      generate_preferences.py
      train_preference.py
    generate_completions.sh
    train_alignment.sh
    leaderboard_eval.sh
    setup.sh

Usage examples:
  modal run modal_jobs.py::smoke_test
  modal run modal_jobs.py::generate_completions_1b_armo
  modal run modal_jobs.py::generate_preferences_leaderboard
  modal run modal_jobs.py::train_dpo_1b
  modal run modal_jobs.py::train_kldpo_1b
  modal run modal_jobs.py::train_wdpo_1b
  modal run modal_jobs.py::train_kldpo_3b
  modal run modal_jobs.py::train_kldpo_8b
  modal run modal_jobs.py::eval_leaderboard --model-path outputs/llama1b_kldpo/global_step_158
"""

import os
import subprocess
import modal

# =============================================================================
# Secrets and config
# =============================================================================

HF_SECRET = modal.Secret.from_name("huggingface-secret")  # set via: modal secret create huggingface-secret HF_TOKEN=hf_...

# Persistent volume — datasets, model checkpoints, outputs all live here
# so they survive between runs and you don't re-download every time.
VOLUME = modal.Volume.from_name("drdpo-vol", create_if_missing=True)
VOLUME_MOUNT = "/vol"

# =============================================================================
# Docker images
# =============================================================================

# Base image: CUDA 12.4 + Python 3.11, matches paper's requirements exactly.
# We install the paper's conda env dependencies via pip instead of conda
# because Modal images don't use conda — they use pip on top of a CUDA base.
BASE_IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "git", "wget", "curl", "build-essential", "ninja-build",
        "libssl-dev", "libffi-dev",
    )
    # Torch first (matches paper's torch==2.5.1 + cu124)
    .pip_install(
        "torch==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    # DeepSpeed with fused kernels (matches paper's DS v0.16.4)
    .run_commands(
        "git clone https://github.com/microsoft/DeepSpeed.git --branch v0.16.4 -c advice.detachedHead=false /tmp/DeepSpeed",
        "cd /tmp/DeepSpeed && DS_BUILD_UTILS=1 DS_BUILD_FUSED_ADAM=1 pip install . && cd / && rm -rf /tmp/DeepSpeed",
    )
    # OpenRLHF (paper uses openrlhf[vllm]==0.6.1.post1)
    .pip_install("openrlhf[vllm]==0.6.1.post1")
    # LM Evaluation Harness pinned to paper's commit
    .run_commands(
        "git clone https://github.com/EleutherAI/lm-evaluation-harness /tmp/lm-evaluation-harness",
        "cd /tmp/lm-evaluation-harness && git checkout 19ba1b16fef9fa6354a3e4ef3574bb1d03552922 && pip install -e '.[math]' && cd / && rm -rf /tmp/lm-evaluation-harness",
    )
    # Additional utilities
    .pip_install(
        "huggingface_hub",
        "accelerate",
        "transformers",
        "datasets",
        "pandas",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "tqdm",
        "flash-attn==2.7.2.post1",  # flash attention for training
    )
    # Clone the paper's repo into the image
    .run_commands(
        "git clone https://github.com/TheBlackCat22/distributionally_robust_dpo /repo",
    )
    .env({"PYTHONPATH": "/repo", "VLLM_CONFIGURE_LOGGING": "0"})
)

# =============================================================================
# GPU specs
# =============================================================================

# A10G: cheap, sufficient for 1B model generation and ArmoRM scoring
A10G_SPEC = modal.gpu.A10G(count=1)

# A100 80GB: for 1B/3B training with DeepSpeed ZeRO-2
A100_SPEC = modal.gpu.A100(size="80GB", count=4)

# H100: for 8B model and large-scale runs
H100_SPEC = modal.gpu.H100(count=8)

# =============================================================================
# App
# =============================================================================

app = modal.App("drdpo-repro")


# =============================================================================
# Helper: run a shell command inside the container, streaming output
# =============================================================================

def _run(cmd: str, cwd: str = "/repo") -> None:
    """Run a shell command and stream stdout/stderr."""
    print(f"\n>>> {cmd}\n")
    result = subprocess.run(
        cmd, shell=True, cwd=cwd,
        stdout=None, stderr=None,  # inherit — streams live to Modal logs
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {cmd}")


def _hf_login():
    """Log into HuggingFace using the secret token."""
    token = os.environ.get("HF_TOKEN", "")
    if token:
        _run(f"huggingface-cli login --token {token}")


# =============================================================================
# SMOKE TEST — verify the image is healthy before spending money on big jobs
# =============================================================================

@app.function(
    image=BASE_IMAGE,
    gpu=A10G_SPEC,
    secrets=[HF_SECRET],
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=300,
)
def smoke_test():
    """Quick sanity check: imports, GPU visibility, repo structure."""
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    import deepspeed
    print(f"DeepSpeed version: {deepspeed.__version__}")

    import openrlhf
    print(f"OpenRLHF imported OK")

    _run("ls /repo")
    _run("ls /repo/src")
    print("\nSmoke test PASSED.")


# =============================================================================
# STEP 0: Download models and datasets (paper's src/setup.py)
# =============================================================================

@app.function(
    image=BASE_IMAGE,
    secrets=[HF_SECRET],
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=3600,  # downloads can take a while
    gpu=A10G_SPEC,
)
def download_assets():
    """
    Run the paper's src/setup.py which downloads:
      - LLaMA-3.2-1B-Instruct  -> /vol/models/Llama1b
      - LLaMA-3.2-3B-Instruct  -> /vol/models/Llama3b
      - LLaMA-3.1-8B-Instruct  -> /vol/models/Llama8b
      - ArmoRM reward model     -> /vol/models/ArmoRM
      - HelpSteer2 prompts      -> /vol/datasets/helpsteer2_prompts
    """
    _hf_login()
    # The paper's setup.py saves to relative paths; we symlink /repo subdirs
    # to the persistent volume so nothing is lost between runs.
    _run("mkdir -p /vol/models /vol/datasets")
    _run("ln -sfn /vol/models /repo/models")
    _run("ln -sfn /vol/datasets /repo/datasets")
    _run("python src/setup.py", cwd="/repo")
    VOLUME.commit()
    print("Assets downloaded and committed to volume.")


# =============================================================================
# STEP 1: Generate completions
# =============================================================================

@app.function(
    image=BASE_IMAGE,
    secrets=[HF_SECRET],
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=7200,
    gpu=A10G_SPEC,
)
def generate_completions_1b_armo():
    """
    ArmoRM multi-objective experiment: 2 completions per prompt.
    Calls the paper's generate_completions.sh directly.
    Output -> /vol/datasets/helpsteer2_completions_armo
    """
    _hf_login()
    _run("ln -sfn /vol/models /repo/models")
    _run("ln -sfn /vol/datasets /repo/datasets")
    _run("ln -sfn /vol/outputs /repo/outputs 2>/dev/null || true")
    _run(
        "bash generate_completions.sh "
        "--save_path=datasets/helpsteer2_completions_armo "
        "--model_path=models/Llama1b "
        "--best_of_n=2 "
        "--temperature=0.7",
        cwd="/repo",
    )
    VOLUME.commit()


@app.function(
    image=BASE_IMAGE,
    secrets=[HF_SECRET],
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=14400,
    gpu=A10G_SPEC,
)
def generate_completions_1b_leaderboard():
    """
    Leaderboard experiment: 10 completions per prompt.
    Calls the paper's generate_completions.sh directly.
    Output -> /vol/datasets/helpsteer2_completions_leaderboard
    """
    _hf_login()
    _run("ln -sfn /vol/models /repo/models")
    _run("ln -sfn /vol/datasets /repo/datasets")
    _run(
        "bash generate_completions.sh "
        "--save_path=datasets/helpsteer2_completions_leaderboard "
        "--model_path=models/Llama1b "
        "--best_of_n=10 "
        "--temperature=0.7",
        cwd="/repo",
    )
    VOLUME.commit()


# =============================================================================
# STEP 2: Generate preference datasets
# =============================================================================

@app.function(
    image=BASE_IMAGE,
    secrets=[HF_SECRET],
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=3600,
    gpu=A10G_SPEC,
)
def generate_preferences_armo_plot1():
    """Plot 1: ultrafeedback_truthfulness + helpsteer_complexity (50/50)."""
    _run("ln -sfn /vol/models /repo/models")
    _run("ln -sfn /vol/datasets /repo/datasets")
    _run(
        "python src/generate_preferences.py "
        "--completions=datasets/helpsteer2_completions_armo "
        "--output_path=datasets/helpsteer2_prefs_armo_plot1 "
        "--ultrafeedback_truthfulness=0.5 --helpsteer_complexity=0.5",
        cwd="/repo",
    )
    VOLUME.commit()


@app.function(
    image=BASE_IMAGE,
    secrets=[HF_SECRET],
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=3600,
    gpu=A10G_SPEC,
)
def generate_preferences_armo_plot2():
    """Plot 2: ultrafeedback_helpfulness + helpsteer_coherence (50/50)."""
    _run("ln -sfn /vol/models /repo/models")
    _run("ln -sfn /vol/datasets /repo/datasets")
    _run(
        "python src/generate_preferences.py "
        "--completions=datasets/helpsteer2_completions_armo "
        "--output_path=datasets/helpsteer2_prefs_armo_plot2 "
        "--ultrafeedback_helpfulness=0.5 --helpsteer_coherence=0.5",
        cwd="/repo",
    )
    VOLUME.commit()


@app.function(
    image=BASE_IMAGE,
    secrets=[HF_SECRET],
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=3600,
    gpu=A10G_SPEC,
)
def generate_preferences_armo_plot3():
    """Plot 3: helpsteer_correctness + helpsteer_helpfulness (50/50)."""
    _run("ln -sfn /vol/models /repo/models")
    _run("ln -sfn /vol/datasets /repo/datasets")
    _run(
        "python src/generate_preferences.py "
        "--completions=datasets/helpsteer2_completions_armo "
        "--output_path=datasets/helpsteer2_prefs_armo_plot3 "
        "--helpsteer_correctness=0.5 --helpsteer_helpfulness=0.5",
        cwd="/repo",
    )
    VOLUME.commit()


@app.function(
    image=BASE_IMAGE,
    secrets=[HF_SECRET],
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=3600,
    gpu=A10G_SPEC,
)
def generate_preferences_leaderboard():
    """Leaderboard experiment: ArmoRM stage-2 scalar reward."""
    _run("ln -sfn /vol/models /repo/models")
    _run("ln -sfn /vol/datasets /repo/datasets")
    _run(
        "python src/generate_preferences.py "
        "--completions=datasets/helpsteer2_completions_leaderboard "
        "--output_path=datasets/helpsteer2_prefs_leaderboard "
        "--ArmoRM=1.0",
        cwd="/repo",
    )
    VOLUME.commit()


# =============================================================================
# STEP 3: Train — 1B model (DPO / KLDPO / WDPO)
# =============================================================================
# train_alignment.sh uses $SLURM_GPUS_ON_NODE for --zpg.
# On Modal we set that env var manually to match GPU count.

def _train(
    model_path: str,
    dataset_path: str,
    save_path: str,
    method: str,           # "dpo" | "kldpo" | "wdpo"
    extra_flag: str = "",  # e.g. "--kldpo_tau=0.05"
    gpu_count: int = 4,
):
    """Internal helper that calls train_alignment.sh."""
    os.environ["SLURM_GPUS_ON_NODE"] = str(gpu_count)
    _run("mkdir -p /vol/outputs")
    _run("ln -sfn /vol/models /repo/models")
    _run("ln -sfn /vol/datasets /repo/datasets")
    _run("ln -sfn /vol/outputs /repo/outputs")
    flag = extra_flag
    _run(
        f"bash train_alignment.sh "
        f"--model_path={model_path} "
        f"--dataset_path={dataset_path} "
        f"--save_path={save_path} "
        f"{flag}",
        cwd="/repo",
    )
    VOLUME.commit()


@app.function(
    image=BASE_IMAGE,
    secrets=[HF_SECRET],
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=18000,
    gpu=A100_SPEC,
)
def train_dpo_1b():
    _train(
        model_path="models/Llama1b",
        dataset_path="datasets/helpsteer2_prefs_leaderboard",
        save_path="outputs/llama1b_dpo",
        method="dpo",
        gpu_count=4,
    )


@app.function(
    image=BASE_IMAGE,
    secrets=[HF_SECRET],
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=18000,
    gpu=A100_SPEC,
)
def train_kldpo_1b(tau: float = 0.05):
    _train(
        model_path="models/Llama1b",
        dataset_path="datasets/helpsteer2_prefs_leaderboard",
        save_path=f"outputs/llama1b_kldpo_tau{tau}",
        method="kldpo",
        extra_flag=f"--kldpo_tau={tau}",
        gpu_count=4,
    )


@app.function(
    image=BASE_IMAGE,
    secrets=[HF_SECRET],
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=18000,
    gpu=A100_SPEC,
)
def train_wdpo_1b(rho: float = 0.01):
    _train(
        model_path="models/Llama1b",
        dataset_path="datasets/helpsteer2_prefs_leaderboard",
        save_path=f"outputs/llama1b_wdpo_rho{rho}",
        method="wdpo",
        extra_flag=f"--wdpo_rho={rho}",
        gpu_count=4,
    )


# ---------- 3B ----------

@app.function(
    image=BASE_IMAGE,
    secrets=[HF_SECRET],
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=36000,
    gpu=A100_SPEC,
)
def train_dpo_3b():
    _train(
        model_path="models/Llama3b",
        dataset_path="datasets/helpsteer2_prefs_leaderboard",
        save_path="outputs/llama3b_dpo",
        method="dpo",
        gpu_count=4,
    )


@app.function(
    image=BASE_IMAGE,
    secrets=[HF_SECRET],
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=36000,
    gpu=A100_SPEC,
)
def train_kldpo_3b(tau: float = 0.005):
    _train(
        model_path="models/Llama3b",
        dataset_path="datasets/helpsteer2_prefs_leaderboard",
        save_path=f"outputs/llama3b_kldpo_tau{tau}",
        method="kldpo",
        extra_flag=f"--kldpo_tau={tau}",
        gpu_count=4,
    )


@app.function(
    image=BASE_IMAGE,
    secrets=[HF_SECRET],
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=36000,
    gpu=A100_SPEC,
)
def train_wdpo_3b(rho: float = 0.005):
    _train(
        model_path="models/Llama3b",
        dataset_path="datasets/helpsteer2_prefs_leaderboard",
        save_path=f"outputs/llama3b_wdpo_rho{rho}",
        method="wdpo",
        extra_flag=f"--wdpo_rho={rho}",
        gpu_count=4,
    )


# ---------- 8B (KLDPO only — matches paper) ----------

@app.function(
    image=BASE_IMAGE,
    secrets=[HF_SECRET],
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=72000,
    gpu=H100_SPEC,
)
def train_kldpo_8b(tau: float = 0.005):
    _train(
        model_path="models/Llama8b",
        dataset_path="datasets/helpsteer2_prefs_leaderboard",
        save_path=f"outputs/llama8b_kldpo_tau{tau}",
        method="kldpo",
        extra_flag=f"--kldpo_tau={tau}",
        gpu_count=8,
    )


# =============================================================================
# STEP 4a: ArmoRM evaluation (generate completions from trained model)
# =============================================================================

@app.function(
    image=BASE_IMAGE,
    secrets=[HF_SECRET],
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=7200,
    gpu=A10G_SPEC,
)
def eval_armo(model_path: str = "outputs/llama1b_kldpo_tau0.05/global_step_316"):
    """Generate completions for ArmoRM eval (only_test=true)."""
    _run("ln -sfn /vol/models /repo/models")
    _run("ln -sfn /vol/datasets /repo/datasets")
    _run("ln -sfn /vol/outputs /repo/outputs")
    _run(
        f"bash generate_completions.sh "
        f"--only_test=true "
        f"--save_path={model_path} "
        f"--model_path={model_path} "
        f"--best_of_n=10 "
        f"--temperature=0.1",
        cwd="/repo",
    )
    VOLUME.commit()


# =============================================================================
# STEP 4b: OpenLLM Leaderboard evaluation
# =============================================================================

@app.function(
    image=BASE_IMAGE,
    secrets=[HF_SECRET],
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=36000,
    gpu=A100_SPEC,   # leaderboard eval script uses 4 GPUs
)
def eval_leaderboard(model_path: str = "outputs/llama1b_kldpo_tau0.05/global_step_158"):
    """Run OpenLLM leaderboard v2 eval via the paper's leaderboard_eval.sh."""
    _run("ln -sfn /vol/models /repo/models")
    _run("ln -sfn /vol/datasets /repo/datasets")
    _run("ln -sfn /vol/outputs /repo/outputs")
    _run("mkdir -p /vol/results")
    _run("ln -sfn /vol/results /repo/results")
    _run(
        f"bash leaderboard_eval.sh --model_path={model_path}",
        cwd="/repo",
    )
    VOLUME.commit()
    print(f"Results saved to /vol/results/")


# =============================================================================
# UTILITY: List volume contents
# =============================================================================

@app.function(
    image=BASE_IMAGE,
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=60,
)
def ls_volume(path: str = "/vol"):
    """List what's in the persistent volume."""
    _run(f"find {path} -maxdepth 3 -type f | head -100")


# =============================================================================
# LOCAL entrypoint for quick CLI use
# =============================================================================

@app.local_entrypoint()
def main():
    print("DR-DPO Modal jobs loaded. Run individual functions with:")
    print("  modal run modal_jobs.py::smoke_test")
    print("  modal run modal_jobs.py::download_assets")
    print("  modal run modal_jobs.py::generate_completions_1b_armo")
    print("  modal run modal_jobs.py::train_kldpo_1b")
    print("  modal run modal_jobs.py::eval_leaderboard --model-path outputs/llama1b_kldpo_tau0.05/global_step_158")
