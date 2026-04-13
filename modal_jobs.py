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
# BASE_IMAGE = (
#     modal.Image.from_registry(
#         "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
#         add_python="3.11",
#     )

#     # ---------------- SYSTEM ----------------
#     .apt_install(
#         "git", "wget", "curl",
#         "build-essential",
#         "ninja-build",
#         "pkg-config",
#         "libssl-dev",
#         "libffi-dev",
#     )

#     # ---------------- TORCH (matches paper) ----------------
#     .pip_install(
#         "torch==2.5.1+cu124",
#         "torchvision",
#         "torchaudio",
#         extra_index_url="https://download.pytorch.org/whl/cu124",
#     )

#     # ---------------- CORE STACK ----------------
#     .pip_install(
#         "transformers==4.48.3",
#         "accelerate",
#         "datasets",
#         "huggingface_hub",
#         "tqdm",
#         "pandas",
#         "numpy",
#     )

#     # ---------------- DEEPSPEED (NO SOURCE BUILD) ----------------
#     # paper builds from source, but this is the closest stable equivalent
#     .pip_install("deepspeed==0.16.4")

#     # ---------------- CRITICAL FIX ----------------
#     # install openrlhf WITHOUT triggering flash-attn build
#     .run_commands(
#         "pip install openrlhf==0.6.1.post1 --no-deps"
#     )

#     # ---------------- SAFE DEPENDENCIES ----------------
#     .pip_install(
#         "peft",
#         "optimum",
#         "wandb",
#         "tensorboard",
#         "torchmetrics",
#         "bitsandbytes",
#     )

#     # ---------------- OPTIONAL (vllm, safer than openrlhf[vllm]) ----------------
#     .pip_install("vllm==0.6.3")

#     # ---------------- LM EVAL ----------------
#     .run_commands(
#         "git clone https://github.com/EleutherAI/lm-evaluation-harness /tmp/lm-eval",
#         "cd /tmp/lm-eval && git checkout 19ba1b16fef9fa6354a3e4ef3574bb1d03552922 && pip install -e .",
#         "rm -rf /tmp/lm-eval",
#     )

#     # ---------------- REPO ----------------
#     .run_commands(
#         "git clone https://github.com/TheBlackCat22/distributionally_robust_dpo /repo",
#     )

#     .env({
#         "PYTHONPATH": "/repo",
#         "VLLM_CONFIGURE_LOGGING": "0",
#     })
# )
BASE_IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
 
    # ── System packages ────────────────────────────────────────
    # gcc-12 (not gcc-11 from build-essential) for better CUDA 12.4
    # compat. gcc-13 needs a PPA which Modal's apt_install can't add,
    # so gcc-12 is the highest available in Ubuntu 22.04's default repos.
    # libstdc++-12-dev: C++17 STL headers DeepSpeed fused kernels need.
    # libnccl-dev: available from NVIDIA's repo pre-configured in base image.
    .apt_install(
        "git", "wget", "curl",
        "build-essential",
        "gcc-12", "g++-12",
        "libstdc++-12-dev",
        "ninja-build",
        "pkg-config",
        "libssl-dev",
        "libffi-dev",
    )
 
    # ── Set gcc-12 as the default compiler ────────────────────
    .run_commands(
        "update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100",
        "update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100",
        "update-alternatives --install /usr/bin/cc  cc  /usr/bin/gcc-12 100",
    )
 
    # ── Environment (set BEFORE DeepSpeed build) ──────────────
    # TORCH_CUDA_ARCH_LIST: without this, nvcc can't detect GPU arch
    #   during `docker build` (no GPU attached) and the kernel
    #   compilation silently fails or errors out. Covers V100→H100.
    # NVCC_PREPEND_FLAGS: explicitly points nvcc at gcc-12 to avoid
    #   host-compiler detection failures.
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        # Force gcc-12 as host compiler for ALL C/C++ extension builds.
        # Modal's add_python="3.11" ships a Python binary compiled with clang.
        # torch.utils.cpp_extension detects the Python compiler and tries to
        # use clang -- but clang is not installed, crashing with
        # "command 'clang' failed: No such file or directory".
        # CC/CXX override that detection and pin gcc-12 explicitly.
        "CC": "/usr/bin/gcc-12",
        "CXX": "/usr/bin/g++-12",
        "TORCH_CUDA_ARCH_LIST": "7.0 7.5 8.0 8.6 9.0+PTX",
        "NVCC_PREPEND_FLAGS": "-ccbin /usr/bin/gcc-12",
        "PYTHONPATH": "/repo",
        "VLLM_CONFIGURE_LOGGING": "0",
    })
 
    # ── Build tooling (must come before DeepSpeed) ────────────
    # packaging: DeepSpeed's setup.py imports it at the top level.
    # wheel: required to produce installable artifacts during build.
    .pip_install(
        "packaging>=23.0",
        "wheel>=0.43",
        "ninja>=1.11",
    )
 
    # ── PyTorch 2.5.1 + CUDA 12.4 ────────────────────────────
    .pip_install(
        "torch==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
 
    # ── DeepSpeed from source with CUDA extensions ────────────
    # MAX_JOBS=4: caps parallel compiler procs so the build doesn't
    #   OOM-kill itself (Modal build containers have limited RAM).
    #   Raise to 8 if builds fail with exit code 137 (OOM).
    # DS_BUILD_FUSED_ADAM + DS_BUILD_UTILS match the original script.
    .run_commands(
        "git clone https://github.com/microsoft/DeepSpeed.git"
        " --branch v0.16.4 --depth 1 -c advice.detachedHead=false",
        # --no-build-isolation: pip normally builds wheels in a fresh
        # isolated venv that doesn't inherit installed packages. DeepSpeed's
        # setup.py does `import torch` at the top to detect GPU arch and
        # select which ops to compile — without this flag, torch is invisible
        # in the isolated env and the build crashes immediately.
        "cd DeepSpeed && MAX_JOBS=4 DS_BUILD_UTILS=1 DS_BUILD_FUSED_ADAM=1"
        " pip install --no-cache-dir --no-build-isolation . && cd ..",
        "rm -rf DeepSpeed",
    )
 

    # ── OpenRLHF (no flash-attn build, matches your original) ─
    .run_commands("pip install --no-build-isolation openrlhf[vllm]==0.6.1.post1")
   
 
    # ── Re-pin torch after vLLM (may have downgraded it) ──────
    .pip_install(
        "torch==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
 
    # ── LM Evaluation Harness (pinned commit, math extras) ────
    # BUG FIX: your original used `pip install -e .` then `rm -rf`.
    # Editable installs symlink to the source dir — deleting the dir
    # breaks all imports at runtime. Regular install is safe to delete.
    .run_commands(
        "git clone https://github.com/EleutherAI/lm-evaluation-harness /tmp/lm-eval",
        "cd /tmp/lm-eval && git checkout 19ba1b16fef9fa6354a3e4ef3574bb1d03552922"
        " && pip install --no-cache-dir '.[math]'",
        "rm -rf /tmp/lm-eval",
    )
 
    # ── Clone project repo ────────────────────────────────────
    .run_commands(
        "git clone https://github.com/TheBlackCat22/distributionally_robust_dpo /repo",
    )
    .env({
        "PYTHONPATH": "/repo",
        "VLLM_CONFIGURE_LOGGING": "0",
    })
)


# =============================================================================
# GPU specs — Modal 1.x uses plain strings, not modal.gpu objects.
# Syntax: "TYPE" for 1 GPU, "TYPE:N" for N GPUs.
# A100 80GB variant is specifically "A100-80GB".
# Strings are case-insensitive.
# =============================================================================

A10G_SPEC    = "A10G"           # 1x A10G: cheap, for 1B generation & ArmoRM scoring
A100_SPEC    = "A100-80GB:4"    # 4x A100 80GB: for 1B/3B training with DeepSpeed ZeRO-2
H100_SPEC    = "H100:8"         # 8x H100: for 8B model and large-scale runs

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
    timeout=10000,
    gpu=A10G_SPEC,
)
def generate_completions_1b_armo():
    """
    ArmoRM multi-objective experiment: 2 completions per prompt.
    Output -> /vol/datasets/helpsteer2_completions
    """
    _hf_login()
    _run("ln -sfn /vol/models /repo/models")
    _run("ln -sfn /vol/datasets /repo/datasets")
    _run("ln -sfn /vol/outputs /repo/outputs 2>/dev/null || true")

    # ── Patch 1: Lower RM micro_batch_size 64 → 4 ─────────────────
    # ArmoRM is 8B (~16GB weights). On a 22GB A10G there's ~6GB left
    # for activations. batch=64 needs ~8GB of activations → OOM.
    # batch=4 uses ~0.5GB → fits with headroom.
    _run("""sed -i \
        's/--micro_batch_size 64/--micro_batch_size 16/g' \
        /repo/generate_completions.sh""")
    
    # ---- Patch 2 --------------

    script_path = "/repo/src/generate_completions.py"
    with open(script_path, "r") as f:
        src = f.read()

    # These two lines together are the problem:
    # Line 1: batch apply_chat_template fails on Arrow column data
    # Line 2: TokensPrompt wrapping + best_of_n repetition (this part is correct)
    # We collapse both into one comprehension that does all three things correctly.
    old = (
        "prompts = tokenizer.apply_chat_template(prompts_data[args.input_key], add_generation_prompt=True)\n"
        "    prompts = [TokensPrompt(prompt_token_ids=prompt) for prompt in prompts for _ in range(args.best_of_n)]"
    )
    new = (
        "prompts = [\n"
        "        TokensPrompt(prompt_token_ids=tokenizer.apply_chat_template(\n"
        "            conv, add_generation_prompt=True, tokenize=True\n"
        "        ))\n"
        "        for conv in prompts_data[args.input_key]\n"
        "        for _ in range(args.best_of_n)\n"
        "    ]"
    )

    assert old in src, "Pattern not found — check exact whitespace in the source file"
    patched = src.replace(old, new, 1)

    with open(script_path, "w") as f:
        f.write(patched)

    print("Patch applied successfully.")

    # Bug fix: note the space added before --only_test so flags don't merge
    _run(
        "bash generate_completions.sh "
        "--save_path=datasets/helpsteer2_completions "
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
    timeout=20000,
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
    # Fix syntax error in paper's code: assert assignment vs equality
    _run(
        "sed -i 's/assert weight_sum=1,/assert weight_sum == 1,/' "
        "/repo/src/generate_preferences.py"
    )
    _run(
        "python src/generate_preferences.py "
        "--completions=datasets/helpsteer2_completions "
        "--output_path=datasets/helpsteer2_prefs_armo_plot1 "
        "--ultrafeedback-truthfulness=0.5 --helpsteer-complexity=0.5",
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
    # Fix syntax error in paper's code: assert assignment vs equality
    _run(
        "sed -i 's/assert weight_sum=1,/assert weight_sum == 1,/' "
        "/repo/src/generate_preferences.py"
    )
    _run(
        "python src/generate_preferences.py "
        "--completions=datasets/helpsteer2_completions "
        "--output_path=datasets/helpsteer2_prefs_armo_plot2 "
        "--ultrafeedback-helpfulness=0.5 --helpsteer-coherence=0.5",
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
    # Fix syntax error in paper's code: assert assignment vs equality
    _run(
        "sed -i 's/assert weight_sum=1,/assert weight_sum == 1,/' "
        "/repo/src/generate_preferences.py"
    )
    _run(
        "python src/generate_preferences.py "
        "--completions=datasets/helpsteer2_completions "
        "--output_path=datasets/helpsteer2_prefs_armo_plot3 "
        "--helpsteer-correctness=0.5 --helpsteer-helpfulness=0.5",
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
    method: str,
    extra_flag: str = "",
    gpu_count: int = 4,
    max_epochs: int = 8,
):
    os.environ["SLURM_GPUS_ON_NODE"] = str(gpu_count)
    _run("mkdir -p /vol/outputs")
    _run("ln -sfn /vol/models /repo/models")
    _run("ln -sfn /vol/datasets /repo/datasets")
    _run("ln -sfn /vol/outputs /repo/outputs")

    micro_train_batch_size = "1" if method == "wdpo" else "4"

    cmd = (
        f"deepspeed src/train_alignment.py "
        f"--save_path {save_path} "
        f"--save_steps 158 "
        f"--save_hf_ckpt "
        f"--disable_ds_ckpt "
        f"--eval_steps 79 "
        f"--ckpt_path {save_path} "
        f"--micro_train_batch_size {micro_train_batch_size} "
        f"--train_batch_size 128 "
        f"--gradient_checkpointing "
        f"--zero_stage 2 "
        f"--bf16 "
        f"--learning_rate 5.0e-7 "
        f"--lr_warmup_ratio 0.1 "
        f"--zpg {gpu_count} "
        f"--flash_attn "
        f"--train_task {method} "
        f"--max_epochs {max_epochs} "
        f"--beta 0.01 "
        f"--pretrain {model_path} "
        f"--dataset json@{dataset_path} "
        f"--prompt_key prompt "
        f"--apply_chat_template "
        f"--max_len 2048 "
        f"--use_tensorboard {save_path} "
        f"{extra_flag}"
    )
    _run(cmd, cwd="/repo")
    VOLUME.commit()


@app.function(
    image=BASE_IMAGE,
    secrets=[HF_SECRET],
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=18000,
    gpu=A100_SPEC,
)
def train_dpo_1b(dataset_path: str = "datasets/helpsteer2_prefs_leaderboard", max_epochs: int = 8):
    _train(
        model_path="models/Llama1b",
        dataset_path=dataset_path,
        save_path=f"outputs/llama1b_dpo_{dataset_path.split('/')[-1]}",
        method="dpo",
        gpu_count=4,
        max_epochs=max_epochs,
    )


@app.function(
    image=BASE_IMAGE,
    secrets=[HF_SECRET],
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=18000,
    gpu=A100_SPEC,
)
def train_kldpo_1b(dataset_path: str = "datasets/helpsteer2_prefs_leaderboard",tau: float = 0.05, max_epochs: int = 8):
    _train(
        model_path="models/Llama1b",
        dataset_path=dataset_path,
        save_path=f"outputs/llama1b_kldpo_tau{tau}_{dataset_path.split('/')[-1]}",
        method="kldpo",
        extra_flag=f"--kldpo_tau={tau}",
        max_epochs=max_epochs,
        gpu_count=4,
    )


@app.function(
    image=BASE_IMAGE,
    secrets=[HF_SECRET],
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=18000,
    gpu=A100_SPEC,
)
def train_wdpo_1b(dataset_path: str = "datasets/helpsteer2_prefs_leaderboard", rho: float = 0.01, max_epochs: int = 8):
    _train(
        model_path="models/Llama1b",
        dataset_path=dataset_path,
        save_path=f"outputs/llama1b_wdpo_rho{rho}_{dataset_path.split('/')[-1]}",
        method="wdpo",
        extra_flag=f"--wdpo_rho={rho}",
        max_epochs=max_epochs,
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
def train_dpo_3b(dataset_path: str = "datasets/helpsteer2_prefs_leaderboard", max_epochs: int = 8):
    _train(
        model_path="models/Llama3b",
        dataset_path=dataset_path,
        save_path=f"outputs/llama3b_dpo_{dataset_path.split('/')[-1]}",
        method="dpo",
        gpu_count=4,
        max_epochs=max_epochs,
    )

@app.function(
    image=BASE_IMAGE,
    secrets=[HF_SECRET],
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=36000,
    gpu=A100_SPEC,
)
def train_kldpo_3b(dataset_path: str = "datasets/helpsteer2_prefs_leaderboard", tau: float = 0.005, max_epochs: int = 8):
    _train(
        model_path="models/Llama3b",
        dataset_path=dataset_path,
        save_path=f"outputs/llama3b_kldpo_tau{tau}_{dataset_path.split('/')[-1]}",
        method="kldpo",
        extra_flag=f"--kldpo_tau={tau}",
        gpu_count=4,
        max_epochs=max_epochs,
    )



@app.function(
    image=BASE_IMAGE,
    secrets=[HF_SECRET],
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=36000,
    gpu=A100_SPEC,
)
def train_wdpo_3b(dataset_path: str = "datasets/helpsteer2_prefs_leaderboard", rho: float = 0.005, max_epochs: int = 8):
    _train(
        model_path="models/Llama3b",
        dataset_path=dataset_path,
        save_path=f"outputs/llama3b_wdpo_rho{rho}_{dataset_path.split('/')[-1]}",
        method="wdpo",
        extra_flag=f"--wdpo_rho={rho}",
        gpu_count=4,
        max_epochs=max_epochs,
    )



# ---------- 8B (KLDPO only — matches paper) ----------

@app.function(
    image=BASE_IMAGE,
    secrets=[HF_SECRET],
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=72000,
    gpu=H100_SPEC,
)
def train_kldpo_8b(dataset_path: str = "datasets/helpsteer2_prefs_leaderboard", tau: float = 0.005, max_epochs: int = 8):
    _train(
        model_path="models/Llama8b",
        dataset_path=dataset_path,
        save_path=f"outputs/llama8b_kldpo_tau{tau}_{dataset_path.split('/')[-1]}",
        method="kldpo",
        extra_flag=f"--kldpo_tau={tau}",
        gpu_count=8,
        max_epochs=max_epochs,
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
def eval_armo(model_path: str = "outputs/llama1b_kldpo_tau0.05_helpsteer2_prefs_armo_plot1/global_step_316"):
    _run("ln -sfn /vol/datasets /repo/datasets")
    _run("ln -sfn /vol/outputs /repo/outputs")
    _run("ln -sfn /vol/models /repo/models")
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
def eval_leaderboard(model_path: str = "outputs/llama1b_kldpo_tau0.05_helpsteer2_prefs_leaderboard/global_step_158"):
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