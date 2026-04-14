"""
emotion_experiment.py
=====================
Reproduces Figure 2 from "Robust LLM Alignment via Distributionally Robust DPO".

One single Modal function `run_emotion_experiment` runs the full pipeline:
  1. Build multi-label Emotion Reward Dataset (drop surprise, merge 3 texts)
  2. Train GPT-2 reward model (classification head, BCE loss, 8 epochs)
  3. SFT GPT-2 on emotion texts (next-token prediction, 10 epochs)
  4. Generate 2 completions per prompt using SFT model (first 4 tokens as prompt)
  5. Label preferences using BT model with convex and geometric mixing (alpha_0=0.1)
  6. Train DPO / WDPO / KLDPO on both mixing types (40 epochs, single GPU)
  7. Evaluate all models at alpha in {0,0.1,0.3,0.5,0.7,0.9,1.0}
  8. Save Figure 2 replica as a PNG to /vol/results/emotion_figure2.png

Usage:
  modal run emotion_experiment.py::run_emotion_experiment
  modal run emotion_experiment.py::run_emotion_experiment --skip-training  # just plot from saved models
"""

import modal
import os

# =============================================================================
# Image — CPU-only is fine for GPT-2 scale, but we use A10G for speed.
# All packages are standard; no DeepSpeed needed (single GPU, plain Adam).
# =============================================================================

EMOTION_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers==4.48.3",
        "datasets",
        "accelerate",
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "tqdm",
    )
)

VOLUME = modal.Volume.from_name("drdpo-vol", create_if_missing=True)
VOLUME_MOUNT = "/vol"

app = modal.App("drdpo-emotion")

# smoke test 

@app.function(
    image=EMOTION_IMAGE,
    gpu="A10G",
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=600,  # 10 min max
)
def smoke_test():
    import torch
    import numpy as np
    from datasets import load_dataset
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SMOKE TEST] Device: {DEVICE}")

    # ── Step 1: Load tiny dataset ─────────────────────────────
    raw = load_dataset("dair-ai/emotion", split="train[:50]")
    texts = raw["text"]
    print(f"[SMOKE TEST] Loaded {len(texts)} samples")

    # ── Step 2: Tokenizer + model forward ─────────────────────
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)

    enc = tokenizer(texts[:4], return_tensors="pt", padding=True, truncation=True)
    out = model(
        input_ids=enc["input_ids"].to(DEVICE),
        attention_mask=enc["attention_mask"].to(DEVICE),
        labels=enc["input_ids"].to(DEVICE),
    )

    print(f"[SMOKE TEST] Forward pass loss: {out.loss.item():.4f}")

    # ── Step 3: Generation ────────────────────────────────────
    prompt = texts[0]
    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    gen = model.generate(
        enc["input_ids"],
        max_new_tokens=10,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    print("[SMOKE TEST] Generated:", tokenizer.decode(gen[0]))

    # ── Step 4: Fake reward scoring ───────────────────────────
    dummy_scores = np.random.rand(4, 5)
    alpha = 0.1
    reward = alpha * dummy_scores[:, 3] + (1 - alpha) * dummy_scores[:, 4]

    print(f"[SMOKE TEST] Reward sample: {reward[:2]}")

    # ── Step 5: DPO loss sanity ───────────────────────────────
    def dummy_log_probs():
        return torch.randn(4, device=DEVICE)

    pol_chosen = dummy_log_probs()
    pol_rejected = dummy_log_probs()
    ref_chosen = dummy_log_probs()
    ref_rejected = dummy_log_probs()

    beta = 0.1
    log_ratio = (pol_chosen - ref_chosen) - (pol_rejected - ref_rejected)
    loss = -torch.nn.functional.logsigmoid(beta * log_ratio).mean()

    print(f"[SMOKE TEST] DPO loss: {loss.item():.4f}")

    print("\n[SMOKE TEST] ✅ PASSED")

# =============================================================================
# The full experiment in one function
# =============================================================================

@app.function(
    image=EMOTION_IMAGE,
    gpu="A10G",
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=36000,  # ~10 hours worst case; typically 4-6h
)
def run_emotion_experiment(skip_training: bool = False):
    """
    Full Figure 2 reproduction. All methodology from Appendix F of the paper.
    Results saved to /vol/results/emotion_figure2.png
    """
    import json
    import math
    import random
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
        get_linear_schedule_with_warmup,
    )
    from datasets import load_dataset
    from tqdm import tqdm
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Paths ──────────────────────────────────────────────────────────────
    BASE = "/vol/emotion"
    os.makedirs(f"{BASE}/reward_model", exist_ok=True)
    os.makedirs(f"{BASE}/reward_model/checkpoints", exist_ok=True)
    os.makedirs(f"{BASE}/sft_model", exist_ok=True)
    os.makedirs(f"{BASE}/sft_model/checkpoints", exist_ok=True)
    os.makedirs(f"{BASE}/completions", exist_ok=True)
    os.makedirs(f"{BASE}/preferences", exist_ok=True)
    os.makedirs(f"{BASE}/trained_models", exist_ok=True)
    os.makedirs("/vol/results", exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # ── Emotion labels (drop surprise) ─────────────────────────────────────
    EMOTIONS = ["sadness", "joy", "love", "anger", "fear"]
    EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}
    # Original dataset label mapping (dair-ai/emotion)
    # 0=sadness,1=joy,2=love,3=anger,4=fear,5=surprise
    ORIG_TO_KEEP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}  # drop 5=surprise

    # =========================================================================
    # STEP 1: Build multi-label Emotion Reward Dataset
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 1: Building multi-label Emotion Reward Dataset")
    print("="*60)

    raw = load_dataset("dair-ai/emotion", split="train+validation+test")

    # Filter out surprise (label=5)
    raw = raw.filter(lambda x: x["label"] != 5)

    texts = raw["text"]
    labels_single = raw["label"]  # single int label per text

    # Build multi-label dataset: concatenate up to 3 random texts,
    # merge their labels into a 5-dim binary vector.
    # Paper: "up to three random text samples were concatenated,
    #         and their associated labels were merged"
    def build_multilabel_dataset(texts, labels, n_samples=10000, seed=42):
        rng = random.Random(seed)
        samples = []
        n = len(texts)
        for _ in range(n_samples):
            k = rng.randint(1, 3)
            idxs = rng.sample(range(n), k)
            merged_text = " ".join(texts[i] for i in idxs)
            label_vec = [0.0] * 5
            for i in idxs:
                orig_label = labels[i]
                if orig_label in ORIG_TO_KEEP:
                    label_vec[ORIG_TO_KEEP[orig_label]] = 1.0
            samples.append({"text": merged_text, "labels": label_vec})
        return samples

    print("Building multi-label samples...")
    all_samples = build_multilabel_dataset(texts, labels_single, n_samples=15000)
    split = int(0.9 * len(all_samples))
    train_samples = all_samples[:split]
    test_samples = all_samples[split:]
    print(f"  Train: {len(train_samples)}, Test: {len(test_samples)}")

    # Also keep original single-label texts for SFT and completions generation
    orig_texts = list(texts)
    # ── Step 1 assertions ────────────────────────────────────────────────
    assert len(orig_texts) > 0, \
        f"[STEP 1] FAILED: No texts loaded from dataset (got {len(orig_texts)})"
    assert len(all_samples) >= 10000, \
        f"[STEP 1] FAILED: Multi-label dataset too small (got {len(all_samples)}, expected ≥10000)"
    assert len(train_samples) > 0 and len(test_samples) > 0, \
        "[STEP 1] FAILED: Train or test split is empty"
    assert all("text" in s and "labels" in s for s in all_samples[:10]), \
        "[STEP 1] FAILED: Samples missing 'text' or 'labels' keys"
    assert all(len(s["labels"]) == 5 for s in all_samples[:10]), \
        "[STEP 1] FAILED: Label vectors are not 5-dimensional"
    print(f"  [STEP 1] ✅ Assertions passed — {len(all_samples)} samples, "
          f"{len(orig_texts)} original texts")

    

    # =========================================================================
    # STEP 2: Train GPT-2 reward model
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 2: Training GPT-2 reward model")
    print("="*60)

    REWARD_MODEL_PATH = f"{BASE}/reward_model"
    reward_model_trained = os.path.exists(f"{REWARD_MODEL_PATH}/pytorch_model.bin")

    tokenizer_rm = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer_rm.pad_token = tokenizer_rm.eos_token

    class EmotionRewardModel(nn.Module):
        """GPT-2 + linear classification head on last token. 5 outputs."""
        def __init__(self):
            super().__init__()
            self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
            # Replace lm_head with a 5-class head
            hidden = self.gpt2.config.n_embd
            self.gpt2.lm_head = nn.Identity()
            # We'll grab the last hidden state manually
            self.classifier = nn.Linear(hidden, 5)

        def forward(self, input_ids, attention_mask=None):
            out = self.gpt2.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # Last non-padding token
            if attention_mask is not None:
                seq_lens = attention_mask.sum(dim=1) - 1
                last_hidden = out.last_hidden_state[
                    torch.arange(len(seq_lens), device=input_ids.device),
                    seq_lens,
                ]
            else:
                last_hidden = out.last_hidden_state[:, -1, :]
            return self.classifier(last_hidden)  # (B, 5) logits

    class RewardDataset(Dataset):
        def __init__(self, samples, tokenizer, max_len=128):
            self.samples = samples
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            s = self.samples[idx]
            enc = self.tokenizer(
                s["text"], max_length=self.max_len,
                truncation=True, padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(s["labels"], dtype=torch.float),
            }

    if not reward_model_trained and not skip_training:
        print("Training reward model from scratch...")
        reward_model = EmotionRewardModel().to(DEVICE)
        train_ds_rm = RewardDataset(train_samples, tokenizer_rm)
        test_ds_rm  = RewardDataset(test_samples,  tokenizer_rm)
        train_dl_rm = DataLoader(train_ds_rm, batch_size=64, shuffle=True,  num_workers=2)
        test_dl_rm  = DataLoader(test_ds_rm,  batch_size=64, shuffle=False, num_workers=2)

        optimizer_rm = torch.optim.Adam(
            reward_model.parameters(), lr=5e-5, weight_decay=0.01
        )
        criterion = nn.BCEWithLogitsLoss()

        # ── Resume from latest checkpoint if available ──────────────────
        RM_CKPT_DIR = f"{BASE}/reward_model/checkpoints"
        start_epoch_rm = 0
        for ep in range(7, -1, -1):
            ckpt_path = f"{RM_CKPT_DIR}/epoch_{ep+1}.pt"
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=DEVICE)
                reward_model.load_state_dict(ckpt["model"])
                optimizer_rm.load_state_dict(ckpt["optimizer"])
                start_epoch_rm = ep + 1
                print(f"  Resumed reward model from epoch {start_epoch_rm} checkpoint")
                break

        for epoch in range(start_epoch_rm, 8):
            reward_model.train()
            total_loss = 0
            for batch in tqdm(train_dl_rm, desc=f"RM Epoch {epoch+1}/8"):
                ids  = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                lbl  = batch["labels"].to(DEVICE)
                logits = reward_model(ids, mask)
                loss = criterion(logits, lbl)
                optimizer_rm.zero_grad()
                loss.backward()
                optimizer_rm.step()
                total_loss += loss.item()
            avg = total_loss / len(train_dl_rm)
            print(f"  Epoch {epoch+1}: loss={avg:.4f}")

            # ── Sanity check: loss should not be NaN ────────────────────
            assert not math.isnan(avg), \
                f"[STEP 2] FAILED: Reward model loss is NaN at epoch {epoch+1}"

            # ── Per-epoch checkpoint ────────────────────────────────────
            ckpt_path = f"{RM_CKPT_DIR}/epoch_{epoch+1}.pt"
            torch.save({"epoch": epoch + 1,
                        "model": reward_model.state_dict(),
                        "optimizer": optimizer_rm.state_dict(),
                        "loss": avg}, ckpt_path)
            VOLUME.commit()
            print(f"  Checkpoint saved: {ckpt_path}")

        # Eval
        reward_model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_dl_rm:
                ids  = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                lbl  = batch["labels"].to(DEVICE)
                logits = reward_model(ids, mask)
                preds = (torch.sigmoid(logits) > 0.5).float()
                all_preds.append(preds.cpu())
                all_labels.append(lbl.cpu())
        all_preds  = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        acc = (all_preds == all_labels).float().mean().item()
        print(f"  Test accuracy: {acc:.4f} (paper reports 0.84)")

        # ── Step 2 assertions ────────────────────────────────────────────
        assert 0.0 <= acc <= 1.0, \
            f"[STEP 2] FAILED: Reward model accuracy out of range: {acc}"
        assert acc > 0.5, \
            (f"[STEP 2] FAILED: Reward model accuracy {acc:.4f} is no better than chance. "
             "Check data or training setup.")
        print(f"  [STEP 2] ✅ Assertions passed — accuracy={acc:.4f}")

        # Save
        torch.save(reward_model.state_dict(), f"{REWARD_MODEL_PATH}/pytorch_model.bin")
        tokenizer_rm.save_pretrained(REWARD_MODEL_PATH)
        VOLUME.commit()
        print(f"  Saved reward model to {REWARD_MODEL_PATH}")
    else:
        print("Loading existing reward model...")
        reward_model = EmotionRewardModel().to(DEVICE)
        reward_model.load_state_dict(
            torch.load(f"{REWARD_MODEL_PATH}/pytorch_model.bin", map_location=DEVICE)
        )

    reward_model.eval()

    # ── Post-load assertion: verify reward model produces valid outputs ──────
    _test_enc = tokenizer_rm(["test sentence"], return_tensors="pt",
                              max_length=128, truncation=True, padding="max_length")
    with torch.no_grad():
        _test_logits = reward_model(
            _test_enc["input_ids"].to(DEVICE),
            _test_enc["attention_mask"].to(DEVICE),
        )
    assert _test_logits.shape == (1, 5), \
        f"[STEP 2] FAILED: Reward model output shape {_test_logits.shape}, expected (1, 5)"
    assert not torch.isnan(_test_logits).any(), \
        "[STEP 2] FAILED: Reward model produced NaN logits on test input"
    print("  [STEP 2] ✅ Reward model output shape and NaN check passed")

    # Helper: score a list of texts, returns (N, 5) numpy array of sigmoid outputs
    def score_texts(texts_list, batch_size=64):
        scores = []
        for i in range(0, len(texts_list), batch_size):
            batch_texts = texts_list[i:i+batch_size]
            enc = tokenizer_rm(
                batch_texts, max_length=128, truncation=True,
                padding="max_length", return_tensors="pt",
            )
            with torch.no_grad():
                logits = reward_model(
                    enc["input_ids"].to(DEVICE),
                    enc["attention_mask"].to(DEVICE),
                )
            scores.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(scores, axis=0)

    # =========================================================================
    # STEP 3: SFT GPT-2
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 3: SFT GPT-2 on emotion texts")
    print("="*60)

    SFT_MODEL_PATH = f"{BASE}/sft_model"
    sft_done = os.path.exists(f"{SFT_MODEL_PATH}/pytorch_model.bin")

    tokenizer_sft = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer_sft.pad_token = tokenizer_sft.eos_token

    class SFTDataset(Dataset):
        """Next-token prediction on emotion texts, max 68 tokens."""
        def __init__(self, texts, tokenizer, max_len=68):
            self.tokenizer = tokenizer
            self.max_len = max_len
            self.encodings = tokenizer(
                texts, max_length=max_len, truncation=True,
                padding="max_length", return_tensors="pt",
            )

        def __len__(self):
            return self.encodings["input_ids"].shape[0]

        def __getitem__(self, idx):
            ids  = self.encodings["input_ids"][idx]
            mask = self.encodings["attention_mask"][idx]
            # labels = ids shifted: -100 for padding
            labels = ids.clone()
            labels[mask == 0] = -100
            return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    if not sft_done:
        print("Training SFT model...")
        sft_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
        sft_ds = SFTDataset(orig_texts, tokenizer_sft)
        sft_dl = DataLoader(sft_ds, batch_size=64, shuffle=True, num_workers=2)

        optimizer_sft = torch.optim.Adam(
            sft_model.parameters(), lr=5e-7
        )
        # 12 warmup steps
        scheduler_sft = get_linear_schedule_with_warmup(
            optimizer_sft,
            num_warmup_steps=12,
            num_training_steps=10 * len(sft_dl),
        )

        # ── Resume from latest SFT checkpoint if available ──────────────
        SFT_CKPT_DIR = f"{BASE}/sft_model/checkpoints"
        start_epoch_sft = 0
        for ep in range(9, -1, -1):
            ckpt_path = f"{SFT_CKPT_DIR}/epoch_{ep+1}.pt"
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=DEVICE)
                sft_model.load_state_dict(ckpt["model"])
                optimizer_sft.load_state_dict(ckpt["optimizer"])
                scheduler_sft.load_state_dict(ckpt["scheduler"])
                start_epoch_sft = ep + 1
                print(f"  Resumed SFT from epoch {start_epoch_sft} checkpoint")
                break

        for epoch in range(start_epoch_sft, 10):
            sft_model.train()
            total_loss = 0
            for batch in tqdm(sft_dl, desc=f"SFT Epoch {epoch+1}/10"):
                ids    = batch["input_ids"].to(DEVICE)
                mask   = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                out = sft_model(input_ids=ids, attention_mask=mask, labels=labels)
                loss = out.loss
                optimizer_sft.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sft_model.parameters(), 10.0)
                optimizer_sft.step()
                scheduler_sft.step()
                total_loss += loss.item()
            avg_sft = total_loss / len(sft_dl)
            print(f"  SFT Epoch {epoch+1}: loss={avg_sft:.4f}")

            # ── Sanity check: loss should not be NaN ────────────────────
            assert not math.isnan(avg_sft), \
                f"[STEP 3] FAILED: SFT loss is NaN at epoch {epoch+1}"

            # ── Per-epoch checkpoint ────────────────────────────────────
            ckpt_path = f"{SFT_CKPT_DIR}/epoch_{epoch+1}.pt"
            torch.save({"epoch": epoch + 1,
                        "model": sft_model.state_dict(),
                        "optimizer": optimizer_sft.state_dict(),
                        "scheduler": scheduler_sft.state_dict(),
                        "loss": avg_sft}, ckpt_path)
            VOLUME.commit()
            print(f"  Checkpoint saved: {ckpt_path}")

        sft_model.save_pretrained(SFT_MODEL_PATH)
        tokenizer_sft.save_pretrained(SFT_MODEL_PATH)
        VOLUME.commit()
        print(f"  Saved SFT model to {SFT_MODEL_PATH}")
    else:
        print("Loading existing SFT model...")
        sft_model = GPT2LMHeadModel.from_pretrained(SFT_MODEL_PATH).to(DEVICE)

    sft_model.eval()

    # ── Step 3 assertions: verify SFT model is sane ──────────────────────────
    assert os.path.exists(f"{SFT_MODEL_PATH}/pytorch_model.bin") or \
           os.path.exists(f"{SFT_MODEL_PATH}/model.safetensors"), \
        f"[STEP 3] FAILED: SFT model weights not found at {SFT_MODEL_PATH}"
    _sft_enc = tokenizer_sft(["hello world"], return_tensors="pt")
    with torch.no_grad():
        _sft_out = sft_model(**{k: v.to(DEVICE) for k, v in _sft_enc.items()})
    assert _sft_out.logits is not None, \
        "[STEP 3] FAILED: SFT model forward pass returned no logits"
    assert not torch.isnan(_sft_out.logits).any(), \
        "[STEP 3] FAILED: SFT model produced NaN logits"
    print("  [STEP 3] ✅ SFT model assertions passed")

    # =========================================================================
    # STEP 4: Generate completions
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 4: Generating completions")
    print("="*60)

    COMPLETIONS_PATH = f"{BASE}/completions/completions.jsonl"

    if not os.path.exists(COMPLETIONS_PATH):
        print("Generating 2 completions per prompt...")
        # Prompts = first 4 tokens of each text
        prompts_raw = orig_texts
        completions_data = []

        for text in tqdm(prompts_raw[:5000], desc="Generating"):  # cap at 5000 prompts
            # First 4 tokens
            enc = tokenizer_sft(text, return_tensors="pt", truncation=True, max_length=68)
            prompt_ids = enc["input_ids"][0, :4].unsqueeze(0).to(DEVICE)

            two_completions = []
            for _ in range(2):
                with torch.no_grad():
                    out = sft_model.generate(
                        prompt_ids,
                        max_new_tokens=64,
                        do_sample=True,
                        top_k=0,
                        top_p=1.0,
                        pad_token_id=tokenizer_sft.eos_token_id,
                    )
                # Decode only the new tokens
                new_tokens = out[0, prompt_ids.shape[1]:]
                completion_text = tokenizer_sft.decode(
                    new_tokens, skip_special_tokens=True
                )
                prompt_text = tokenizer_sft.decode(
                    prompt_ids[0], skip_special_tokens=True
                )
                two_completions.append(prompt_text + completion_text)

            completions_data.append({
                "prompt": tokenizer_sft.decode(prompt_ids[0], skip_special_tokens=True),
                "completion_1": two_completions[0],
                "completion_2": two_completions[1],
            })

        with open(COMPLETIONS_PATH, "w") as f:
            for item in completions_data:
                f.write(json.dumps(item) + "\n")
        VOLUME.commit()
        print(f"  Saved {len(completions_data)} completion pairs")
    else:
        print("Loading existing completions...")
        with open(COMPLETIONS_PATH) as f:
            completions_data = [json.loads(l) for l in f]
        print(f"  Loaded {len(completions_data)} completion pairs")

    # ── Step 4 assertions ────────────────────────────────────────────────────
    assert len(completions_data) > 0, \
        "[STEP 4] FAILED: completions_data is empty"
    assert all(
        "prompt" in d and "completion_1" in d and "completion_2" in d
        for d in completions_data[:20]
    ), "[STEP 4] FAILED: Completion entries missing required keys"
    assert all(
        len(d["completion_1"]) > 0 and len(d["completion_2"]) > 0
        for d in completions_data[:20]
    ), "[STEP 4] FAILED: Some completions are empty strings"
    print(f"  [STEP 4] ✅ Assertions passed — {len(completions_data)} completion pairs")

    # =========================================================================
    # STEP 5: Label preferences
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 5: Labeling preferences (alpha_0=0.1, r1=anger, r2=fear)")
    print("="*60)

    # Score all completions once
    ANGER_IDX = EMOTION_TO_IDX["anger"]   # 3
    FEAR_IDX  = EMOTION_TO_IDX["fear"]    # 4

    PREFS_CONVEX_PATH   = f"{BASE}/preferences/prefs_convex.jsonl"
    PREFS_GEOMETRIC_PATH = f"{BASE}/preferences/prefs_geometric.jsonl"

    def reward_convex(scores, alpha):
        """r*(alpha) = alpha * r_anger + (1-alpha) * r_fear"""
        return alpha * scores[:, ANGER_IDX] + (1 - alpha) * scores[:, FEAR_IDX]

    def reward_geometric(scores, alpha):
        """r*(alpha) = r_anger^alpha * r_fear^(1-alpha)"""
        # clip to avoid log(0)
        a = np.clip(scores[:, ANGER_IDX], 1e-8, None)
        f = np.clip(scores[:, FEAR_IDX],  1e-8, None)
        return (a ** alpha) * (f ** (1 - alpha))

    def bt_label(r1_score, r2_score, rng):
        """Bradley-Terry: P(c1 > c2) = sigmoid(r(c1) - r(c2)), sample stochastically."""
        prob_c1_wins = 1.0 / (1.0 + np.exp(-(r1_score - r2_score)))
        return 1 if rng.random() < prob_c1_wins else 0  # 1 = c1 preferred

    if not os.path.exists(PREFS_CONVEX_PATH):
        print("Scoring completions with reward model...")
        c1_texts = [d["completion_1"] for d in completions_data]
        c2_texts = [d["completion_2"] for d in completions_data]
        scores_c1 = score_texts(c1_texts)  # (N, 5)
        scores_c2 = score_texts(c2_texts)  # (N, 5)

        ALPHA_0 = 0.1
        rng = random.Random(42)

        # Convex mixing preferences
        r_c1_convex = reward_convex(scores_c1, ALPHA_0)
        r_c2_convex = reward_convex(scores_c2, ALPHA_0)

        # Geometric mixing preferences
        r_c1_geo = reward_geometric(scores_c1, ALPHA_0)
        r_c2_geo = reward_geometric(scores_c2, ALPHA_0)

        prefs_convex, prefs_geo = [], []
        for i, d in enumerate(completions_data):
            label_c = bt_label(r_c1_convex[i], r_c2_convex[i], rng)
            label_g = bt_label(r_c1_geo[i],    r_c2_geo[i],    rng)
            base = {"prompt": d["prompt"],
                    "completion_1": d["completion_1"],
                    "completion_2": d["completion_2"]}
            prefs_convex.append({**base, "label": label_c,
                                  "chosen":   d[f"completion_{label_c+1}"],
                                  "rejected": d[f"completion_{2-label_c}"]})
            prefs_geo.append({**base, "label": label_g,
                               "chosen":   d[f"completion_{label_g+1}"],
                               "rejected": d[f"completion_{2-label_g}"]})

        # Also cache raw scores for eval
        np.save(f"{BASE}/preferences/scores_c1.npy", scores_c1)
        np.save(f"{BASE}/preferences/scores_c2.npy", scores_c2)

        with open(PREFS_CONVEX_PATH, "w") as f:
            for p in prefs_convex: f.write(json.dumps(p) + "\n")
        with open(PREFS_GEOMETRIC_PATH, "w") as f:
            for p in prefs_geo:    f.write(json.dumps(p) + "\n")
        VOLUME.commit()
        print(f"  Saved {len(prefs_convex)} convex and geometric preference pairs")
    else:
        print("Loading existing preferences...")
        with open(PREFS_CONVEX_PATH) as f:
            prefs_convex = [json.loads(l) for l in f]
        with open(PREFS_GEOMETRIC_PATH) as f:
            prefs_geo = [json.loads(l) for l in f]
        scores_c1 = np.load(f"{BASE}/preferences/scores_c1.npy")
        scores_c2 = np.load(f"{BASE}/preferences/scores_c2.npy")

    # ── Step 5 assertions ────────────────────────────────────────────────────
    assert len(prefs_convex) > 0, \
        "[STEP 5] FAILED: prefs_convex is empty"
    assert len(prefs_geo) > 0, \
        "[STEP 5] FAILED: prefs_geo is empty"
    assert len(prefs_convex) == len(prefs_geo), \
        (f"[STEP 5] FAILED: Convex and geometric preference sets have different lengths "
         f"({len(prefs_convex)} vs {len(prefs_geo)})")
    assert all(
        "chosen" in p and "rejected" in p and "label" in p
        for p in prefs_convex[:20]
    ), "[STEP 5] FAILED: Convex preference entries missing required keys"
    assert scores_c1.shape == scores_c2.shape and scores_c1.ndim == 2 and scores_c1.shape[1] == 5, \
        f"[STEP 5] FAILED: Unexpected reward score shape: {scores_c1.shape}"
    print(f"  [STEP 5] ✅ Assertions passed — {len(prefs_convex)} preference pairs, "
          f"score arrays shape {scores_c1.shape}")

    # =========================================================================
    # STEP 6: Train DPO / WDPO / KLDPO on both mixing types
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 6: Training DPO / WDPO / KLDPO (40 epochs)")
    print("="*60)

    # ── DPO loss helpers ────────────────────────────────────────────────────

    def get_log_probs(model, input_ids, attention_mask, labels_ids):
        """Per-sequence sum of log probs for non-padding tokens."""
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, :-1, :]           # (B, T-1, V)
        targets = labels_ids[:, 1:]              # (B, T-1)
        mask = (targets != tokenizer_sft.pad_token_id).float()
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, targets.unsqueeze(2)).squeeze(2)
        return (token_log_probs * mask).sum(dim=1)  # (B,)

    def dpo_loss(policy, ref_model, chosen_ids, chosen_mask,
                 rejected_ids, rejected_mask, beta=0.1):
        """Standard DPO loss."""
        with torch.no_grad():
            ref_chosen   = get_log_probs(ref_model, chosen_ids,   chosen_mask,   chosen_ids)
            ref_rejected = get_log_probs(ref_model, rejected_ids, rejected_mask, rejected_ids)

        pol_chosen   = get_log_probs(policy, chosen_ids,   chosen_mask,   chosen_ids)
        pol_rejected = get_log_probs(policy, rejected_ids, rejected_mask, rejected_ids)

        log_ratio = (pol_chosen - ref_chosen) - (pol_rejected - ref_rejected)
        loss = -F.logsigmoid(beta * log_ratio).mean()
        return loss

    def wdpo_loss(policy, ref_model, chosen_ids, chosen_mask,
                  rejected_ids, rejected_mask, beta=0.1, rho=50.0):
        """
        WDPO: DPO loss + gradient regularization penalty.
        L_W = L_DPO + rho * sqrt(mean(||grad_embed(l)||^2))
        Approximated as L_DPO + rho * ||grad_embed(L_DPO)||_2
        per the pointwise upper bound from Appendix F.
        """
        # Need gradients w.r.t. embeddings for the penalty
        embed = policy.transformer.wte  # word token embeddings

        # Forward with embedding gradient tracking
        chosen_embed   = embed(chosen_ids.to(DEVICE))
        rejected_embed = embed(rejected_ids.to(DEVICE))
        chosen_embed.retain_grad()
        rejected_embed.retain_grad()

        # Compute base DPO loss
        base_loss = dpo_loss(
            policy, ref_model,
            chosen_ids, chosen_mask,
            rejected_ids, rejected_mask, beta,
        )

        # Gradient of loss w.r.t. embeddings
        grads = torch.autograd.grad(
            base_loss, [chosen_embed, rejected_embed],
            create_graph=True, retain_graph=True,
        )
        grad_norm = torch.stack([g.norm() for g in grads]).mean()
        return base_loss + rho * grad_norm

    def kldpo_loss(policy, ref_model, chosen_ids, chosen_mask,
                   rejected_ids, rejected_mask, beta=0.1, tau=0.5):
        """
        KLDPO: reweight samples by worst-case KL distribution.
        P(i) ∝ exp((1/tau) * (l_i - mean_l))
        """
        with torch.no_grad():
            ref_chosen   = get_log_probs(ref_model, chosen_ids,   chosen_mask,   chosen_ids)
            ref_rejected = get_log_probs(ref_model, rejected_ids, rejected_mask, rejected_ids)

        pol_chosen   = get_log_probs(policy, chosen_ids,   chosen_mask,   chosen_ids)
        pol_rejected = get_log_probs(policy, rejected_ids, rejected_mask, rejected_ids)

        log_ratio = (pol_chosen - ref_chosen) - (pol_rejected - ref_rejected)
        pointwise_loss = -F.logsigmoid(beta * log_ratio)  # (B,)

        # Worst-case reweighting
        with torch.no_grad():
            mean_loss = pointwise_loss.mean()
            weights = F.softmax((pointwise_loss - mean_loss) / tau, dim=0)

        return (weights * pointwise_loss).sum()

    # ── Dataset for DPO training ─────────────────────────────────────────
    class DPODataset(Dataset):
        def __init__(self, prefs, tokenizer, max_len=128):
            self.prefs = prefs
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.prefs)

        def __getitem__(self, idx):
            p = self.prefs[idx]

            def enc(text):
                e = self.tokenizer(
                    text, max_length=self.max_len, truncation=True,
                    padding="max_length", return_tensors="pt",
                )
                return e["input_ids"].squeeze(0), e["attention_mask"].squeeze(0)

            c_ids, c_mask = enc(p["chosen"])
            r_ids, r_mask = enc(p["rejected"])
            return c_ids, c_mask, r_ids, r_mask

    # ── Training loop ────────────────────────────────────────────────────
    def train_dpo_variant(
        prefs, method, save_path,
        beta=0.1, rho=None, tau=None,
        epochs=40, batch_size=32, grad_accum=2, lr=5e-7,
    ):
        """
        Train one DPO variant. Effective batch = batch_size * grad_accum = 64.
        Single GPU, plain Adam, 12 warmup steps.
        Saves an epoch-level checkpoint every 5 epochs so that GPU credits
        are not lost if the pipeline crashes mid-run.
        """
        saved = (os.path.exists(f"{save_path}/pytorch_model.bin") or os.path.exists(f"{save_path}/model.safetensors"))
        if saved:
            print(f"  Already trained: {save_path}")
            return

        print(f"  Training {method} -> {save_path}")
        policy = GPT2LMHeadModel.from_pretrained(SFT_MODEL_PATH).to(DEVICE)
        ref_model = GPT2LMHeadModel.from_pretrained(SFT_MODEL_PATH).to(DEVICE)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        dataset = DPODataset(prefs, tokenizer_sft)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        total_steps = epochs * math.ceil(len(loader) / grad_accum)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=12, num_training_steps=total_steps
        )

        # ── Resume from latest DPO checkpoint if available ──────────────
        dpo_ckpt_dir = f"{save_path}/checkpoints"
        os.makedirs(dpo_ckpt_dir, exist_ok=True)
        start_epoch = 0
        for ep in range(epochs - 1, -1, -1):
            ckpt_path = f"{dpo_ckpt_dir}/epoch_{ep+1}.pt"
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=DEVICE)
                policy.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])
                scheduler.load_state_dict(ckpt["scheduler"])
                start_epoch = ep + 1
                print(f"    Resumed {method} from epoch {start_epoch} checkpoint")
                break

        for epoch in range(start_epoch, epochs):
            policy.train()
            total_loss = 0
            optimizer.zero_grad()
            for step, batch in enumerate(loader):
                c_ids, c_mask, r_ids, r_mask = [x.to(DEVICE) for x in batch]

                if method == "dpo":
                    loss = dpo_loss(policy, ref_model, c_ids, c_mask, r_ids, r_mask, beta)
                elif method == "wdpo":
                    loss = wdpo_loss(policy, ref_model, c_ids, c_mask, r_ids, r_mask, beta, rho)
                elif method == "kldpo":
                    loss = kldpo_loss(policy, ref_model, c_ids, c_mask, r_ids, r_mask, beta, tau)

                (loss / grad_accum).backward()
                total_loss += loss.item()

                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            avg_loss = total_loss / len(loader)
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

            # ── NaN guard ───────────────────────────────────────────────
            assert not math.isnan(avg_loss), \
                (f"[STEP 6] FAILED: {method} loss is NaN at epoch {epoch+1} "
                 f"(save_path={save_path})")

            # ── Checkpoint every 5 epochs ───────────────────────────────
            if (epoch + 1) % 5 == 0:
                ckpt_path = f"{dpo_ckpt_dir}/epoch_{epoch+1}.pt"
                torch.save({"epoch": epoch + 1,
                            "model": policy.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "loss": avg_loss}, ckpt_path)
                VOLUME.commit()
                print(f"    Checkpoint saved: {ckpt_path}")

        os.makedirs(save_path, exist_ok=True)
        policy.save_pretrained(save_path)
        tokenizer_sft.save_pretrained(save_path)
        VOLUME.commit()
        print(f"  Saved to {save_path}")

        # ── Step 6 per-model assertion ───────────────────────────────────
        assert os.path.exists(f"{save_path}/pytorch_model.bin") or \
               os.path.exists(f"{save_path}/model.safetensors"), \
            f"[STEP 6] FAILED: Model weights not found after training at {save_path}"

    if not skip_training:
        for mix_type, prefs in [("convex", prefs_convex), ("geometric", prefs_geo)]:
            print(f"\n  === {mix_type.upper()} MIXING ===")

            train_dpo_variant(prefs, "dpo",
                f"{BASE}/trained_models/dpo_{mix_type}")

            for tau in [0.5, 0.75, 1.0]:
                train_dpo_variant(prefs, "kldpo",
                    f"{BASE}/trained_models/kldpo_{mix_type}_tau{tau}", tau=tau)

            for rho in [50, 75, 100]:
                train_dpo_variant(prefs, "wdpo",
                    f"{BASE}/trained_models/wdpo_{mix_type}_rho{rho}", rho=rho)

        # ── Step 6 assertions: all expected model dirs must exist ────────────
        expected_models = (
            [f"dpo_{mt}" for mt in ["convex", "geometric"]] +
            [f"kldpo_{mt}_tau{tau}" for mt in ["convex", "geometric"]
                                    for tau in [0.5, 0.75, 1.0]] +
            [f"wdpo_{mt}_rho{rho}" for mt in ["convex", "geometric"]
                                   for rho in [50, 75, 100]]
        )
        missing = [
            m for m in expected_models
            if not (os.path.exists(f"{BASE}/trained_models/{m}/pytorch_model.bin") or
                    os.path.exists(f"{BASE}/trained_models/{m}/model.safetensors"))
        ]
        assert not missing, \
            f"[STEP 6] FAILED: The following models are missing after training: {missing}"
        print(f"  [STEP 6] ✅ All {len(expected_models)} DPO variant models confirmed saved")

    # =========================================================================
    # STEP 7 & 8: Evaluate and plot Figure 2
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 7: Evaluating all models and plotting Figure 2")
    print("="*60)

    ALPHAS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    def eval_model_at_alpha(model_path, mix_fn, alpha):
        """
        Generate completions from a trained model on the test prompts,
        score them, return mean reward under mix_fn(alpha).
        """
        model = GPT2LMHeadModel.from_pretrained(model_path).to(DEVICE)
        model.eval()

        # Use the first 500 prompts from completions_data for fast eval
        prompts = [d["prompt"] for d in completions_data[:500]]
        all_completions = []

        for prompt in prompts:
            enc = tokenizer_sft(prompt, return_tensors="pt", truncation=True, max_length=10)
            ids = enc["input_ids"].to(DEVICE)
            with torch.no_grad():
                out = model.generate(
                    ids, max_new_tokens=64, do_sample=False,
                    pad_token_id=tokenizer_sft.eos_token_id,
                )
            new_tokens = out[0, ids.shape[1]:]
            text = tokenizer_sft.decode(new_tokens, skip_special_tokens=True)
            all_completions.append(prompt + text)

        scores = score_texts(all_completions)  # (N, 5)
        rewards = mix_fn(scores, alpha)
        return float(np.mean(rewards))

    # Build results dict: results[mix_type][model_label][alpha] = reward
    results = {
        "convex":   {},
        "geometric": {},
    }

    mix_fns = {
        "convex":    reward_convex,
        "geometric": reward_geometric,
    }

    model_configs = [
        ("DPO",            "dpo",   None,  None),
        ("KLDPO-0.5",      "kldpo", None,  0.5),
        ("KLDPO-0.75",     "kldpo", None,  0.75),
        ("KLDPO-1",        "kldpo", None,  1.0),
        ("WDPO-50",        "wdpo",  50,    None),
        ("WDPO-75",        "wdpo",  75,    None),
        ("WDPO-100",       "wdpo",  100,   None),
    ]

    for mix_type, mix_fn in mix_fns.items():
        print(f"\n  Evaluating {mix_type} mixing...")
        for label, method, rho, tau in model_configs:
            if method == "dpo":
                path = f"{BASE}/trained_models/dpo_{mix_type}"
            elif method == "kldpo":
                path = f"{BASE}/trained_models/kldpo_{mix_type}_tau{tau}"
            elif method == "wdpo":
                path = f"{BASE}/trained_models/wdpo_{mix_type}_rho{rho}"

            if not os.path.exists(f"{path}/pytorch_model.bin"):
                print(f"    Skipping {label} (not trained)")
                continue

            rewards = []
            for alpha in ALPHAS:
                r = eval_model_at_alpha(path, mix_fn, alpha)
                rewards.append(r)
                print(f"    {label} alpha={alpha:.1f}: {r:.4f}")
            results[mix_type][label] = rewards

    # Also compute nominal (SFT baseline at alpha_0=0.1)
    for mix_type, mix_fn in mix_fns.items():
        nominal_rewards = []
        for alpha in ALPHAS:
            scores = score_texts([d["completion_1"] for d in completions_data[:500]])
            r = float(np.mean(mix_fn(scores, alpha)))
            nominal_rewards.append(r)
        results[mix_type]["nominal_0"] = nominal_rewards

    # ── Plot ────────────────────────────────────────────────────────────────
    COLORS = {
        "DPO":       "black",
        "KLDPO-0.5": "blue",
        "KLDPO-0.75": "royalblue",
        "KLDPO-1":   "steelblue",
        "WDPO-50":   "red",
        "WDPO-75":   "tomato",
        "WDPO-100":  "salmon",
        "nominal_0": "gray",
    }
    LINESTYLES = {
        "DPO": "--", "nominal_0": ":",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    titles = {
        "convex":    r"Mixture Reward $r^*_{\mathrm{convex}}(\alpha)$",
        "geometric": r"Mixture Reward $r^*_{\mathrm{geometric}}(\alpha)$",
    }

    for ax, (mix_type, title) in zip(axes, titles.items()):
        for label, rewards in results[mix_type].items():
            ax.plot(
                ALPHAS, rewards,
                label=label,
                color=COLORS.get(label, "purple"),
                linestyle=LINESTYLES.get(label, "-"),
                marker="o", markersize=4,
            )
        ax.axvline(x=0.1, color="gray", linestyle=":", alpha=0.5, label="train α₀=0.1")
        ax.set_xlabel("Alpha")
        ax.set_ylabel(title)
        ax.set_title(f"{mix_type.capitalize()} Mixing")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 2: DPO, WDPO, KLDPO in Emotion Alignment", fontsize=13)
    plt.tight_layout()

    out_path = "/vol/results/emotion_figure2.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure 2 saved to {out_path}")

    # Also save raw numbers as JSON for later inspection
    with open("/vol/results/emotion_results.json", "w") as f:
        json.dump({"alphas": ALPHAS, "results": results}, f, indent=2)
    print("Raw results saved to /vol/results/emotion_results.json")

    VOLUME.commit()
    print("\nDone. All outputs committed to volume.")


# =============================================================================
# Local entrypoint
# =============================================================================

@app.local_entrypoint()
def main():
    print("Emotion experiment. Run with:")
    print("  modal run emotion_experiment.py::run_emotion_experiment")
    print("  modal run emotion_experiment.py::run_emotion_experiment --skip-training")