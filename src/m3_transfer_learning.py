# ============================================================
#  m3_transfer_learning.py
#  Phase 3 – Transfer Learning + Optuna Hyperparameter Tuning
#  Framework  : PyTorch
#  Backbone   : ResNet50 (ImageNet pre-trained)
#  Dataset    : blanchon/EuroSAT_RGB  (10 classes, 64×64 RGB)
#  Regularisation: Dropout + Weight Decay (L2) only
# ============================================================

import os
import csv
import time
import platform

import numpy as np
from datasets import load_dataset
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

import optuna
from optuna.trial import Trial
from sklearn.metrics import classification_report, confusion_matrix

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Constants ────────────────────────────────────────────────────────────────
NUM_CLASSES   = 10
LOG_FILE      = "optuna_hyperparameter_log.csv"
MODEL_SAVE    = "optimized_transfer_model.pth"

# Epoch budgets (kept small to save time)
WARMUP_EPOCHS_DEFAULT   = 2   # head-only warm-up
FINETUNE_EPOCHS_DEFAULT = 3   # backbone fine-tuning

# Optuna trial budget
N_TRIALS = 5    # increase for a more thorough search


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING & PRE-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def extract_images_labels(split):
    """
    Convert a Hugging Face dataset split to float32 NumPy arrays.
    Returns images normalised to [0, 1] in (N, H, W, C) layout.
    """
    images, labels = [], []
    for sample in split:
        img = np.array(sample["image"], dtype=np.float32) / 255.0
        images.append(img)
        labels.append(sample["label"])
    return np.array(images), np.array(labels)


class EuroSATDataset(Dataset):
    """
    Wraps NumPy arrays into a PyTorch Dataset.
    Applies the ImageNet normalisation transform expected by ResNet50.
    Images are resized to 224×224 (required by the pre-trained backbone).
    """

    # ImageNet statistics
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.labels = torch.tensor(labels, dtype=torch.long)

        # Pre-build transform: resize → tensor → normalise
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

        # Keep PIL images so the transform can work
        self.pil_images = [
            Image.fromarray((img * 255).astype(np.uint8))
            for img in images
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.transform(self.pil_images[idx]), self.labels[idx]


def get_dataloader_config() -> dict:
    """Return OS-appropriate DataLoader kwargs."""
    if platform.system() == "Windows":
        return {"num_workers": 0, "pin_memory": False}
    return {"num_workers": 2, "pin_memory": True}


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MODEL BUILDING
# ─────────────────────────────────────────────────────────────────────────────

def build_transfer_model(dropout_rate: float = 0.4,
                         num_classes:  int   = NUM_CLASSES) -> nn.Module:
    """
    ResNet50 pre-trained on ImageNet.
    FC head replaced with:   Dropout → Linear → ReLU → Dropout → Linear
    Regularisation:  Dropout (in head) + Weight Decay / L2 (in optimiser).
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features   # 2048

    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=dropout_rate / 2),
        nn.Linear(512, num_classes),
    )

    total = sum(p.numel() for p in model.parameters())
    print(f"  [Model] Total parameters: {total:,}")
    return model


def freeze_backbone(model: nn.Module):
    """Freeze all layers except the FC head (warm-up stage)."""
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [FREEZE] Trainable params: {trainable:,}  (head only)")


def unfreeze_backbone(model: nn.Module, unfreeze_from: str = "layer4"):
    """
    Unfreeze ResNet50 from a given block onward (fine-tuning stage).
    Blocks:  layer1 → layer2 → layer3 → layer4 → fc
    """
    do_unfreeze = False
    for name, param in model.named_parameters():
        if unfreeze_from in name:
            do_unfreeze = True
        if do_unfreeze:
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [UNFREEZE from {unfreeze_from}] Trainable params: {trainable:,}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, is_train) -> tuple:
    """One full pass over a DataLoader. Returns (avg_loss, accuracy)."""
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss  += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct     += (predicted == labels).sum().item()
            total       += labels.size(0)

    return total_loss / total, correct / total


def log_trial_to_csv(row: dict):
    """Append one trial's results to the CSV log file."""
    write_header = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  OPTUNA OBJECTIVE
# ─────────────────────────────────────────────────────────────────────────────

def objective(trial: Trial) -> float:
    """
    Optuna searches over:
      ├── dropout_rate    : 0.2 → 0.6
      ├── warmup_lr       : 1e-4 → 1e-2  (log scale)
      ├── finetune_lr     : 1e-6 → 1e-3  (log scale)
      ├── weight_decay    : 1e-6 → 1e-2  (log scale)   ← L2 regularisation
      ├── batch_size      : 32 or 64
      └── unfreeze_from   : layer3 or layer4

    Epochs are fixed to WARMUP_EPOCHS_DEFAULT / FINETUNE_EPOCHS_DEFAULT
    to keep total runtime manageable.

    Returns the best validation accuracy seen across all epochs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Suggest hyperparameters ───────────────────────────────────────────────
    dropout_rate  = trial.suggest_float("dropout_rate", 0.2,  0.6)
    warmup_lr     = trial.suggest_float("warmup_lr",    1e-4, 1e-2,  log=True)
    finetune_lr   = trial.suggest_float("finetune_lr",  1e-6, 1e-3,  log=True)
    weight_decay  = trial.suggest_float("weight_decay", 1e-6, 1e-2,  log=True)
    batch_size    = trial.suggest_categorical("batch_size",   [32, 64])
    unfreeze_from = trial.suggest_categorical("unfreeze_from", ["layer3", "layer4"])

    warmup_epochs   = WARMUP_EPOCHS_DEFAULT
    finetune_epochs = FINETUNE_EPOCHS_DEFAULT

    print(f"\n{'─'*60}")
    print(f"  Trial {trial.number}")
    print(f"  dropout_rate  = {dropout_rate:.3f}")
    print(f"  warmup_lr     = {warmup_lr:.2e}")
    print(f"  finetune_lr   = {finetune_lr:.2e}")
    print(f"  weight_decay  = {weight_decay:.2e}  (L2)")
    print(f"  batch_size    = {batch_size}")
    print(f"  unfreeze_from = {unfreeze_from}")
    print(f"  warmup_epochs = {warmup_epochs}  |  finetune_epochs = {finetune_epochs}")
    print(f"{'─'*60}\n")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    dl_cfg       = get_dataloader_config()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **dl_cfg)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **dl_cfg)

    # ── Model & loss ──────────────────────────────────────────────────────────
    model     = build_transfer_model(dropout_rate, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_val_acc  = 0.0
    best_weights  = None
    start_time    = time.time()

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1 – WARM-UP  (backbone frozen, only FC head trained)
    # ══════════════════════════════════════════════════════════════════════════
    print("  ── Stage 1: Warm-Up (backbone frozen) ──")
    freeze_backbone(model)

    optimizer_warmup = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = warmup_lr,
        weight_decay = weight_decay,   # ← L2 regularisation
    )
    scheduler_warmup = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_warmup, mode="min", factor=0.5, patience=2
    )
    prev_lr = warmup_lr

    for epoch in range(1, warmup_epochs + 1):
        tr_loss, tr_acc   = run_epoch(model, train_loader, criterion,
                                      optimizer_warmup, device, is_train=True)
        val_loss, val_acc = run_epoch(model, val_loader,   criterion,
                                      None,               device, is_train=False)

        scheduler_warmup.step(val_loss)
        cur_lr = optimizer_warmup.param_groups[0]["lr"]
        if cur_lr != prev_lr:
            print(f"    📉 LR: {prev_lr:.2e} → {cur_lr:.2e}")
            prev_lr = cur_lr

        print(f"    [WarmUp] Ep {epoch}/{warmup_epochs} | "
              f"TrLoss:{tr_loss:.4f} TrAcc:{tr_acc:.4f} | "
              f"VLoss:{val_loss:.4f} VAcc:{val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            best_weights  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2 – FINE-TUNING  (partial backbone unfrozen, small LR)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  ── Stage 2: Fine-Tuning (unfreeze from {unfreeze_from}) ──")
    unfreeze_backbone(model, unfreeze_from)

    optimizer_finetune = torch.optim.Adam([
        {
            "params": [p for n, p in model.named_parameters()
                       if "fc" not in n and p.requires_grad],
            "lr": finetune_lr,            # small LR for backbone layers
        },
        {
            "params": model.fc.parameters(),
            "lr":     finetune_lr * 10,   # larger LR for the head
        },
    ], weight_decay=weight_decay)         # ← L2 regularisation

    scheduler_finetune = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_finetune, T_max=finetune_epochs, eta_min=1e-7
    )

    for epoch in range(1, finetune_epochs + 1):
        tr_loss, tr_acc   = run_epoch(model, train_loader, criterion,
                                      optimizer_finetune, device, is_train=True)
        val_loss, val_acc = run_epoch(model, val_loader,   criterion,
                                      None,                device, is_train=False)

        scheduler_finetune.step()
        cur_lr = optimizer_finetune.param_groups[0]["lr"]

        print(f"    [FineTune] Ep {epoch}/{finetune_epochs} | "
              f"TrLoss:{tr_loss:.4f} TrAcc:{tr_acc:.4f} | "
              f"VLoss:{val_loss:.4f} VAcc:{val_acc:.4f} | "
              f"LR:{cur_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            best_weights  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"    ✅ New best val_loss={best_val_loss:.4f}")

        global_step = warmup_epochs + epoch
        trial.report(val_acc, global_step)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # ── Log trial to CSV ──────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    log_trial_to_csv({
        "trial_number"   : trial.number,
        "dropout_rate"   : round(dropout_rate, 4),
        "warmup_lr"      : warmup_lr,
        "finetune_lr"    : finetune_lr,
        "weight_decay"   : weight_decay,
        "batch_size"     : batch_size,
        "unfreeze_from"  : unfreeze_from,
        "warmup_epochs"  : warmup_epochs,
        "finetune_epochs": finetune_epochs,
        "best_val_acc"   : round(best_val_acc,  4),
        "best_val_loss"  : round(best_val_loss, 4),
        "training_time_s": round(elapsed, 1),
    })

    print(f"\n  Trial {trial.number} done | "
          f"best_val_acc={best_val_acc:.4f} | time={elapsed:.1f}s")

    trial.set_user_attr("best_weights", best_weights)
    return best_val_acc   # Optuna maximises this


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FINAL RETRAIN WITH BEST PARAMS
# ─────────────────────────────────────────────────────────────────────────────

def train_final_model(params: dict) -> nn.Module:
    """
    Retrain from scratch using the best Optuna parameters.
    Includes early stopping based on validation loss.
    Saves final weights to MODEL_SAVE.
    """
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl_cfg    = get_dataloader_config()

    train_loader = DataLoader(train_ds, batch_size=params["batch_size"],
                              shuffle=True,  **dl_cfg)
    val_loader   = DataLoader(val_ds,   batch_size=params["batch_size"],
                              shuffle=False, **dl_cfg)

    model     = build_transfer_model(params["dropout_rate"], NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_weights  = None

    # ── Stage 1: Warm-Up ─────────────────────────────────────────────────────
    print("\n── [Final] Stage 1: Warm-Up ──")
    freeze_backbone(model)

    optimizer_warmup = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = params["warmup_lr"],
        weight_decay = params["weight_decay"],
    )
    scheduler_warmup = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_warmup, mode="min", factor=0.5, patience=2
    )
    prev_lr        = params["warmup_lr"]
    patience_count = 0
    patience_limit = 3   # early stopping patience

    for epoch in range(1, params["warmup_epochs"] + 1):
        tr_loss, tr_acc   = run_epoch(model, train_loader, criterion,
                                      optimizer_warmup, device, is_train=True)
        val_loss, val_acc = run_epoch(model, val_loader,   criterion,
                                      None,               device, is_train=False)

        scheduler_warmup.step(val_loss)
        cur_lr = optimizer_warmup.param_groups[0]["lr"]
        if cur_lr != prev_lr:
            print(f"    📉 LR: {prev_lr:.2e} → {cur_lr:.2e}")
            prev_lr = cur_lr

        print(f"  [WarmUp] Ep {epoch}/{params['warmup_epochs']} | "
              f"TrLoss:{tr_loss:.4f} TrAcc:{tr_acc:.4f} | "
              f"VLoss:{val_loss:.4f} VAcc:{val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_weights   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
            print(f"    ✅ Best val_loss={best_val_loss:.4f}")
        else:
            patience_count += 1
            if patience_count >= patience_limit:
                print(f"    ⏹  Early stopping triggered at warm-up epoch {epoch}")
                break

    # ── Stage 2: Fine-Tuning ─────────────────────────────────────────────────
    print(f"\n── [Final] Stage 2: Fine-Tuning (unfreeze from {params['unfreeze_from']}) ──")
    unfreeze_backbone(model, params["unfreeze_from"])

    optimizer_finetune = torch.optim.Adam([
        {
            "params": [p for n, p in model.named_parameters()
                       if "fc" not in n and p.requires_grad],
            "lr": params["finetune_lr"],
        },
        {
            "params": model.fc.parameters(),
            "lr":     params["finetune_lr"] * 10,
        },
    ], weight_decay=params["weight_decay"])

    scheduler_finetune = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_finetune, T_max=params["finetune_epochs"], eta_min=1e-7
    )
    patience_count = 0

    for epoch in range(1, params["finetune_epochs"] + 1):
        tr_loss, tr_acc   = run_epoch(model, train_loader, criterion,
                                      optimizer_finetune, device, is_train=True)
        val_loss, val_acc = run_epoch(model, val_loader,   criterion,
                                      None,                device, is_train=False)

        scheduler_finetune.step()
        cur_lr = optimizer_finetune.param_groups[0]["lr"]

        print(f"  [FineTune] Ep {epoch}/{params['finetune_epochs']} | "
              f"TrLoss:{tr_loss:.4f} TrAcc:{tr_acc:.4f} | "
              f"VLoss:{val_loss:.4f} VAcc:{val_acc:.4f} | "
              f"LR:{cur_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_weights   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
            print(f"    ✅ Best val_loss={best_val_loss:.4f}")
        else:
            patience_count += 1
            if patience_count >= patience_limit:
                print(f"    ⏹  Early stopping triggered at fine-tune epoch {epoch}")
                break

    # ── Restore best weights & save ───────────────────────────────────────────
    model.load_state_dict(best_weights)
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", MODEL_SAVE)
    torch.save(best_weights, save_path)
    print(f"\n  ✅ Best weights restored  (val_loss={best_val_loss:.4f})")
    print(f"  💾 Saved → {save_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TEST EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake",
]

def evaluate_on_test(model: nn.Module):
    """Evaluate the final model on the held-out test split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl_cfg = get_dataloader_config()
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, **dl_cfg)
    criterion   = nn.CrossEntropyLoss()

    model.eval()
    all_preds, all_labels = [], []
    test_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            test_loss   += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct     += (predicted == labels).sum().item()
            total       += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"\n{'='*60}")
    print("  TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Test Loss     : {test_loss / total:.4f}")
    print(f"  Test Accuracy : {correct / total:.4f}")
    print()
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Load dataset ─────────────────────────────────────────────────────────
    print("Loading blanchon/EuroSAT_RGB …")
    dataset = load_dataset("blanchon/EuroSAT_RGB")
    print(dataset)

    print("\nExtracting splits …")
    X_train, y_train = extract_images_labels(dataset["train"])
    X_val,   y_val   = extract_images_labels(dataset["validation"])
    X_test,  y_test  = extract_images_labels(dataset["test"])

    print(f"  Train : {X_train.shape}  Labels: {y_train.shape}")
    print(f"  Val   : {X_val.shape}    Labels: {y_val.shape}")
    print(f"  Test  : {X_test.shape}   Labels: {y_test.shape}")

    # Build Dataset objects (transform applied lazily per sample)
    train_ds = EuroSATDataset(X_train, y_train)
    val_ds   = EuroSATDataset(X_val,   y_val)
    test_ds  = EuroSATDataset(X_test,  y_test)
    
    

    # ── Optuna hyperparameter search ─────────────────────────────────────────
    print(f"\nStarting Optuna study  ({N_TRIALS} trials) …")
    study = optuna.create_study(
        direction  = "maximize",
        pruner     = optuna.pruners.MedianPruner(
            n_startup_trials = 2,
            n_warmup_steps   = 2,
        ),
        study_name = "eurosat_transfer_learning",
    )

    study.optimize(
        objective,
        n_trials       = N_TRIALS,
        timeout        = 7200,       # hard cap: 2 hours
        gc_after_trial = True,
    )

    # ── Print study summary ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("           OPTUNA STUDY RESULTS")
    print("="*60)
    print(f"  Total trials     : {len(study.trials)}")
    print(f"  Completed        : {sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)}")
    print(f"  Pruned           : {sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)}")
    print(f"  Best val_acc     : {study.best_value:.4f}")
    print(f"\n  Best Parameters  :")
    for k, v in study.best_params.items():
        print(f"    {k:<22} = {v}")
    print("="*60)

    # ── Retrain final model with best parameters ──────────────────────────────
    best_params = study.best_params
    # Inject fixed epoch counts into params dict
    best_params["warmup_epochs"]   = WARMUP_EPOCHS_DEFAULT
    best_params["finetune_epochs"] = FINETUNE_EPOCHS_DEFAULT

    print("\n🔁 Retraining final model with best parameters …")
    final_model = train_final_model(best_params)

    # ── Evaluate on test set ──────────────────────────────────────────────────
    evaluate_on_test(final_model)
