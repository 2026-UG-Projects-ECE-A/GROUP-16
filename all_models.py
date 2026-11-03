"""
UPPER LIMB REHABILITATION - ALL 5 MODELS IN ONE FILE
====================================================
 1. LSTM          (Bi-LSTM classifier + Seq2Seq trajectory)
 2. CNN           (1D-CNN classifier + trajectory)
 3. GAN           (Conditional GAN + classifier)
 4. YOLO          (pose extraction from video - template)
 5. LSTM + LLM    (LSTM prediction explained by an LLM)

Usage:
    python all_models.py lstm [epochs]
    python all_models.py cnn  [epochs]
    python all_models.py gan  [gan_epochs]
    python all_models.py yolo [video_path]
    python all_models.py lstm_llm
    python all_models.py all  [epochs]     # runs lstm, cnn, gan

Needs in the same folder: dataset.py (KiMOREDataset), models.py (for LSTM part).
Install: torch numpy matplotlib seaborn scikit-learn ultralytics opencv-python
         tensorflow openai
"""
import os
import sys
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset, random_split

# Project-specific modules (only needed for LSTM / CNN / GAN parts)
try:
    from dataset import KiMOREDataset
except ImportError:
    KiMOREDataset = None
try:
    from models import UpperLimbBiLSTMEvaluator, UpperLimbSeq2SeqLSTM
except ImportError:
    UpperLimbBiLSTMEvaluator = UpperLimbSeq2SeqLSTM = None

# Plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11


def sec(n, title):
    return "\n\n" + "#" * 70 + f"\n# PART {n}: {title}\n" + "#" * 70 + "\n\n"



######################################################################
# PART 1: LSTM
######################################################################

def train_and_evaluate_upperlimb(data_dir=None, epochs=15, batch_size=32, lr=1e-3):
    if data_dir is None:
        data_dir = os.path.join("kimore", "skeleton")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f"  Training Upper Limb LSTM Models on Device: {device}")
    print(f"==================================================")

    # 1. Load Upper Limb Classification Dataset
    print(f"\n[1/3] Loading Upper Limb Skeleton Dataset (classification mode)...")
    dataset_cls = KiMOREDataset(data_dir, seq_len=60, mode='classification', upperlimb_only=True)
    print(f"Loaded {len(dataset_cls)} upper-limb window samples (Features per frame: 45).")

    if len(dataset_cls) == 0:
        print("Error: No skeleton samples found.")
        return

    # Train / Test split
    train_size = int(0.8 * len(dataset_cls))
    test_size = len(dataset_cls) - train_size
    train_set_cls, test_set_cls = random_split(dataset_cls, [train_size, test_size])

    train_loader_cls = DataLoader(train_set_cls, batch_size=batch_size, shuffle=True)
    test_loader_cls = DataLoader(test_set_cls, batch_size=batch_size, shuffle=False)

    # Initialize Upper Limb Bi-LSTM Assessor
    model_cls = UpperLimbBiLSTMEvaluator(input_dim=45, hidden_dim=128, num_classes=2).to(device)
    criterion_cls = torch.nn.CrossEntropyLoss()
    optimizer_cls = torch.optim.Adam(model_cls.parameters(), lr=lr)

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    print(f"\n[2/3] Training Upper Limb Bi-LSTM Classifier ({epochs} Epochs)...")
    for epoch in range(epochs):
        model_cls.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, targets in train_loader_cls:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_cls.zero_grad()
            outputs = model_cls(inputs)
            loss = criterion_cls(outputs, targets)
            loss.backward()
            optimizer_cls.step()

            train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100.0

        # Validation phase
        model_cls.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, targets in test_loader_cls:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model_cls(inputs)
                loss = criterion_cls(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct_val += (preds == targets).sum().item()
                total_val += targets.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100.0

        train_loss_history.append(epoch_train_loss)
        val_loss_history.append(epoch_val_loss)
        train_acc_history.append(epoch_train_acc)
        val_acc_history.append(epoch_val_acc)

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

    # Save classification model
    torch.save(model_cls.state_dict(), "upperlimb_bilstm_evaluator.pth")
    print("Model saved to 'upperlimb_bilstm_evaluator.pth'.")

    # Final Test Set Evaluation & Confusion Matrix
    model_cls.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader_cls:
            inputs = inputs.to(device)
            outputs = model_cls(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    target_names = ["Healthy Upper Limb", "Impaired Upper Limb"]
    print("\n--- Upper Limb Classification Report ---")
    print(classification_report(all_targets, all_preds, target_names=target_names))

    # Compute Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)

    # 2. Also train Seq2Seq LSTM Simulator for Upper Limb Trajectory Prediction Line Graph
    print(f"\n[3/3] Training Upper Limb Seq2Seq Trajectory LSTM Simulator...")
    dataset_seq = KiMOREDataset(data_dir, seq_len=60, mode='simulation', upperlimb_only=True)
    train_size_seq = int(0.8 * len(dataset_seq))
    test_size_seq = len(dataset_seq) - train_size_seq
    train_set_seq, test_set_seq = random_split(dataset_seq, [train_size_seq, test_size_seq])

    train_loader_seq = DataLoader(train_set_seq, batch_size=batch_size, shuffle=True)
    test_loader_seq = DataLoader(test_set_seq, batch_size=batch_size, shuffle=False)

    model_seq = UpperLimbSeq2SeqLSTM(input_dim=45, hidden_dim=128, output_dim=45).to(device)
    optimizer_seq = torch.optim.Adam(model_seq.parameters(), lr=lr)
    criterion_seq = torch.nn.MSELoss()

    for epoch in range(10):
        model_seq.train()
        for inputs, targets in train_loader_seq:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_seq.zero_grad()
            outputs = model_seq(inputs)
            loss = criterion_seq(outputs, targets)
            loss.backward()
            optimizer_seq.step()

    torch.save(model_seq.state_dict(), "upperlimb_seq2seq_lstm.pth")
    print("Seq2Seq model saved to 'upperlimb_seq2seq_lstm.pth'.")

    # Extract sample trajectory for line graph plotting
    model_seq.eval()
    with torch.no_grad():
        for inputs, targets in test_loader_seq:
            inputs, targets = inputs.to(device), targets.to(device)
            predicted_seq = model_seq(inputs)
            sample_inp = inputs[0].cpu().numpy() # (59, 45)
            sample_tgt = targets[0].cpu().numpy() # (59, 45)
            sample_pred = predicted_seq[0].cpu().numpy() # (59, 45)
            break

    # ==========================================
    # GENERATE OUTPUT 1: CONFUSION MATRIX
    # ==========================================
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6.5))
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"

    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=True,
                xticklabels=target_names, yticklabels=target_names,
                annot_kws={"size": 14, "weight": "bold"}, ax=ax_cm)

    ax_cm.set_title("Confusion Matrix: Upper Limb Motor Assessment (Bi-LSTM)", fontsize=14, fontweight='bold', pad=15)
    ax_cm.set_xlabel("Predicted Upper Limb Condition", fontsize=12, labelpad=10)
    ax_cm.set_ylabel("True Upper Limb Condition", fontsize=12, labelpad=10)
    plt.tight_layout()
    cm_path = "upperlimb_confusion_matrix.png"
    fig_cm.savefig(cm_path, dpi=300)
    plt.close(fig_cm)
    print(f"\n[OUTPUT 1 SAVED] Upper Limb Confusion Matrix saved to '{cm_path}'.")

    # ==========================================
    # GENERATE OUTPUT 2: MULTI-PANEL LINE GRAPHS
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs_range = np.arange(1, epochs + 1)

    # Panel 1: Loss Curves over Epochs
    ax1 = axes[0, 0]
    ax1.plot(epochs_range, train_loss_history, 'o-', color='#1f77b4', linewidth=2.5, label='Training Loss')
    ax1.plot(epochs_range, val_loss_history, 's--', color='#ff7f0e', linewidth=2.5, label='Validation Loss')
    ax1.set_title("Panel A: Model Loss over Epochs", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Panel 2: Accuracy Curves over Epochs
    ax2 = axes[0, 1]
    ax2.plot(epochs_range, train_acc_history, 'o-', color='#2ca02c', linewidth=2.5, label='Training Accuracy')
    ax2.plot(epochs_range, val_acc_history, 's--', color='#d62728', linewidth=2.5, label='Validation Accuracy')
    ax2.set_title("Panel B: Classification Accuracy over Epochs", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_ylim(40, 105)
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Panel 3: Upper Limb Wrist Trajectory (Actual vs Predicted)
    time_steps = np.arange(len(sample_tgt))
    ax3 = axes[1, 0]
    ax3.plot(time_steps, sample_tgt[:, 12], '-', color='#1f77b4', linewidth=2.0, label='Actual Wrist X')
    ax3.plot(time_steps, sample_pred[:, 12], '--', color='#1f77b4', alpha=0.8, linewidth=2.0, label='Predicted Wrist X')
    ax3.plot(time_steps, sample_tgt[:, 13], '-', color='#2ca02c', linewidth=2.0, label='Actual Wrist Y')
    ax3.plot(time_steps, sample_pred[:, 13], '--', color='#2ca02c', alpha=0.8, linewidth=2.0, label='Predicted Wrist Y')
    ax3.set_title("Panel C: Upper Limb Wrist Trajectory (Actual vs. LSTM Predicted)", fontsize=13, fontweight='bold')
    ax3.set_xlabel("Time Frame (Sequence Step)", fontsize=11)
    ax3.set_ylabel("Normalized 3D Displacement (m)", fontsize=11)
    ax3.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    ax3.grid(True, linestyle=':', alpha=0.6)

    # Panel 4: Per-Joint Mean Absolute Error across 15 Upper Limb Joints
    joint_names = ['Neck', 'Head', 'ShoulderL', 'ElbowL', 'WristL', 'HandL',
                   'ShoulderR', 'ElbowR', 'WristR', 'HandR', 'SpineShoulder',
                   'HandTipL', 'ThumbL', 'HandTipR', 'ThumbR']

    mae_per_joint = np.mean(np.abs(sample_tgt - sample_pred).reshape(-1, 15, 3), axis=(0, 2))
    ax4 = axes[1, 1]
    bars = ax4.bar(joint_names, mae_per_joint, color='#9467bd', edgecolor='black', alpha=0.85)
    ax4.set_title("Panel D: Per-Joint Prediction MAE on Upper Limb", fontsize=13, fontweight='bold')
    ax4.set_xlabel("Upper Limb Joint", fontsize=11)
    ax4.set_ylabel("Mean Absolute Error (m)", fontsize=11)
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, linestyle=':', alpha=0.6, axis='y')

    plt.suptitle("Upper Limb Rehabilitation LSTM Model - Performance & Trajectory Analysis", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    graphs_path = "upperlimb_line_graphs.png"
    fig.savefig(graphs_path, dpi=300)
    plt.close(fig)
    print(f"[OUTPUT 2 SAVED] Upper Limb Line Graphs saved to '{graphs_path}'.")
    print(f"\n==================================================")
    print(f"  Upper Limb LSTM Training & Evaluation Complete!")
    print(f"==================================================")


######################################################################
# PART 2: CNN
######################################################################

nn = torch.nn


# ============================================================
# 1. CNN CLASSIFIER FOR UPPER LIMB
# ============================================================
class UpperLimbCNNClassifier(nn.Module):
    def __init__(self, input_dim=45, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # Input: (Batch, Sequence, Features)
        x = x.permute(0, 2, 1)
        x = self.features(x)
        return self.classifier(x)


# ============================================================
# 2. CNN TRAJECTORY PREDICTOR
# ============================================================
class UpperLimbCNNTrajectory(nn.Module):
    def __init__(self, input_dim=45, output_dim=45):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(128, output_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # Input: (Batch, Sequence, Features)
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x.permute(0, 2, 1)


# ============================================================
# 3. MAIN TRAINING FUNCTION
# ============================================================
def train_and_evaluate_upperlimb_cnn(data_dir=None, epochs=15, batch_size=32, lr=1e-3):
    if data_dir is None:
        data_dir = os.path.join("kimore", "skeleton")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("     UPPER LIMB CNN REHABILITATION MODEL")
    print("     Device:", device)
    print("=" * 60)

    # ---------------- STEP 1: LOAD DATA ----------------
    print("\n[1/3] Loading Upper Limb Skeleton Dataset...")
    dataset_cls = KiMOREDataset(data_dir, seq_len=60, mode="classification", upperlimb_only=True)
    print(f"Loaded {len(dataset_cls)} upper-limb window samples.")

    if len(dataset_cls) == 0:
        print("ERROR: No skeleton samples found.")
        return

    train_size = int(0.8 * len(dataset_cls))
    test_size = len(dataset_cls) - train_size
    train_set, test_set = random_split(dataset_cls, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = UpperLimbCNNClassifier(input_dim=45, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []

    # ---------------- STEP 2: TRAIN CNN ----------------
    print(f"\n[2/3] Training Upper Limb CNN ({epochs} Epochs)...")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            predictions = torch.argmax(outputs, dim=1)
            correct_train += (predictions == targets).sum().item()
            total_train += targets.size(0)

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100

        # Validation
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                predictions = torch.argmax(outputs, dim=1)
                correct_val += (predictions == targets).sum().item()
                total_val += targets.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100

        train_loss_history.append(epoch_train_loss)
        val_loss_history.append(epoch_val_loss)
        train_acc_history.append(epoch_train_acc)
        val_acc_history.append(epoch_val_acc)

        print(
            f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Train Acc: {epoch_train_acc:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {epoch_val_acc:.2f}%"
        )

    torch.save(model.state_dict(), "upperlimb_cnn_classifier.pth")
    print("\nCNN model saved to 'upperlimb_cnn_classifier.pth'")

    # ---------------- FINAL TEST EVALUATION ----------------
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            preds = torch.argmax(model(inputs), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    target_names = ["Healthy Upper Limb", "Impaired Upper Limb"]

    print("\n--- Upper Limb CNN Classification Report ---")
    print(classification_report(all_targets, all_preds, target_names=target_names))

    # ---------------- CONFUSION MATRIX ----------------
    cm = confusion_matrix(all_targets, all_preds)
    cm_percent = (cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]) * 100

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i,j]}\n({cm_percent[i,j]:.1f}%)"

    fig_cm, ax_cm = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=True,
                xticklabels=target_names, yticklabels=target_names,
                annot_kws={"size": 14, "weight": "bold"}, ax=ax_cm)
    ax_cm.set_title("Confusion Matrix: Upper Limb Motor Assessment (CNN)", fontsize=14, fontweight="bold")
    ax_cm.set_xlabel("Predicted Upper Limb Condition")
    ax_cm.set_ylabel("True Upper Limb Condition")
    plt.tight_layout()

    cm_path = "upperlimb_cnn_confusion_matrix.png"
    fig_cm.savefig(cm_path, dpi=300)
    plt.close(fig_cm)
    print(f"\n[OUTPUT 1 SAVED] {cm_path}")

    # ---------------- STEP 3: CNN TRAJECTORY MODEL ----------------
    print("\n[3/3] Training Upper Limb CNN Trajectory Predictor...")

    dataset_seq = KiMOREDataset(data_dir, seq_len=60, mode="simulation", upperlimb_only=True)

    if len(dataset_seq) == 0:
        print("No simulation samples found.")
        return

    train_size_seq = int(0.8 * len(dataset_seq))
    test_size_seq = len(dataset_seq) - train_size_seq
    train_set_seq, test_set_seq = random_split(dataset_seq, [train_size_seq, test_size_seq])

    train_loader_seq = DataLoader(train_set_seq, batch_size=batch_size, shuffle=True)
    test_loader_seq = DataLoader(test_set_seq, batch_size=batch_size, shuffle=False)

    model_seq = UpperLimbCNNTrajectory(input_dim=45, output_dim=45).to(device)
    optimizer_seq = torch.optim.Adam(model_seq.parameters(), lr=lr)
    criterion_seq = nn.MSELoss()

    trajectory_loss_history = []
    trajectory_epochs = 10

    for epoch in range(trajectory_epochs):
        model_seq.train()
        total_loss, total_samples = 0.0, 0

        for inputs, targets in train_loader_seq:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_seq.zero_grad()
            outputs = model_seq(inputs)
            loss = criterion_seq(outputs, targets)
            loss.backward()
            optimizer_seq.step()

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

        epoch_loss = total_loss / total_samples
        trajectory_loss_history.append(epoch_loss)
        print(f"Trajectory Epoch [{epoch+1:02d}/{trajectory_epochs:02d}] Loss: {epoch_loss:.6f}")

    torch.save(model_seq.state_dict(), "upperlimb_cnn_trajectory.pth")
    print("CNN trajectory model saved to 'upperlimb_cnn_trajectory.pth'")

    # Sample prediction
    model_seq.eval()
    with torch.no_grad():
        for inputs, targets in test_loader_seq:
            inputs, targets = inputs.to(device), targets.to(device)
            predicted_seq = model_seq(inputs)
            sample_inp = inputs[0].cpu().numpy()
            sample_tgt = targets[0].cpu().numpy()
            sample_pred = predicted_seq[0].cpu().numpy()
            break

    # ---------------- PERFORMANCE GRAPHS ----------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs_range = np.arange(1, epochs + 1)

    # Panel A: Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs_range, train_loss_history, "o-", linewidth=2.5, label="Training Loss")
    ax1.plot(epochs_range, val_loss_history, "s--", linewidth=2.5, label="Validation Loss")
    ax1.set_title("Panel A: CNN Loss over Epochs", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.legend()
    ax1.grid(True, linestyle=":")

    # Panel B: Accuracy
    ax2 = axes[0, 1]
    ax2.plot(epochs_range, train_acc_history, "o-", linewidth=2.5, label="Training Accuracy")
    ax2.plot(epochs_range, val_acc_history, "s--", linewidth=2.5, label="Validation Accuracy")
    ax2.set_title("Panel B: CNN Classification Accuracy", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(40, 105)
    ax2.legend()
    ax2.grid(True, linestyle=":")

    # Panel C: Wrist trajectory
    time_steps = np.arange(len(sample_tgt))
    ax3 = axes[1, 0]
    ax3.plot(time_steps, sample_tgt[:, 12], "-", linewidth=2, label="Actual Wrist X")
    ax3.plot(time_steps, sample_pred[:, 12], "--", linewidth=2, label="Predicted Wrist X")
    ax3.plot(time_steps, sample_tgt[:, 13], "-", linewidth=2, label="Actual Wrist Y")
    ax3.plot(time_steps, sample_pred[:, 13], "--", linewidth=2, label="Predicted Wrist Y")
    ax3.set_title("Panel C: Upper Limb Wrist Trajectory (CNN)", fontweight="bold")
    ax3.set_xlabel("Time Frame")
    ax3.set_ylabel("Normalized 3D Displacement")
    ax3.legend(fontsize=9)
    ax3.grid(True, linestyle=":")

    # Panel D: Per-joint MAE
    joint_names = ["Neck", "Head", "ShoulderL", "ElbowL", "WristL", "HandL",
                   "ShoulderR", "ElbowR", "WristR", "HandR", "SpineShoulder",
                   "HandTipL", "ThumbL", "HandTipR", "ThumbR"]

    mae_per_joint = np.mean(np.abs(sample_tgt - sample_pred).reshape(-1, 15, 3), axis=(0, 2))

    ax4 = axes[1, 1]
    ax4.bar(joint_names, mae_per_joint, edgecolor="black", alpha=0.85)
    ax4.set_title("Panel D: Per-Joint Prediction MAE using CNN", fontweight="bold")
    ax4.set_xlabel("Upper Limb Joint")
    ax4.set_ylabel("Mean Absolute Error")
    ax4.tick_params(axis="x", rotation=45)
    ax4.grid(True, linestyle=":", axis="y")

    plt.suptitle("Upper Limb Rehabilitation CNN Model - Performance & Trajectory Analysis",
                 fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    graphs_path = "upperlimb_cnn_line_graphs.png"
    fig.savefig(graphs_path, dpi=300)
    plt.close(fig)
    print(f"\n[OUTPUT 2 SAVED] {graphs_path}")

    print("\n" + "=" * 60)
    print(" Upper Limb CNN Training & Evaluation Complete!")
    print("=" * 60)


######################################################################
# PART 3: GAN
######################################################################

# ============================================================
# CONDITIONAL GAN GENERATOR
# ============================================================
class UpperLimbGANGenerator(torch.nn.Module):

    def __init__(self, input_dim=45, hidden_dim=128, num_classes=2, seq_len=60):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim

        self.label_embedding = torch.nn.Embedding(num_classes, hidden_dim)
        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)

        self.gru = torch.nn.GRU(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, labels):
        # x shape: (batch, sequence, 45)  -> noise or seed sequence
        x_features = self.input_layer(x)
        label_features = self.label_embedding(labels).unsqueeze(1)
        label_features = label_features.repeat(1, x.size(1), 1)

        combined = torch.cat([x_features, label_features], dim=2)
        hidden_seq, _ = self.gru(combined)
        return self.output_layer(hidden_seq)


# ============================================================
# CONDITIONAL GAN DISCRIMINATOR
# ============================================================
class UpperLimbGANDiscriminator(torch.nn.Module):

    def __init__(self, input_dim=45, hidden_dim=128, num_classes=2):
        super().__init__()

        self.label_embedding = torch.nn.Embedding(num_classes, hidden_dim)
        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)

        self.gru = torch.nn.GRU(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x, labels):
        x_features = self.input_layer(x)
        label_features = self.label_embedding(labels).unsqueeze(1)
        label_features = label_features.repeat(1, x.size(1), 1)

        combined = torch.cat([x_features, label_features], dim=2)
        hidden_seq, _ = self.gru(combined)

        last_feature = hidden_seq[:, -1, :]   # last temporal feature
        return self.classifier(last_feature)  # raw logit (use BCEWithLogitsLoss)


# ============================================================
# GAN CLASSIFIER  (completed - original code was cut off here)
# ============================================================
class UpperLimbGANClassifier(torch.nn.Module):
    """Classifier trained on real + GAN-generated sequences.
    Input (batch, seq, 45) -> mean + std over time -> MLP -> class logits."""

    def __init__(self, input_dim=45, hidden_dim=128, num_classes=2):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim * 2, hidden_dim),
            torch.nn.ReLU(),

            torch.nn.Linear(hidden_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x):
        feats = torch.cat([x.mean(dim=1), x.std(dim=1)], dim=1)
        return self.network(feats)


# ============================================================
# HELPERS
# ============================================================
def evaluate(model, loader, device):
    model.eval()
    preds_all, targets_all = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            out = model(inputs.to(device))
            preds_all.extend(torch.argmax(out, 1).cpu().numpy())
            targets_all.extend(targets.numpy())
    return np.array(preds_all), np.array(targets_all)


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================
def train_and_evaluate_upperlimb_gan(data_dir=None, gan_epochs=30, clf_epochs=15,
                                     batch_size=32, lr=2e-4):
    if data_dir is None:
        data_dir = os.path.join("kimore", "skeleton")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("   UPPER LIMB CONDITIONAL GAN MODEL  | Device:", device)
    print("=" * 60)

    # ---------------- 1. LOAD DATA ----------------
    print("\n[1/4] Loading dataset...")
    dataset = KiMOREDataset(data_dir, seq_len=60, mode="classification", upperlimb_only=True)
    print(f"Loaded {len(dataset)} samples.")
    if len(dataset) == 0:
        print("ERROR: No skeleton samples found.")
        return

    train_size = int(0.8 * len(dataset))
    train_set, test_set = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # ---------------- 2. TRAIN cGAN ----------------
    print(f"\n[2/4] Training Conditional GAN ({gan_epochs} epochs)...")
    G = UpperLimbGANGenerator().to(device)
    D = UpperLimbGANDiscriminator().to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = torch.nn.BCEWithLogitsLoss()

    g_hist, d_hist = [], []
    for epoch in range(gan_epochs):
        g_tot, d_tot, n = 0.0, 0.0, 0
        for real, labels in train_loader:
            real, labels = real.to(device), labels.to(device)
            b = real.size(0)
            real_t = torch.ones(b, 1, device=device)
            fake_t = torch.zeros(b, 1, device=device)

            # --- Discriminator ---
            noise = torch.randn_like(real)
            fake = G(noise, labels).detach()
            loss_D = bce(D(real, labels), real_t) + bce(D(fake, labels), fake_t)
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # --- Generator ---
            noise = torch.randn_like(real)
            fake = G(noise, labels)
            loss_G = bce(D(fake, labels), real_t)
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            g_tot += loss_G.item() * b
            d_tot += loss_D.item() * b
            n += b

        g_hist.append(g_tot / n)
        d_hist.append(d_tot / n)
        print(f"GAN Epoch [{epoch+1:02d}/{gan_epochs:02d}] | G Loss: {g_hist[-1]:.4f} | D Loss: {d_hist[-1]:.4f}")

    torch.save(G.state_dict(), "upperlimb_gan_generator.pth")
    torch.save(D.state_dict(), "upperlimb_gan_discriminator.pth")
    print("GAN models saved.")

    # ---------------- 3. AUGMENT + TRAIN CLASSIFIER ----------------
    print("\n[3/4] Generating synthetic data & training classifier...")
    G.eval()
    real_x = torch.stack([train_set[i][0] for i in range(len(train_set))])
    real_y = torch.tensor([int(train_set[i][1]) for i in range(len(train_set))])

    with torch.no_grad():
        fake_x = G(torch.randn_like(real_x).to(device), real_y.to(device)).cpu()

    aug_set = TensorDataset(torch.cat([real_x, fake_x]), torch.cat([real_y, real_y]))
    aug_loader = DataLoader(aug_set, batch_size=batch_size, shuffle=True)

    clf = UpperLimbGANClassifier().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    opt_c = torch.optim.Adam(clf.parameters(), lr=1e-3)

    tr_loss_h, va_loss_h, tr_acc_h, va_acc_h = [], [], [], []
    for epoch in range(clf_epochs):
        clf.train()
        tl, tc, tn = 0.0, 0, 0
        for x, y in aug_loader:
            x, y = x.to(device), y.to(device)
            opt_c.zero_grad()
            out = clf(x)
            loss = criterion(out, y)
            loss.backward()
            opt_c.step()
            tl += loss.item() * x.size(0)
            tc += (out.argmax(1) == y).sum().item()
            tn += y.size(0)

        clf.eval()
        vl, vc, vn = 0.0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = clf(x)
                vl += criterion(out, y).item() * x.size(0)
                vc += (out.argmax(1) == y).sum().item()
                vn += y.size(0)

        tr_loss_h.append(tl / tn); va_loss_h.append(vl / vn)
        tr_acc_h.append(tc / tn * 100); va_acc_h.append(vc / vn * 100)
        print(f"Clf Epoch [{epoch+1:02d}/{clf_epochs:02d}] | Train Loss: {tr_loss_h[-1]:.4f} | "
              f"Train Acc: {tr_acc_h[-1]:.2f}% | Val Loss: {va_loss_h[-1]:.4f} | Val Acc: {va_acc_h[-1]:.2f}%")

    torch.save(clf.state_dict(), "upperlimb_gan_classifier.pth")

    # ---------------- 4. EVALUATION & PLOTS ----------------
    print("\n[4/4] Evaluating & saving plots...")
    preds, targets = evaluate(clf, test_loader, device)
    target_names = ["Healthy Upper Limb", "Impaired Upper Limb"]
    print("\n--- Upper Limb GAN Classification Report ---")
    print(classification_report(targets, preds, target_names=target_names))

    cm = confusion_matrix(targets, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"

    fig_cm, ax_cm = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", xticklabels=target_names,
                yticklabels=target_names, annot_kws={"size": 14, "weight": "bold"}, ax=ax_cm)
    ax_cm.set_title("Confusion Matrix: Upper Limb Assessment (GAN-augmented)", fontsize=14, fontweight="bold")
    ax_cm.set_xlabel("Predicted Upper Limb Condition")
    ax_cm.set_ylabel("True Upper Limb Condition")
    plt.tight_layout()
    fig_cm.savefig("upperlimb_gan_confusion_matrix.png", dpi=300)
    plt.close(fig_cm)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ge = np.arange(1, gan_epochs + 1)
    ce = np.arange(1, clf_epochs + 1)

    axes[0, 0].plot(ge, g_hist, "o-", label="Generator Loss")
    axes[0, 0].plot(ge, d_hist, "s--", label="Discriminator Loss")
    axes[0, 0].set_title("Panel A: GAN Losses", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("BCE Loss"); axes[0, 0].legend()

    axes[0, 1].plot(ce, tr_loss_h, "o-", label="Train Loss")
    axes[0, 1].plot(ce, va_loss_h, "s--", label="Val Loss")
    axes[0, 1].set_title("Panel B: Classifier Loss", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Cross Entropy"); axes[0, 1].legend()

    axes[1, 0].plot(ce, tr_acc_h, "o-", label="Train Acc")
    axes[1, 0].plot(ce, va_acc_h, "s--", label="Val Acc")
    axes[1, 0].set_title("Panel C: Classifier Accuracy", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Accuracy (%)"); axes[1, 0].legend()

    # Panel D: real vs generated wrist trajectory
    sample_real = real_x[0].numpy()
    sample_fake = fake_x[0].numpy()
    t = np.arange(sample_real.shape[0])
    axes[1, 1].plot(t, sample_real[:, 12], "-", label="Real Wrist X")
    axes[1, 1].plot(t, sample_fake[:, 12], "--", label="Generated Wrist X")
    axes[1, 1].set_title("Panel D: Real vs GAN-Generated Wrist Trajectory", fontweight="bold")
    axes[1, 1].set_xlabel("Time Frame"); axes[1, 1].set_ylabel("Normalized Displacement"); axes[1, 1].legend()

    for ax in axes.flat:
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Upper Limb Rehabilitation GAN Model - Performance Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("upperlimb_gan_line_graphs.png", dpi=300)
    plt.close(fig)

    print("\nSaved: upperlimb_gan_confusion_matrix.png, upperlimb_gan_line_graphs.png")
    print("=" * 60)
    print(" Upper Limb GAN Training & Evaluation Complete!")
    print("=" * 60)


######################################################################
# PART 4: YOLO (template)
######################################################################

# COCO keypoint indices: 5/6 shoulders, 7/8 elbows, 9/10 wrists
UPPER_LIMB_IDS = [5, 6, 7, 8, 9, 10]


def extract_upperlimb_sequence(video_path, model_path="yolov8n-pose.pt",
                               seq_len=60, feat_dim=45, show=False):
    import cv2
    from ultralytics import YOLO
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        result = model(frame, verbose=False)[0]
        if result.keypoints is None or len(result.keypoints.xyn) == 0:
            continue

        kpts = result.keypoints.xyn[0].cpu().numpy()          # (17, 2) normalized x,y
        upper = kpts[UPPER_LIMB_IDS].flatten()                # (12,)
        vec = np.zeros(feat_dim, dtype=np.float32)            # pad to 45 features
        vec[:upper.size] = upper
        frames.append(vec)

        if show:
            cv2.imshow("YOLO Pose", result.plot())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    if not frames:
        raise RuntimeError("No person detected in video.")

    seq = np.array(frames, dtype=np.float32)
    # Pad / crop to fixed length
    if len(seq) >= seq_len:
        seq = seq[:seq_len]
    else:
        seq = np.vstack([seq, np.zeros((seq_len - len(seq), feat_dim), dtype=np.float32)])
    return seq


######################################################################
# PART 5: LSTM + LLM
######################################################################

CLASS_NAMES = ["Correct movement", "Incorrect movement"]
_lstm_model = None
_openai_client = None


def lstm_llm_prediction(sequence):
    global _lstm_model, _openai_client
    import json
    from tensorflow.keras.models import load_model
    from openai import OpenAI

    if _lstm_model is None:
        _lstm_model = load_model("lstm_model.keras")
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    sequence = np.asarray(sequence, dtype=np.float32)
    model_input = np.expand_dims(sequence, axis=0)

    # LSTM prediction
    probabilities = _lstm_model.predict(model_input, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    results = {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": round(float(probabilities[predicted_index]), 3)
    }

    # LLM explanation
    response = _openai_client.responses.create(
        model="gpt-5.5",
        input=f"""
        Explain this LSTM prediction for a stroke
        rehabilitation monitoring system in simple English.

        Prediction: {json.dumps(results)}

        Explain the predicted movement and confidence.
        Confidence is not a measure of patient recovery.
        Do not diagnose or prescribe treatment.
        Do not invent measurements.
        """
    )

    return results, response.output_text


# ======================================================================
# MAIN MENU
# ======================================================================
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    choice = sys.argv[1].lower()
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    n = int(arg) if arg and arg.isdigit() else None

    if choice == "lstm":
        train_and_evaluate_upperlimb(epochs=n or 15)
    elif choice == "cnn":
        train_and_evaluate_upperlimb_cnn(epochs=n or 15)
    elif choice == "gan":
        train_and_evaluate_upperlimb_gan(gan_epochs=n or 30)
    elif choice == "yolo":
        video = arg or "input_video.mp4"
        seq = extract_upperlimb_sequence(video)
        np.save("movement_sequence.npy", seq)
        print("Saved movement_sequence.npy with shape", seq.shape)
    elif choice == "lstm_llm":
        seq = np.load("movement_sequence.npy")
        prediction, explanation = lstm_llm_prediction(seq)
        print("LSTM Prediction:", prediction)
        print("\nLLM Explanation:", explanation)
    elif choice == "all":
        train_and_evaluate_upperlimb(epochs=n or 15)
        train_and_evaluate_upperlimb_cnn(epochs=n or 15)
        train_and_evaluate_upperlimb_gan(gan_epochs=n or 30)
    else:
        print("Unknown option:", choice)
        print(__doc__)


if __name__ == "__main__":
    main()
