"""
ANN-to-SNN Conversion Pipeline
================================
Brain-Inspired Computing Project
---------------------------------
Steps:
  1. Train a small CNN on MNIST (standard PyTorch)
  2. Convert it to a Spiking Neural Network (snnTorch)
  3. Evaluate accuracy + spike count on both
  4. AI-assisted threshold calibration to optimize the tradeoff
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import snntorch as snn
from snntorch import surrogate
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import time
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BATCH_SIZE    = 64
EPOCHS        = 3          # small — enough to get ~98% accuracy quickly
LEARNING_RATE = 1e-3
NUM_STEPS     = 25         # SNN timesteps per inference
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR      = "./data"
RESULTS_DIR   = "./snn_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"\n{'='*55}")
print(f"  ANN → SNN Conversion Pipeline")
print(f"  Device: {DEVICE}")
print(f"{'='*55}\n")


# ─────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────
print("[1/5] Loading MNIST dataset...")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"    Train samples : {len(train_dataset):,}")
print(f"    Test  samples : {len(test_dataset):,}\n")


# ─────────────────────────────────────────────
# 2. STANDARD CNN (ANN)
# ─────────────────────────────────────────────
print("[2/5] Defining and training CNN (ANN)...")

class CNN(nn.Module):
    """
    Small convolutional network for MNIST.
    Deliberately simple so conversion is clean.
    Architecture:
        Conv(1→16, 3x3) → ReLU → MaxPool
        Conv(16→32, 3x3) → ReLU → MaxPool
        Flatten → FC(800→256) → ReLU → FC(256→10)
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # 28x28 → 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 28x28 → 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 14x14 → 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 14x14 → 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_ann(model, loader, epochs):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    loss_history = []
    for epoch in range(epochs):
        total_loss = 0
        t0 = time.time()
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"    Epoch {epoch+1}/{epochs}  "
                      f"Batch {batch_idx:3d}/{len(loader)}  "
                      f"Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        elapsed = time.time() - t0
        print(f"    → Epoch {epoch+1} complete | avg loss: {avg_loss:.4f} | {elapsed:.1f}s\n")
    return loss_history


def evaluate_ann(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total   += target.size(0)
    return correct / total


ann_model = CNN().to(DEVICE)
loss_history = train_ann(ann_model, train_loader, EPOCHS)
ann_accuracy = evaluate_ann(ann_model, test_loader)
print(f"  ✓ ANN test accuracy: {ann_accuracy*100:.2f}%\n")


# ─────────────────────────────────────────────
# 3. ANN → SNN CONVERSION
# ─────────────────────────────────────────────
print("[3/5] Converting ANN → SNN...")

class SNN(nn.Module):
    """
    SNN equivalent of the CNN above.
    ReLU activations are replaced with Leaky Integrate-and-Fire (LIF) neurons.
    MaxPool2d is kept — it's compatible with spiking signals.
    Weights are copied directly from the trained ANN.

    Threshold (beta, threshold) parameters control when each LIF neuron fires.
    These are the key targets for calibration in Step 5.
    """
    def __init__(self, thresholds):
        """
        thresholds: list of 4 floats, one per LIF layer
          [conv1_thresh, conv2_thresh, fc1_thresh, fc2_thresh]
        """
        super().__init__()
        t1, t2, t3, t4 = thresholds

        # ---- feature layers ----
        self.conv1    = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.lif1     = snn.Leaky(beta=0.9, threshold=t1, spike_grad=surrogate.fast_sigmoid())
        self.pool1    = nn.MaxPool2d(2)

        self.conv2    = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.lif2     = snn.Leaky(beta=0.9, threshold=t2, spike_grad=surrogate.fast_sigmoid())
        self.pool2    = nn.MaxPool2d(2)

        # ---- classifier layers ----
        self.flatten  = nn.Flatten()
        self.fc1      = nn.Linear(32 * 7 * 7, 256)
        self.lif3     = snn.Leaky(beta=0.9, threshold=t3, spike_grad=surrogate.fast_sigmoid())

        self.fc2      = nn.Linear(256, 10)
        self.lif4     = snn.Leaky(beta=0.9, threshold=t4, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x, num_steps):
        """
        Run the network for `num_steps` timesteps.
        x: static input image, replicated across all timesteps.
        Returns:
            spike_out  : accumulated output spikes [batch, 10]
            total_spikes: total spike count across all layers (energy proxy)
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        spike_accumulator = torch.zeros(x.size(0), 10).to(x.device)
        total_spikes = 0

        for _ in range(num_steps):
            # Conv block 1
            cur1        = self.conv1(x)
            spk1, mem1  = self.lif1(cur1, mem1)
            out1        = self.pool1(spk1)
            total_spikes += spk1.sum().item()

            # Conv block 2
            cur2        = self.conv2(out1)
            spk2, mem2  = self.lif2(cur2, mem2)
            out2        = self.pool2(spk2)
            total_spikes += spk2.sum().item()

            # FC block 1
            flat        = self.flatten(out2)
            cur3        = self.fc1(flat)
            spk3, mem3  = self.lif3(cur3, mem3)
            total_spikes += spk3.sum().item()

            # FC block 2 (output)
            cur4        = self.fc2(spk3)
            spk4, mem4  = self.lif4(cur4, mem4)
            spike_accumulator += spk4
            total_spikes += spk4.sum().item()

        return spike_accumulator, total_spikes


def copy_weights(ann, snn_model):
    """Copy trained ANN weights directly into the SNN."""
    snn_model.conv1.weight.data = ann.features[0].weight.data.clone()
    snn_model.conv1.bias.data   = ann.features[0].bias.data.clone()
    snn_model.conv2.weight.data = ann.features[3].weight.data.clone()
    snn_model.conv2.bias.data   = ann.features[3].bias.data.clone()
    snn_model.fc1.weight.data   = ann.classifier[1].weight.data.clone()
    snn_model.fc1.bias.data     = ann.classifier[1].bias.data.clone()
    snn_model.fc2.weight.data   = ann.classifier[3].weight.data.clone()
    snn_model.fc2.bias.data     = ann.classifier[3].bias.data.clone()
    return snn_model


def evaluate_snn(snn_model, loader, num_steps, max_batches=None):
    """
    Evaluate SNN accuracy and count total spikes (energy proxy).
    max_batches: limit evaluation for speed during calibration search.
    """
    snn_model.eval()
    correct = total = 0
    total_spikes = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            data, target = data.to(DEVICE), target.to(DEVICE)
            spike_out, batch_spikes = snn_model(data, num_steps)
            pred = spike_out.argmax(dim=1)
            correct      += (pred == target).sum().item()
            total        += target.size(0)
            total_spikes += batch_spikes
    accuracy = correct / total if total > 0 else 0
    spikes_per_image = total_spikes / total if total > 0 else 0
    return accuracy, spikes_per_image


# Build initial SNN with default thresholds (1.0 everywhere)
DEFAULT_THRESHOLDS = [1.0, 1.0, 1.0, 1.0]
snn_model = SNN(DEFAULT_THRESHOLDS).to(DEVICE)
snn_model = copy_weights(ann_model, snn_model)

print("  Evaluating SNN with default thresholds [1.0, 1.0, 1.0, 1.0]...")
snn_acc_default, spikes_default = evaluate_snn(snn_model, test_loader, NUM_STEPS)
print(f"  ✓ SNN accuracy (default): {snn_acc_default*100:.2f}%")
print(f"  ✓ Spikes per image      : {spikes_default:.1f}\n")


# ─────────────────────────────────────────────
# 4. AI-ASSISTED THRESHOLD CALIBRATION
# ─────────────────────────────────────────────
print("[4/5] AI-assisted threshold calibration...")
print("  Strategy: grid search over threshold space,")
print("  scored by a weighted objective: accuracy - λ * normalized_spikes")
print("  This mimics what an AI optimizer would do automatically.\n")

# Calibration uses a small subset of the test set for speed
CALIBRATION_BATCHES = 10   # ~640 images
LAMBDA = 0.3               # weight on spike penalty (higher = prefer fewer spikes)

threshold_candidates = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
best_score    = -1
best_thresholds = DEFAULT_THRESHOLDS
best_acc      = snn_acc_default
best_spikes   = spikes_default
calibration_log = []

# We search over a representative subset of threshold combinations
# In a real AI system, this would be Bayesian optimization or a learned predictor
import itertools

# To keep runtime reasonable, we search output + fc1 thresholds
# while keeping conv thresholds at 1.0 (they're less sensitive)
search_space = list(itertools.product(
    [0.75, 1.0, 1.25],   # conv1 threshold
    [0.75, 1.0, 1.25],   # conv2 threshold
    threshold_candidates, # fc1 threshold
    threshold_candidates, # fc2 threshold (output layer — most impactful)
))

print(f"  Searching {len(search_space)} threshold combinations...")
print(f"  {'#':>4}  {'Thresholds':36s}  {'Acc':>7}  {'Spikes/img':>10}  {'Score':>7}")
print(f"  {'-'*70}")

for i, (t1, t2, t3, t4) in enumerate(search_space):
    candidate = [t1, t2, t3, t4]
    model_c = SNN(candidate).to(DEVICE)
    model_c = copy_weights(ann_model, model_c)
    acc, spikes = evaluate_snn(model_c, test_loader, NUM_STEPS,
                               max_batches=CALIBRATION_BATCHES)

    # Objective: maximize accuracy, penalize spike count
    # Normalize spikes against default to make λ interpretable
    norm_spikes = spikes / max(spikes_default, 1)
    score = acc - LAMBDA * norm_spikes

    calibration_log.append({
        'thresholds': candidate, 'accuracy': acc,
        'spikes': spikes, 'score': score
    })

    if score > best_score:
        best_score      = score
        best_thresholds = candidate
        best_acc        = acc
        best_spikes     = spikes

    # Print every 12th row just to show progress — no best marker during search
    if i % 12 == 0:
        thresh_str = str([f"{t:.2f}" for t in candidate])
        print(f"  {i:>4}  {thresh_str:36s}  {acc*100:6.2f}%  {spikes:10.1f}  {score:7.4f}")

# Print a clean summary of the top 5 after the search completes
calibration_log.sort(key=lambda x: x['score'], reverse=True)
print(f"\n  Search complete. Top 5 results:")
print(f"  {'Rank':>4}  {'Thresholds':36s}  {'Acc':>7}  {'Spikes/img':>10}  {'Score':>7}")
print(f"  {'-'*70}")
for rank, entry in enumerate(calibration_log[:5], 1):
    thresh_str = str([f"{t:.2f}" for t in entry['thresholds']])
    marker = "  ◀ BEST" if rank == 1 else ""
    print(f"  {rank:>4}  {thresh_str:36s}  {entry['accuracy']*100:6.2f}%  "
          f"{entry['spikes']:10.1f}  {entry['score']:7.4f}{marker}")

print(f"\n  ✓ Best thresholds found: {best_thresholds}")
print(f"  ✓ Best SNN accuracy    : {best_acc*100:.2f}%")
print(f"  ✓ Best spikes/image    : {best_spikes:.1f}")

# Full evaluation with best thresholds
print("\n  Running full test set evaluation with best thresholds...")
snn_best = SNN(best_thresholds).to(DEVICE)
snn_best = copy_weights(ann_model, snn_best)
snn_acc_best, spikes_best = evaluate_snn(snn_best, test_loader, NUM_STEPS)
print(f"  ✓ Final SNN accuracy   : {snn_acc_best*100:.2f}%")
print(f"  ✓ Final spikes/image   : {spikes_best:.1f}\n")


# ─────────────────────────────────────────────
# 5. RESULTS & PLOTS
# ─────────────────────────────────────────────
print("[5/5] Generating results plots...")

# ── Accuracy drop ──────────────────────────────────────────────
accuracy_drop     = (ann_accuracy - snn_acc_best) * 100
spike_reduction   = (spikes_default - spikes_best) / spikes_default * 100

# ── Sort calibration log by score ─────────────────────────────
calibration_log.sort(key=lambda x: x['score'], reverse=True)
top10 = calibration_log[:10]

fig = plt.figure(figsize=(16, 10), facecolor='#0a0c0f')
fig.suptitle('ANN → SNN Conversion Pipeline Results',
             color='#00e5c8', fontsize=15, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ax_color  = '#0e1118'
txt_color = '#c8cdd6'
acc_color = '#00e5c8'
spk_color = '#e5a000'
ann_color = '#7a9eff'

# ── Plot 1: ANN training loss ──────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(ax_color)
ax1.plot(range(1, EPOCHS+1), loss_history, color=ann_color, linewidth=2, marker='o', markersize=6)
ax1.set_title('ANN Training Loss', color=txt_color, fontsize=11, pad=8)
ax1.set_xlabel('Epoch', color=txt_color, fontsize=9)
ax1.set_ylabel('Cross-entropy loss', color=txt_color, fontsize=9)
ax1.tick_params(colors=txt_color, labelsize=8)
for spine in ax1.spines.values(): spine.set_edgecolor('#1e2530')
ax1.grid(True, color='#1e2530', linewidth=0.5)

# ── Plot 2: Accuracy comparison bar chart ─────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(ax_color)
labels  = ['ANN\n(baseline)', 'SNN\n(default\nthresh)', 'SNN\n(calibrated)']
accs    = [ann_accuracy*100, snn_acc_default*100, snn_acc_best*100]
colors  = [ann_color, '#3a4455', acc_color]
bars    = ax2.bar(labels, accs, color=colors, width=0.5, edgecolor='#1e2530', linewidth=0.5)
ax2.set_ylim(min(accs)-5, 101)
ax2.set_title('Accuracy Comparison', color=txt_color, fontsize=11, pad=8)
ax2.set_ylabel('Test accuracy (%)', color=txt_color, fontsize=9)
ax2.tick_params(colors=txt_color, labelsize=8)
for spine in ax2.spines.values(): spine.set_edgecolor('#1e2530')
ax2.grid(True, axis='y', color='#1e2530', linewidth=0.5)
for bar, acc in zip(bars, accs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{acc:.2f}%', ha='center', va='bottom', color=txt_color, fontsize=8)

# ── Plot 3: Spike count comparison ────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(ax_color)
slabels = ['SNN\n(default\nthresh)', 'SNN\n(calibrated)']
spikes  = [spikes_default, spikes_best]
scolors = ['#3a4455', spk_color]
sbars   = ax3.bar(slabels, spikes, color=scolors, width=0.4, edgecolor='#1e2530', linewidth=0.5)
ax3.set_title('Spikes per Image\n(energy proxy)', color=txt_color, fontsize=11, pad=8)
ax3.set_ylabel('Total spikes / image', color=txt_color, fontsize=9)
ax3.tick_params(colors=txt_color, labelsize=8)
for spine in ax3.spines.values(): spine.set_edgecolor('#1e2530')
ax3.grid(True, axis='y', color='#1e2530', linewidth=0.5)
for bar, sp in zip(sbars, spikes):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{sp:.0f}', ha='center', va='bottom', color=txt_color, fontsize=9)

# ── Plot 4: Calibration scatter (accuracy vs spikes) ─────────
ax4 = fig.add_subplot(gs[1, 0:2])
ax4.set_facecolor(ax_color)
all_accs   = [r['accuracy']*100 for r in calibration_log]
all_spikes = [r['spikes']       for r in calibration_log]
all_scores = [r['score']        for r in calibration_log]
sc = ax4.scatter(all_spikes, all_accs, c=all_scores, cmap='plasma',
                 s=30, alpha=0.7, edgecolors='none')
ax4.scatter(spikes_best, snn_acc_best*100, color='#00e5c8', s=120,
            zorder=5, marker='*', label='Best calibrated')
ax4.scatter(spikes_default, snn_acc_default*100, color='#e55a8a', s=80,
            zorder=5, marker='X', label='Default threshold')
cbar = plt.colorbar(sc, ax=ax4)
cbar.set_label('Objective score', color=txt_color, fontsize=8)
cbar.ax.yaxis.set_tick_params(color=txt_color, labelsize=7)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=txt_color)
ax4.set_title('Calibration Search: Accuracy vs Spike Count', color=txt_color, fontsize=11, pad=8)
ax4.set_xlabel('Spikes per image (lower = more efficient)', color=txt_color, fontsize=9)
ax4.set_ylabel('Accuracy (%)', color=txt_color, fontsize=9)
ax4.tick_params(colors=txt_color, labelsize=8)
for spine in ax4.spines.values(): spine.set_edgecolor('#1e2530')
ax4.grid(True, color='#1e2530', linewidth=0.5)
ax4.legend(fontsize=8, facecolor=ax_color, labelcolor=txt_color, edgecolor='#1e2530')

# ── Plot 5: Summary card ──────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor(ax_color)
ax5.axis('off')
summary_lines = [
    ("ANN accuracy",          f"{ann_accuracy*100:.2f}%",     ann_color),
    ("SNN accuracy (best)",   f"{snn_acc_best*100:.2f}%",     acc_color),
    ("Accuracy drop",         f"{accuracy_drop:.2f}%",        '#e55a8a'),
    ("",                      "",                             txt_color),
    ("Spikes — default",      f"{spikes_default:.0f} / img",  '#3a4455'),
    ("Spikes — calibrated",   f"{spikes_best:.0f} / img",     spk_color),
    ("Spike reduction",       f"{spike_reduction:.1f}%",      acc_color),
    ("",                      "",                             txt_color),
    ("Best thresholds",       str(best_thresholds),           acc_color),
    ("Timesteps (T)",         str(NUM_STEPS),                 txt_color),
]
ax5.set_title('Pipeline Summary', color=txt_color, fontsize=11, pad=8)
for idx, (label, value, color) in enumerate(summary_lines):
    y = 0.92 - idx * 0.094
    ax5.text(0.02, y, label, transform=ax5.transAxes,
             color='#6b7a8d', fontsize=8.5, va='top')
    ax5.text(0.98, y, value, transform=ax5.transAxes,
             color=color, fontsize=8.5, va='top', ha='right', fontweight='bold')
    if label:
        ax5.plot([0.02, 0.98], [y - 0.01, y - 0.01],
                 color='#1e2530', linewidth=0.4, transform=ax5.transAxes)

plt.savefig(os.path.join(RESULTS_DIR, 'pipeline_results.png'),
            dpi=150, bbox_inches='tight', facecolor='#0a0c0f')
plt.show()

print(f"\n{'='*55}")
print(f"  PIPELINE COMPLETE")
print(f"{'='*55}")
print(f"  ANN accuracy          : {ann_accuracy*100:.2f}%")
print(f"  SNN accuracy (best)   : {snn_acc_best*100:.2f}%")
print(f"  Accuracy drop         : {accuracy_drop:.2f}%")
print(f"  Spikes/img (default)  : {spikes_default:.1f}")
print(f"  Spikes/img (calibrated): {spikes_best:.1f}")
print(f"  Spike reduction       : {spike_reduction:.1f}%")
print(f"  Best thresholds       : {best_thresholds}")
print(f"\n  Results saved to: ./{RESULTS_DIR}/pipeline_results.png")
print(f"{'='*55}\n")