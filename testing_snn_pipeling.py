"""
Test Suite: ANN → SNN Conversion Pipeline
==========================================
Brain-Inspired Computing Project

Run with:
    pytest test_snn_pipeline.py -v

Tests are organized into five groups:
  1. CNN (ANN) architecture and forward pass
  2. SNN architecture and spiking behavior
  3. Weight copying (ANN → SNN)
  4. Evaluation functions
  5. Calibration scoring logic
"""

import pytest
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


# ─────────────────────────────────────────────────────────────────
# Re-define the models here so tests don't depend on running
# the full pipeline script (which trains and takes several minutes).
# These definitions must stay in sync with snn_pipeline.py.
# ─────────────────────────────────────────────────────────────────

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class SNN(nn.Module):
    def __init__(self, thresholds):
        super().__init__()
        t1, t2, t3, t4 = thresholds
        self.conv1   = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.lif1    = snn.Leaky(beta=0.9, threshold=t1, spike_grad=surrogate.fast_sigmoid())
        self.pool1   = nn.MaxPool2d(2)
        self.conv2   = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.lif2    = snn.Leaky(beta=0.9, threshold=t2, spike_grad=surrogate.fast_sigmoid())
        self.pool2   = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(32 * 7 * 7, 256)
        self.lif3    = snn.Leaky(beta=0.9, threshold=t3, spike_grad=surrogate.fast_sigmoid())
        self.fc2     = nn.Linear(256, 10)
        self.lif4    = snn.Leaky(beta=0.9, threshold=t4, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x, num_steps):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        spike_accumulator = torch.zeros(x.size(0), 10)
        total_spikes = 0
        for _ in range(num_steps):
            cur1       = self.conv1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            out1       = self.pool1(spk1)
            total_spikes += spk1.sum().item()
            cur2       = self.conv2(out1)
            spk2, mem2 = self.lif2(cur2, mem2)
            out2       = self.pool2(spk2)
            total_spikes += spk2.sum().item()
            flat       = self.flatten(out2)
            cur3       = self.fc1(flat)
            spk3, mem3 = self.lif3(cur3, mem3)
            total_spikes += spk3.sum().item()
            cur4       = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)
            spike_accumulator += spk4
            total_spikes += spk4.sum().item()
        return spike_accumulator, total_spikes


def copy_weights(ann, snn_model):
    snn_model.conv1.weight.data = ann.features[0].weight.data.clone()
    snn_model.conv1.bias.data   = ann.features[0].bias.data.clone()
    snn_model.conv2.weight.data = ann.features[3].weight.data.clone()
    snn_model.conv2.bias.data   = ann.features[3].bias.data.clone()
    snn_model.fc1.weight.data   = ann.classifier[1].weight.data.clone()
    snn_model.fc1.bias.data     = ann.classifier[1].bias.data.clone()
    snn_model.fc2.weight.data   = ann.classifier[3].weight.data.clone()
    snn_model.fc2.bias.data     = ann.classifier[3].bias.data.clone()
    return snn_model


# ─────────────────────────────────────────────────────────────────
# FIXTURES — reusable objects shared across tests
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def ann():
    """A freshly initialized (untrained) CNN."""
    return CNN()

@pytest.fixture
def default_snn():
    """SNN with all thresholds set to 1.0 (the pipeline default)."""
    return SNN([1.0, 1.0, 1.0, 1.0])

@pytest.fixture
def mnist_batch():
    """A fake batch of 8 MNIST-shaped images (1 channel, 28x28)."""
    return torch.randn(8, 1, 28, 28)

@pytest.fixture
def single_image():
    """A single MNIST-shaped image with batch dimension."""
    return torch.randn(1, 1, 28, 28)


# ─────────────────────────────────────────────────────────────────
# GROUP 1: CNN (ANN) architecture
# ─────────────────────────────────────────────────────────────────

class TestCNNArchitecture:

    def test_output_shape_batch(self, ann, mnist_batch):
        """CNN should output [batch_size, 10] — one score per digit class."""
        output = ann(mnist_batch)
        assert output.shape == (8, 10), (
            f"Expected output shape (8, 10), got {output.shape}"
        )

    def test_output_shape_single(self, ann, single_image):
        """CNN should handle a single image correctly."""
        output = ann(single_image)
        assert output.shape == (1, 10)

    def test_output_is_logits_not_probabilities(self, ann, mnist_batch):
        """
        CNN output should be raw logits (can be any real number),
        not softmax probabilities (which would all be between 0 and 1
        and sum to 1). We check that at least some values are outside [0,1].
        """
        output = ann(mnist_batch)
        has_value_outside_unit_interval = (
            (output > 1.0).any() or (output < 0.0).any()
        )
        assert has_value_outside_unit_interval, (
            "CNN output looks like probabilities — expected raw logits"
        )

    def test_different_inputs_give_different_outputs(self, ann):
        """Two different images should produce different logits."""
        img1 = torch.randn(1, 1, 28, 28)
        img2 = torch.randn(1, 1, 28, 28)
        out1 = ann(img1)
        out2 = ann(img2)
        assert not torch.allclose(out1, out2), (
            "Different inputs produced identical outputs — model may be broken"
        )

    def test_conv1_layer_shape(self, ann):
        """First conv layer: 1 input channel → 16 output channels, 3x3 kernel."""
        conv1 = ann.features[0]
        assert isinstance(conv1, nn.Conv2d)
        assert conv1.in_channels == 1
        assert conv1.out_channels == 16
        assert conv1.kernel_size == (3, 3)

    def test_conv2_layer_shape(self, ann):
        """Second conv layer: 16 → 32 channels."""
        conv2 = ann.features[3]
        assert isinstance(conv2, nn.Conv2d)
        assert conv2.in_channels == 16
        assert conv2.out_channels == 32

    def test_fc_output_classes(self, ann):
        """Final fully connected layer should output exactly 10 classes."""
        fc_out = ann.classifier[3]
        assert isinstance(fc_out, nn.Linear)
        assert fc_out.out_features == 10

    def test_fc1_dimensions(self, ann):
        """First FC layer: 32*7*7=1568 inputs → 256 outputs."""
        fc1 = ann.classifier[1]
        assert fc1.in_features == 32 * 7 * 7
        assert fc1.out_features == 256

    def test_prediction_is_valid_class(self, ann, mnist_batch):
        """Argmax prediction should always be a digit 0–9."""
        output = ann(mnist_batch)
        preds = output.argmax(dim=1)
        assert preds.min() >= 0
        assert preds.max() <= 9

    def test_gradients_flow(self, ann, mnist_batch):
        """
        After a forward + backward pass, all parameters should have gradients.
        If gradients don't flow, the model can't be trained.
        """
        criterion = nn.CrossEntropyLoss()
        targets = torch.zeros(8, dtype=torch.long)  # dummy targets (all class 0)
        output = ann(mnist_batch)
        loss = criterion(output, targets)
        loss.backward()
        for name, param in ann.named_parameters():
            assert param.grad is not None, f"No gradient for parameter: {name}"


# ─────────────────────────────────────────────────────────────────
# GROUP 2: SNN architecture and spiking behavior
# ─────────────────────────────────────────────────────────────────

class TestSNNArchitecture:

    def test_output_shape(self, default_snn, mnist_batch):
        """SNN spike accumulator should be [batch_size, 10]."""
        spike_out, _ = default_snn(mnist_batch, num_steps=10)
        assert spike_out.shape == (8, 10)

    def test_spike_count_is_non_negative(self, default_snn, mnist_batch):
        """Total spike count can never be negative — spikes are binary 0/1."""
        _, total_spikes = default_snn(mnist_batch, num_steps=10)
        assert total_spikes >= 0

    def test_spike_accumulator_non_negative(self, default_snn, mnist_batch):
        """
        Spike accumulator counts how many times each output neuron fired.
        Counts can't be negative.
        """
        spike_out, _ = default_snn(mnist_batch, num_steps=10)
        assert (spike_out >= 0).all(), "Spike accumulator contains negative values"

    def test_more_timesteps_more_or_equal_spikes(self, default_snn, single_image):
        """
        Running for more timesteps should produce at least as many spikes.
        More time = more opportunity to fire.
        """
        torch.manual_seed(0)
        _, spikes_5  = default_snn(single_image, num_steps=5)
        torch.manual_seed(0)
        _, spikes_25 = default_snn(single_image, num_steps=25)
        assert spikes_25 >= spikes_5, (
            "More timesteps produced fewer spikes — unexpected behavior"
        )

    def test_high_threshold_reduces_spikes(self, mnist_batch):
        """
        A very high threshold (neurons rarely fire) should produce fewer
        spikes than a low threshold (neurons fire easily).
        """
        low_thresh  = SNN([0.3, 0.3, 0.3, 0.3])
        high_thresh = SNN([3.0, 3.0, 3.0, 3.0])
        _, spikes_low  = low_thresh(mnist_batch,  num_steps=20)
        _, spikes_high = high_thresh(mnist_batch, num_steps=20)
        assert spikes_high <= spikes_low, (
            f"High threshold ({spikes_high}) produced more spikes than "
            f"low threshold ({spikes_low})"
        )

    def test_zero_input_fewer_spikes_than_strong_input(self):
        """
        Zero input should produce fewer spikes than a strong positive input.

        Note: zero input does NOT guarantee near-zero spikes in an untrained
        model because bias terms still drive membrane potentials even with no
        image signal. In a trained model biases are calibrated to real inputs,
        but here we use a freshly initialized model. What we CAN assert is that
        a strong positive input (which adds on top of the biases) produces at
        least as many spikes as a zero input — after zeroing biases so that
        the image signal is the only driver.
        """
        torch.manual_seed(42)
        model = SNN([1.0, 1.0, 1.0, 1.0])
        # Zero out all biases so the input image is the only current driver
        for layer in [model.conv1, model.conv2, model.fc1, model.fc2]:
            nn.init.zeros_(layer.bias)

        zero_input   = torch.zeros(1, 1, 28, 28)
        strong_input = torch.ones(1, 1, 28, 28) * 5.0

        _, spikes_zero   = model(zero_input,   num_steps=10)
        _, spikes_strong = model(strong_input, num_steps=10)

        assert spikes_zero < spikes_strong, (
            f"Zero input ({spikes_zero}) should produce fewer spikes than "
            f"strong input ({spikes_strong}) when biases are zeroed"
        )

    def test_lif_neurons_exist(self, default_snn):
        """All four LIF layers must be present in the SNN."""
        assert hasattr(default_snn, 'lif1'), "Missing lif1 layer"
        assert hasattr(default_snn, 'lif2'), "Missing lif2 layer"
        assert hasattr(default_snn, 'lif3'), "Missing lif3 layer"
        assert hasattr(default_snn, 'lif4'), "Missing lif4 layer"

    def test_lif_beta_value(self, default_snn):
        """LIF neurons should use beta=0.9 (the leak factor)."""
        assert abs(default_snn.lif1.beta.item() - 0.9) < 1e-5
        assert abs(default_snn.lif2.beta.item() - 0.9) < 1e-5
        assert abs(default_snn.lif3.beta.item() - 0.9) < 1e-5
        assert abs(default_snn.lif4.beta.item() - 0.9) < 1e-5

    def test_custom_thresholds_are_applied(self):
        """The threshold values passed in should be reflected in the LIF layers."""
        thresholds = [0.5, 0.75, 1.25, 2.0]
        model = SNN(thresholds)
        assert abs(model.lif1.threshold.item() - 0.5)  < 1e-5
        assert abs(model.lif2.threshold.item() - 0.75) < 1e-5
        assert abs(model.lif3.threshold.item() - 1.25) < 1e-5
        assert abs(model.lif4.threshold.item() - 2.0)  < 1e-5

    def test_prediction_is_valid_class(self, default_snn, mnist_batch):
        """SNN predictions should always be valid digit classes 0–9."""
        spike_out, _ = default_snn(mnist_batch, num_steps=10)
        preds = spike_out.argmax(dim=1)
        assert preds.min() >= 0
        assert preds.max() <= 9

    def test_deterministic_with_same_input(self, default_snn, single_image):
        """
        SNN is deterministic — same input should always give same output.
        (There is no randomness in a LIF neuron.)
        """
        out1, spikes1 = default_snn(single_image, num_steps=10)
        out2, spikes2 = default_snn(single_image, num_steps=10)
        assert torch.allclose(out1, out2)
        assert spikes1 == spikes2


# ─────────────────────────────────────────────────────────────────
# GROUP 3: Weight copying (ANN → SNN)
# ─────────────────────────────────────────────────────────────────

class TestWeightCopying:

    def test_conv1_weights_copied(self, ann, default_snn):
        """Conv1 weights should be identical after copying."""
        copy_weights(ann, default_snn)
        assert torch.allclose(
            ann.features[0].weight.data,
            default_snn.conv1.weight.data
        ), "conv1 weights were not copied correctly"

    def test_conv1_bias_copied(self, ann, default_snn):
        """Conv1 bias should be identical after copying."""
        copy_weights(ann, default_snn)
        assert torch.allclose(
            ann.features[0].bias.data,
            default_snn.conv1.bias.data
        ), "conv1 bias was not copied correctly"

    def test_conv2_weights_copied(self, ann, default_snn):
        """Conv2 weights should be identical after copying."""
        copy_weights(ann, default_snn)
        assert torch.allclose(
            ann.features[3].weight.data,
            default_snn.conv2.weight.data
        ), "conv2 weights were not copied correctly"

    def test_fc1_weights_copied(self, ann, default_snn):
        """FC1 weights should be identical after copying."""
        copy_weights(ann, default_snn)
        assert torch.allclose(
            ann.classifier[1].weight.data,
            default_snn.fc1.weight.data
        ), "fc1 weights were not copied correctly"

    def test_fc2_weights_copied(self, ann, default_snn):
        """FC2 weights should be identical after copying."""
        copy_weights(ann, default_snn)
        assert torch.allclose(
            ann.classifier[3].weight.data,
            default_snn.fc2.weight.data
        ), "fc2 weights were not copied correctly"

    def test_copy_is_a_clone_not_a_reference(self, ann, default_snn):
        """
        Modifying ANN weights after copying should NOT affect the SNN.
        copy_weights uses .clone(), so the two models are independent.
        """
        copy_weights(ann, default_snn)
        original_snn_weight = default_snn.conv1.weight.data.clone()
        # Mutate the ANN weights
        ann.features[0].weight.data.fill_(99.0)
        # SNN weights should be unchanged
        assert torch.allclose(default_snn.conv1.weight.data, original_snn_weight), (
            "SNN weights changed when ANN weights were modified — "
            "copy_weights is sharing a reference instead of cloning"
        )

    def test_weight_shapes_are_compatible(self, ann, default_snn):
        """All corresponding weight tensors should have matching shapes."""
        assert ann.features[0].weight.shape == default_snn.conv1.weight.shape
        assert ann.features[3].weight.shape == default_snn.conv2.weight.shape
        assert ann.classifier[1].weight.shape == default_snn.fc1.weight.shape
        assert ann.classifier[3].weight.shape == default_snn.fc2.weight.shape


# ─────────────────────────────────────────────────────────────────
# GROUP 4: Evaluation logic
# ─────────────────────────────────────────────────────────────────

class TestEvaluationLogic:

    def test_ann_accuracy_range(self, ann):
        """
        ANN accuracy on a small random batch should be between 0 and 1.
        (A random untrained model should still produce valid probability values.)
        """
        data   = torch.randn(16, 1, 28, 28)
        target = torch.randint(0, 10, (16,))
        ann.eval()
        with torch.no_grad():
            output = ann(data)
            pred   = output.argmax(dim=1)
            acc    = (pred == target).float().mean().item()
        assert 0.0 <= acc <= 1.0

    def test_snn_spike_count_scales_with_batch_size(self, default_snn):
        """
        A batch of 8 images should produce approximately 8× the spikes
        of a single image (since each image is processed independently).
        """
        single = torch.randn(1, 1, 28, 28)
        batch  = single.repeat(8, 1, 1, 1)  # 8 identical images
        _, spikes_single = default_snn(single, num_steps=10)
        _, spikes_batch  = default_snn(batch,  num_steps=10)
        # Should be exactly 8× since all images are identical
        assert abs(spikes_batch - 8 * spikes_single) < 1e-3, (
            f"Expected ~{8 * spikes_single} spikes for batch of 8, "
            f"got {spikes_batch}"
        )

    def test_perfect_predictions_give_accuracy_one(self, ann):
        """If predictions exactly match targets, accuracy should be 1.0."""
        data   = torch.randn(4, 1, 28, 28)
        output = ann(data)
        # Use the model's own predictions as the targets — perfect accuracy
        targets = output.argmax(dim=1)
        correct = (output.argmax(dim=1) == targets).sum().item()
        acc = correct / len(targets)
        assert acc == 1.0

    def test_wrong_predictions_give_accuracy_zero(self):
        """If all predictions are wrong, accuracy should be 0.0."""
        # Predictions: all class 0
        # Targets: all class 1
        preds   = torch.zeros(10, dtype=torch.long)
        targets = torch.ones(10,  dtype=torch.long)
        acc = (preds == targets).float().mean().item()
        assert acc == 0.0


# ─────────────────────────────────────────────────────────────────
# GROUP 5: Calibration scoring logic
# ─────────────────────────────────────────────────────────────────

class TestCalibrationScoring:

    def _score(self, accuracy, spikes, spikes_default, lam=0.3):
        """Mirror of the scoring formula from snn_pipeline.py."""
        norm_spikes = spikes / max(spikes_default, 1)
        return accuracy - lam * norm_spikes

    def test_higher_accuracy_gives_higher_score(self):
        """With equal spike counts, higher accuracy should win."""
        score_high = self._score(accuracy=0.98, spikes=500, spikes_default=500)
        score_low  = self._score(accuracy=0.90, spikes=500, spikes_default=500)
        assert score_high > score_low

    def test_fewer_spikes_gives_higher_score(self):
        """With equal accuracy, fewer spikes should give a higher score."""
        score_efficient = self._score(accuracy=0.95, spikes=300, spikes_default=500)
        score_wasteful  = self._score(accuracy=0.95, spikes=700, spikes_default=500)
        assert score_efficient > score_wasteful

    def test_score_reflects_tradeoff(self):
        """
        A model with slightly lower accuracy but far fewer spikes
        can outscore a more accurate but spike-heavy model,
        depending on lambda.
        """
        # 98% accuracy, 800 spikes vs 95% accuracy, 200 spikes
        score_accurate   = self._score(accuracy=0.98, spikes=800, spikes_default=500)
        score_efficient  = self._score(accuracy=0.95, spikes=200, spikes_default=500)
        # With lambda=0.3, the efficient one should win here
        assert score_efficient > score_accurate, (
            "Expected the efficient (low-spike) model to win with lambda=0.3"
        )

    def test_score_is_deterministic(self):
        """Same inputs should always produce the same score."""
        s1 = self._score(accuracy=0.97, spikes=450, spikes_default=500)
        s2 = self._score(accuracy=0.97, spikes=450, spikes_default=500)
        assert s1 == s2

    def test_score_with_zero_spikes(self):
        """
        If a model somehow produces zero spikes, spike penalty is zero
        and score equals accuracy alone.
        """
        score = self._score(accuracy=0.95, spikes=0, spikes_default=500)
        assert abs(score - 0.95) < 1e-9

    def test_lambda_zero_ignores_spikes(self):
        """Lambda=0 means spike count doesn't matter — score = accuracy."""
        score = self._score(accuracy=0.93, spikes=9999, spikes_default=500, lam=0.0)
        assert abs(score - 0.93) < 1e-9

    def test_best_threshold_has_highest_score(self):
        """
        Given a list of calibration results, the entry with the highest
        score should be selected as best — simulating the pipeline's
        search loop logic.
        """
        results = [
            {'thresholds': [1.0, 1.0, 1.0, 1.0], 'score': 0.62},
            {'thresholds': [0.75, 1.0, 1.5, 0.75], 'score': 0.71},  # best
            {'thresholds': [1.25, 1.0, 1.0, 1.0], 'score': 0.58},
        ]
        best = max(results, key=lambda r: r['score'])
        assert best['thresholds'] == [0.75, 1.0, 1.5, 0.75], (
            "Wrong entry selected as best"
        )
        assert best['score'] == 0.71