"""
Test Suite: ANN → SNN Conversion Pipeline
==========================================
Brain-Inspired Computing Project

PURPOSE:
    These tests verify that every component of the pipeline works correctly
    in isolation. Rather than running the full script (which takes several
    minutes to train), each test uses small, fast, randomly initialized models
    and synthetic data to check one specific behavior at a time.

HOW TO RUN:
    pytest test_snn_pipeline.py -v

    The -v flag (verbose) prints each test name and pass/fail individually,
    which is much more useful than the default dot-summary.

TEST GROUPS:
    1. TestCNNArchitecture   — shape, structure, and forward pass of the ANN
    2. TestSNNArchitecture   — spiking behavior and LIF neuron properties
    3. TestWeightCopying     — correctness of the ANN → SNN weight transfer
    4. TestEvaluationLogic   — accuracy calculations and spike counting
    5. TestCalibrationScoring — the objective function that finds best thresholds

NOTE ON MODEL DEFINITIONS:
    The CNN and SNN classes below are re-defined here (copied from
    snn_pipeline.py) so that tests run independently of the main script.
    If you change the architecture in snn_pipeline.py, update these too.
"""

import pytest
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


# ─────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# Copied from snn_pipeline.py so tests are self-contained.
# ─────────────────────────────────────────────────────────────────

class CNN(nn.Module):
    """
    The standard (non-spiking) convolutional neural network.
    Trained first on MNIST, then its weights are copied into the SNN.

    Architecture:
        Conv(1→16) → ReLU → MaxPool
        Conv(16→32) → ReLU → MaxPool
        Flatten → FC(1568→256) → ReLU → FC(256→10)
    """
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
    """
    The spiking neural network equivalent of the CNN above.
    ReLU activations are replaced with Leaky Integrate-and-Fire (LIF) neurons.
    MaxPool layers are kept — they work fine with binary spike signals.

    The 'thresholds' argument controls when each LIF neuron fires.
    Lower threshold → fires more easily → more spikes → less efficient.
    Higher threshold → fires less easily → fewer spikes → more efficient.
    Finding the right thresholds is what the calibration step does.
    """
    def __init__(self, thresholds):
        super().__init__()
        t1, t2, t3, t4 = thresholds

        # Conv block 1: spatial feature extraction, first level
        self.conv1   = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.lif1    = snn.Leaky(beta=0.9, threshold=t1, spike_grad=surrogate.fast_sigmoid())
        self.pool1   = nn.MaxPool2d(2)

        # Conv block 2: higher-level spatial features
        self.conv2   = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.lif2    = snn.Leaky(beta=0.9, threshold=t2, spike_grad=surrogate.fast_sigmoid())
        self.pool2   = nn.MaxPool2d(2)

        # Fully connected block: combines spatial features into a class decision
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(32 * 7 * 7, 256)
        self.lif3    = snn.Leaky(beta=0.9, threshold=t3, spike_grad=surrogate.fast_sigmoid())

        # Output layer: one neuron per digit class (0–9)
        self.fc2     = nn.Linear(256, 10)
        self.lif4    = snn.Leaky(beta=0.9, threshold=t4, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x, num_steps):
        """
        Runs the network for 'num_steps' timesteps on the same input image.

        Unlike a standard neural network which processes an image once,
        the SNN processes it repeatedly. Each LIF neuron accumulates input
        current into its membrane potential across timesteps, firing a binary
        spike whenever it crosses the threshold, then resetting.

        The final classification is based on which output neuron fired the
        most total spikes across all timesteps — the winner-takes-all decision.

        total_spikes is our proxy for energy cost: on a real neuromorphic chip,
        each spike propagating across a synapse costs a fixed amount of energy.
        Fewer total spikes = lower power consumption.
        """
        # Initialize membrane potential to zero for each LIF layer at the start
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        # Accumulate output spikes across all timesteps to make a class decision
        spike_accumulator = torch.zeros(x.size(0), 10)
        total_spikes = 0  # counts every spike across every layer and timestep

        for _ in range(num_steps):
            # Each timestep: run the full network, collecting spikes at each layer
            cur1       = self.conv1(x)
            spk1, mem1 = self.lif1(cur1, mem1)   # spk1 is binary: 0 or 1 per neuron
            out1       = self.pool1(spk1)
            total_spikes += spk1.sum().item()     # count how many neurons fired

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
            spike_accumulator += spk4             # tally output spikes for classification
            total_spikes += spk4.sum().item()

        return spike_accumulator, total_spikes


def copy_weights(ann, snn_model):
    """
    Transfers trained weights from the ANN into the SNN layer by layer.

    This is the core of the ANN-to-SNN conversion: the SNN uses the same
    learned weights as the ANN, but fires spikes instead of computing
    continuous ReLU activations. The idea is that the trained weights already
    encode what features to look for — we just change HOW those features
    are communicated (continuous values → binary spikes).

    .clone() is used so the SNN gets an independent copy of the weights.
    If we used assignment without clone(), both models would share the same
    underlying tensor, and modifying one would silently corrupt the other.
    """
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
# FIXTURES
# Fixtures are reusable setup objects that pytest injects into tests
# automatically. Instead of creating a new CNN or fake image in every
# single test, we define them once here and reference them by name
# in test function arguments.
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def ann():
    """
    A freshly initialized (untrained) CNN.
    Weights are random — this is enough to test architecture and
    data flow, but not accuracy (which requires training on MNIST).
    """
    return CNN()

@pytest.fixture
def default_snn():
    """
    SNN with all thresholds set to 1.0 — the default the pipeline uses
    before calibration runs. A neuron fires when its accumulated input
    reaches 1.0 units of membrane potential.
    """
    return SNN([1.0, 1.0, 1.0, 1.0])

@pytest.fixture
def mnist_batch():
    """
    A synthetic batch of 8 random images in MNIST format:
    8 images × 1 color channel × 28 pixels tall × 28 pixels wide.
    Using random noise instead of real digits is fine for shape/flow tests.
    """
    return torch.randn(8, 1, 28, 28)

@pytest.fixture
def single_image():
    """
    A single synthetic MNIST-shaped image with the batch dimension included.
    Shape (1, 1, 28, 28): batch=1, channels=1, height=28, width=28.
    """
    return torch.randn(1, 1, 28, 28)


# ─────────────────────────────────────────────────────────────────
# GROUP 1: CNN (ANN) Architecture
#
# These tests verify that the CNN is built correctly: right layer types,
# right dimensions, and that data flows through it without errors.
# None of these require training — we just check the structure and
# that a forward pass produces sensible output shapes.
# ─────────────────────────────────────────────────────────────────

class TestCNNArchitecture:

    def test_output_shape_batch(self, ann, mnist_batch):
        """
        The CNN's output should be shape [8, 10] for a batch of 8 images:
        one row per image, one column per digit class (0 through 9).
        If this shape is wrong, the rest of the pipeline will break because
        the loss function and accuracy calculation both depend on it.
        """
        output = ann(mnist_batch)
        assert output.shape == (8, 10), (
            f"Expected output shape (8, 10), got {output.shape}"
        )

    def test_output_shape_single(self, ann, single_image):
        """
        The CNN should handle a single image (batch size 1) just as well
        as a full batch. This tests that no layer is hardcoded to a specific
        batch size — all operations should be batch-agnostic.
        """
        output = ann(single_image)
        assert output.shape == (1, 10)

    def test_output_is_logits_not_probabilities(self, ann, mnist_batch):
        """
        The CNN outputs raw logits — unconstrained real numbers, one per class.
        These are fed into CrossEntropyLoss during training, which applies
        softmax internally. If someone accidentally added a softmax layer at
        the end of the CNN, all outputs would be in [0,1] and sum to 1,
        which would confuse the loss function. We check that at least some
        values fall outside [0,1], confirming raw logits are being returned.
        """
        output = ann(mnist_batch)
        has_value_outside_unit_interval = (
            (output > 1.0).any() or (output < 0.0).any()
        )
        assert has_value_outside_unit_interval, (
            "CNN output looks like probabilities — expected raw logits"
        )

    def test_different_inputs_give_different_outputs(self, ann):
        """
        Two different random images should produce different logit vectors.
        If both images produced identical output, it would mean the network
        is ignoring its input entirely — likely a bug where gradients aren't
        flowing, or a layer is zeroing everything out.
        """
        img1 = torch.randn(1, 1, 28, 28)
        img2 = torch.randn(1, 1, 28, 28)
        out1 = ann(img1)
        out2 = ann(img2)
        assert not torch.allclose(out1, out2), (
            "Different inputs produced identical outputs — model may be broken"
        )

    def test_conv1_layer_shape(self, ann):
        """
        The first convolutional layer takes a grayscale image (1 channel)
        and produces 16 feature maps using 3×3 filters with padding=1
        (which keeps the spatial dimensions the same: 28×28 → 28×28).
        Verifying this exactly ensures the architecture wasn't accidentally
        changed in a way that would make weight copying fail.
        """
        conv1 = ann.features[0]
        assert isinstance(conv1, nn.Conv2d)
        assert conv1.in_channels == 1       # grayscale: 1 input channel
        assert conv1.out_channels == 16     # produces 16 feature maps
        assert conv1.kernel_size == (3, 3)  # 3x3 sliding window

    def test_conv2_layer_shape(self, ann):
        """
        The second conv layer takes the 16 feature maps from conv1
        and produces 32 deeper feature maps. The channel count doubling
        (16 → 32) is a common CNN design pattern: detect more complex
        patterns at each successive layer.
        """
        conv2 = ann.features[3]
        assert isinstance(conv2, nn.Conv2d)
        assert conv2.in_channels == 16
        assert conv2.out_channels == 32

    def test_fc_output_classes(self, ann):
        """
        The final fully connected layer must output exactly 10 values —
        one score per digit class (0, 1, 2, ..., 9). If this were wrong
        (say, 9 outputs), class 9 would never be predicted, and the model
        would silently fail on one-tenth of the test set.
        """
        fc_out = ann.classifier[3]
        assert isinstance(fc_out, nn.Linear)
        assert fc_out.out_features == 10

    def test_fc1_dimensions(self, ann):
        """
        After two rounds of Conv+MaxPool, the spatial dimensions shrink:
        28×28 → 14×14 → 7×7. With 32 feature maps, that's 32×7×7 = 1568
        values per image entering the first FC layer. The FC layer compresses
        this down to 256. If either number is wrong, the model can't be built
        because the tensor shapes won't match.
        """
        fc1 = ann.classifier[1]
        assert fc1.in_features == 32 * 7 * 7  # = 1568
        assert fc1.out_features == 256

    def test_prediction_is_valid_class(self, ann, mnist_batch):
        """
        The predicted class for any image (the argmax of the output logits)
        must always be an integer between 0 and 9 inclusive. Values outside
        this range would indicate a fundamental output dimension bug.
        """
        output = ann(mnist_batch)
        preds = output.argmax(dim=1)  # pick the class with the highest score
        assert preds.min() >= 0
        assert preds.max() <= 9

    def test_gradients_flow(self, ann, mnist_batch):
        """
        After a forward pass and backward pass (computing the loss and
        calling .backward()), every learnable parameter in the network
        should have a gradient. If any parameter has None as its gradient,
        it means the loss signal can't reach that layer, and those weights
        will never update during training — the model is effectively broken.

        This is one of the most important structural tests: a model can
        look correct (right shapes, right outputs) but be untrainable if
        the computation graph is disconnected somewhere.
        """
        criterion = nn.CrossEntropyLoss()
        # Use all-zero targets (class 0) — we only care that gradients exist,
        # not that the model is making good predictions here
        targets = torch.zeros(8, dtype=torch.long)
        output = ann(mnist_batch)
        loss = criterion(output, targets)
        loss.backward()  # compute gradients through the entire network

        for name, param in ann.named_parameters():
            assert param.grad is not None, f"No gradient for parameter: {name}"


# ─────────────────────────────────────────────────────────────────
# GROUP 2: SNN Architecture and Spiking Behavior
#
# These tests verify the spiking mechanics of the SNN — the properties
# that make it fundamentally different from the ANN and directly
# responsible for the energy efficiency we're demonstrating.
# ─────────────────────────────────────────────────────────────────

class TestSNNArchitecture:

    def test_output_shape(self, default_snn, mnist_batch):
        """
        The SNN returns a spike accumulator of shape [batch_size, 10]:
        for each image in the batch, how many times each of the 10 output
        neurons fired across all timesteps. The class with the most spikes
        is the model's prediction — the same argmax logic as the ANN,
        but computed over time rather than over continuous activations.
        """
        spike_out, _ = default_snn(mnist_batch, num_steps=10)
        assert spike_out.shape == (8, 10)

    def test_spike_count_is_non_negative(self, default_snn, mnist_batch):
        """
        Total spike count must always be ≥ 0. Spikes are binary events
        (a neuron either fires or doesn't), so counting them can only
        ever produce zero or a positive number. A negative count would
        indicate a serious bug in the accumulation logic.
        """
        _, total_spikes = default_snn(mnist_batch, num_steps=10)
        assert total_spikes >= 0

    def test_spike_accumulator_non_negative(self, default_snn, mnist_batch):
        """
        The spike accumulator counts how many times each output neuron fired
        across all timesteps. Since spikes are binary (0 or 1), the count
        per neuron per image can never be negative. A negative value would
        suggest the accumulation is subtracting instead of adding, which
        would corrupt the classification decision.
        """
        spike_out, _ = default_snn(mnist_batch, num_steps=10)
        assert (spike_out >= 0).all(), "Spike accumulator contains negative values"

    def test_more_timesteps_more_or_equal_spikes(self, default_snn, single_image):
        """
        Running the SNN for more timesteps gives neurons more opportunities
        to accumulate membrane potential and cross the firing threshold.
        Therefore, 25 timesteps should produce at least as many total spikes
        as 5 timesteps on the same input image.

        This also confirms the temporal integration property of LIF neurons:
        they are not just threshold functions applied once, but accumulators
        that build up signal over time.
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
        The threshold is the membrane potential value a neuron must reach
        before it fires a spike. A high threshold (3.0) means neurons rarely
        fire — they need a lot of accumulated input. A low threshold (0.3)
        means neurons fire frequently on weak input.

        This property is central to our project: by choosing thresholds
        carefully during calibration, we can reduce the spike count (and
        thus energy use) without hurting accuracy too much. This test
        verifies that the threshold parameter actually does what we claim.
        """
        low_thresh  = SNN([0.3, 0.3, 0.3, 0.3])   # fires easily
        high_thresh = SNN([3.0, 3.0, 3.0, 3.0])   # fires rarely
        _, spikes_low  = low_thresh(mnist_batch,  num_steps=20)
        _, spikes_high = high_thresh(mnist_batch, num_steps=20)
        assert spikes_high <= spikes_low, (
            f"High threshold ({spikes_high}) produced more spikes than "
            f"low threshold ({spikes_low})"
        )

    def test_zero_input_fewer_spikes_than_strong_input(self):
        """
        Zero pixel values should drive less spiking than strong positive input.

        IMPORTANT NOTE on why we zero out biases first:
        In an untrained model, bias terms are randomly initialized and can be
        large positive numbers. Even with a zero image, those biases alone can
        push membrane potentials above the threshold, causing many spikes with
        no actual image signal present. In the real pipeline, training adjusts
        biases to sensible values relative to real digit inputs — but our test
        uses an untrained model. We zero the biases to isolate the effect of
        the image signal, which is the property we actually want to test:
        that stronger pixel values cause more spiking activity.
        """
        torch.manual_seed(42)
        model = SNN([1.0, 1.0, 1.0, 1.0])

        # Remove bias contribution so image pixel values are the only driver
        for layer in [model.conv1, model.conv2, model.fc1, model.fc2]:
            nn.init.zeros_(layer.bias)

        zero_input   = torch.zeros(1, 1, 28, 28)       # blank image
        strong_input = torch.ones(1, 1, 28, 28) * 5.0  # very bright image

        _, spikes_zero   = model(zero_input,   num_steps=10)
        _, spikes_strong = model(strong_input, num_steps=10)

        assert spikes_zero < spikes_strong, (
            f"Zero input ({spikes_zero}) should produce fewer spikes than "
            f"strong input ({spikes_strong}) when biases are zeroed"
        )

    def test_lif_neurons_exist(self, default_snn):
        """
        All four LIF layers must be present in the SNN. Each replaces one
        ReLU activation from the ANN:
          lif1 replaces ReLU after conv1
          lif2 replaces ReLU after conv2
          lif3 replaces ReLU after fc1
          lif4 is the output spiking layer (no ReLU equivalent in the ANN)
        If any of these are missing, the conversion is incomplete and
        weight copying will fail or produce incorrect results.
        """
        assert hasattr(default_snn, 'lif1'), "Missing lif1 layer"
        assert hasattr(default_snn, 'lif2'), "Missing lif2 layer"
        assert hasattr(default_snn, 'lif3'), "Missing lif3 layer"
        assert hasattr(default_snn, 'lif4'), "Missing lif4 layer"

    def test_lif_beta_value(self, default_snn):
        """
        Beta (β) is the leak factor in the LIF equation:
            u[t] = β · u[t-1] + input[t]

        β = 0.9 means 90% of the previous membrane potential is retained
        each timestep — a slow leak. This value was chosen to give neurons
        enough memory to accumulate signal over multiple timesteps, while
        still decaying so that old, stale input doesn't persist indefinitely.

        If β were 1.0, neurons would never leak (perfect memory, unrealistic).
        If β were 0.5, neurons would lose half their charge each step
        (too fast — short-term input would rarely accumulate enough to fire).
        """
        assert abs(default_snn.lif1.beta.item() - 0.9) < 1e-5
        assert abs(default_snn.lif2.beta.item() - 0.9) < 1e-5
        assert abs(default_snn.lif3.beta.item() - 0.9) < 1e-5
        assert abs(default_snn.lif4.beta.item() - 0.9) < 1e-5

    def test_custom_thresholds_are_applied(self):
        """
        Verifies that the threshold values passed into the SNN constructor
        are actually stored in the corresponding LIF layers. This is critical
        for calibration: if the thresholds we search over aren't being applied
        to the LIF neurons, the calibration search is meaningless — we'd be
        evaluating the same model over and over regardless of what thresholds
        we pass in.
        """
        thresholds = [0.5, 0.75, 1.25, 2.0]
        model = SNN(thresholds)
        assert abs(model.lif1.threshold.item() - 0.5)  < 1e-5
        assert abs(model.lif2.threshold.item() - 0.75) < 1e-5
        assert abs(model.lif3.threshold.item() - 1.25) < 1e-5
        assert abs(model.lif4.threshold.item() - 2.0)  < 1e-5

    def test_prediction_is_valid_class(self, default_snn, mnist_batch):
        """
        The SNN's classification decision is the output neuron that fired
        the most times (argmax of spike_accumulator). The result must always
        be a valid digit class: 0 through 9. Values outside this range would
        mean the output layer has the wrong number of neurons.
        """
        spike_out, _ = default_snn(mnist_batch, num_steps=10)
        preds = spike_out.argmax(dim=1)
        assert preds.min() >= 0
        assert preds.max() <= 9

    def test_deterministic_with_same_input(self, default_snn, single_image):
        """
        The SNN has no randomness — given the same input image and starting
        from the same initial membrane potential (zero), it must always produce
        exactly the same output. This is an important property for a real chip:
        if the same sensor reading produced different classifications on
        different runs, the system would be unreliable.

        If this test fails, it suggests some random operation was accidentally
        introduced into the forward pass.
        """
        out1, spikes1 = default_snn(single_image, num_steps=10)
        out2, spikes2 = default_snn(single_image, num_steps=10)
        assert torch.allclose(out1, out2), "Same input gave different spike counts"
        assert spikes1 == spikes2,         "Same input gave different total spikes"


# ─────────────────────────────────────────────────────────────────
# GROUP 3: Weight Copying (ANN → SNN)
#
# The conversion relies entirely on transferring learned weights from
# the ANN into the SNN. If any layer's weights are wrong — swapped,
# incorrectly indexed, or sharing memory with the ANN — the SNN will
# produce garbage outputs regardless of how well the ANN was trained.
# ─────────────────────────────────────────────────────────────────

class TestWeightCopying:

    def test_conv1_weights_copied(self, ann, default_snn):
        """
        After copy_weights(), the conv1 weight tensor in the SNN should be
        numerically identical to conv1 in the ANN. torch.allclose() checks
        that all values match within a tiny floating point tolerance.
        If this fails, conv1 in the SNN is still using its random
        initialization rather than the trained ANN weights.
        """
        copy_weights(ann, default_snn)
        assert torch.allclose(
            ann.features[0].weight.data,
            default_snn.conv1.weight.data
        ), "conv1 weights were not copied correctly"

    def test_conv1_bias_copied(self, ann, default_snn):
        """
        Biases matter as much as weights — they shift the activation of each
        neuron and are trained just like weights. If only weights are copied
        but biases are left at their random initialization, the SNN's neurons
        will have the right sensitivity to features but the wrong baseline
        activation level, which hurts accuracy.
        """
        copy_weights(ann, default_snn)
        assert torch.allclose(
            ann.features[0].bias.data,
            default_snn.conv1.bias.data
        ), "conv1 bias was not copied correctly"

    def test_conv2_weights_copied(self, ann, default_snn):
        """
        Conv2 weights encode higher-level features built on top of conv1's
        edge detectors — things like corners, curves, and stroke patterns
        that distinguish digits from each other. These must be copied
        correctly or the SNN won't recognize the same features the ANN learned.
        """
        copy_weights(ann, default_snn)
        assert torch.allclose(
            ann.features[3].weight.data,
            default_snn.conv2.weight.data
        ), "conv2 weights were not copied correctly"

    def test_fc1_weights_copied(self, ann, default_snn):
        """
        FC1 combines the spatial features from the convolutional layers
        into a 256-dimensional representation that captures the overall
        structure of the digit. Its weights determine how conv features
        combine into this intermediate representation — critical to get right.
        """
        copy_weights(ann, default_snn)
        assert torch.allclose(
            ann.classifier[1].weight.data,
            default_snn.fc1.weight.data
        ), "fc1 weights were not copied correctly"

    def test_fc2_weights_copied(self, ann, default_snn):
        """
        FC2 is the classification head — it maps the 256-dimensional
        representation to 10 class scores. These weights are the most
        directly responsible for the final digit prediction. If they're
        wrong, the SNN will produce the right internal representations
        but draw the wrong conclusions from them.
        """
        copy_weights(ann, default_snn)
        assert torch.allclose(
            ann.classifier[3].weight.data,
            default_snn.fc2.weight.data
        ), "fc2 weights were not copied correctly"

    def test_copy_is_a_clone_not_a_reference(self, ann, default_snn):
        """
        copy_weights() uses .clone() to create independent copies of each
        tensor. This test verifies that independence: if we modify the ANN's
        weights after copying, the SNN's weights must remain unchanged.

        Why this matters: if copy_weights() used direct assignment (=) instead
        of .clone(), both models would point to the same underlying memory.
        Any future modification to the ANN (e.g., fine-tuning, re-training)
        would silently corrupt the SNN — a very hard bug to diagnose because
        the SNN would still pass shape tests but produce wrong predictions.
        """
        copy_weights(ann, default_snn)
        # Save what the SNN's conv1 weights look like right after copying
        original_snn_weight = default_snn.conv1.weight.data.clone()

        # Deliberately corrupt the ANN's conv1 weights
        ann.features[0].weight.data.fill_(99.0)

        # The SNN should be completely unaffected — it has its own copy
        assert torch.allclose(default_snn.conv1.weight.data, original_snn_weight), (
            "SNN weights changed when ANN weights were modified — "
            "copy_weights is sharing a reference instead of cloning"
        )

    def test_weight_shapes_are_compatible(self, ann, default_snn):
        """
        Before copying weights, we need to confirm that corresponding layers
        in the ANN and SNN have the same tensor shapes. If the shapes differ,
        the copy would either raise an error or silently reshape data, both
        of which would produce incorrect results.

        This acts as a guard: if someone changes the ANN architecture without
        updating the SNN (or vice versa), this test catches it immediately.
        """
        assert ann.features[0].weight.shape == default_snn.conv1.weight.shape
        assert ann.features[3].weight.shape == default_snn.conv2.weight.shape
        assert ann.classifier[1].weight.shape == default_snn.fc1.weight.shape
        assert ann.classifier[3].weight.shape == default_snn.fc2.weight.shape


# ─────────────────────────────────────────────────────────────────
# GROUP 4: Evaluation Logic
#
# These tests verify the correctness of accuracy calculation and
# spike counting — the two metrics the pipeline reports. If either
# is wrong, our conclusions about the ANN-to-SNN conversion would
# be based on bad numbers.
# ─────────────────────────────────────────────────────────────────

class TestEvaluationLogic:

    def test_ann_accuracy_range(self, ann):
        """
        Accuracy is the fraction of correct predictions, so it must always
        be between 0.0 (all wrong) and 1.0 (all correct). Even a randomly
        initialized untrained model should produce a value in this range —
        it would just be near 0.1 (random chance on 10 classes). A value
        outside [0, 1] would mean the accuracy formula itself is broken.
        """
        data   = torch.randn(16, 1, 28, 28)
        target = torch.randint(0, 10, (16,))  # random labels, 0–9
        ann.eval()
        with torch.no_grad():
            output = ann(data)
            pred   = output.argmax(dim=1)
            acc    = (pred == target).float().mean().item()
        assert 0.0 <= acc <= 1.0

    def test_snn_spike_count_scales_with_batch_size(self, default_snn):
        """
        Each image in a batch is processed independently by the SNN —
        there is no interaction between images. Therefore, processing the
        same image 8 times in a batch should produce exactly 8× the spikes
        of processing it once. This confirms that the spike counter is
        summing correctly across the batch dimension rather than averaging,
        taking a max, or otherwise mishandling batch aggregation.
        """
        single = torch.randn(1, 1, 28, 28)
        batch  = single.repeat(8, 1, 1, 1)  # stack the same image 8 times

        _, spikes_single = default_snn(single, num_steps=10)
        _, spikes_batch  = default_snn(batch,  num_steps=10)

        # The tolerance is small (1e-3) to allow for floating point rounding
        assert abs(spikes_batch - 8 * spikes_single) < 1e-3, (
            f"Expected {8 * spikes_single:.1f} spikes for batch of 8, "
            f"got {spikes_batch:.1f}"
        )

    def test_perfect_predictions_give_accuracy_one(self, ann):
        """
        If the model's own predictions are used as the targets, every
        prediction is trivially correct, so accuracy must be exactly 1.0.
        This tests the accuracy formula in a scenario where the answer
        is known with certainty — a basic sanity check.
        """
        data    = torch.randn(4, 1, 28, 28)
        output  = ann(data)
        targets = output.argmax(dim=1)  # use model's own choices as ground truth
        correct = (output.argmax(dim=1) == targets).sum().item()
        acc     = correct / len(targets)
        assert acc == 1.0

    def test_wrong_predictions_give_accuracy_zero(self):
        """
        If every prediction is wrong, accuracy must be exactly 0.0.
        Here we use predictions of class 0 against targets of class 1 —
        guaranteed mismatch. This tests the other boundary of the accuracy
        formula, complementing test_perfect_predictions_give_accuracy_one.
        """
        preds   = torch.zeros(10, dtype=torch.long)  # all predict class 0
        targets = torch.ones(10,  dtype=torch.long)  # all labeled class 1
        acc     = (preds == targets).float().mean().item()
        assert acc == 0.0


# ─────────────────────────────────────────────────────────────────
# GROUP 5: Calibration Scoring Logic
#
# The calibration search finds the best threshold values by scoring
# each candidate combination using a weighted objective function:
#
#     score = accuracy − λ × (spikes / spikes_default)
#
# accuracy: fraction of digits correctly classified (higher is better)
# spikes:   total spikes fired per image (lower is better — fewer = less energy)
# λ (lambda): controls how much we penalize spike count vs reward accuracy
#
# These tests verify that the scoring formula behaves correctly across
# a variety of scenarios, including edge cases.
# ─────────────────────────────────────────────────────────────────

class TestCalibrationScoring:

    def _score(self, accuracy, spikes, spikes_default, lam=0.3):
        """
        Local copy of the scoring formula from snn_pipeline.py.
        Defined as a helper so all tests in this group can use the same
        formula without repeating it. If the formula changes in the pipeline,
        update it here too.

        spikes_default is the baseline spike count (default thresholds = 1.0),
        used to normalize — so a score of 1.0 on the spike penalty means
        'same as the default', and >1.0 means 'worse than default'.
        """
        norm_spikes = spikes / max(spikes_default, 1)  # normalize against baseline
        return accuracy - lam * norm_spikes

    def test_higher_accuracy_gives_higher_score(self):
        """
        All else being equal, a more accurate model should score higher.
        This is the primary goal: we never want to trade accuracy for
        efficiency to the point where the model stops being useful.
        Here spike count is held constant so we're isolating the effect
        of accuracy on the score.
        """
        score_high = self._score(accuracy=0.98, spikes=500, spikes_default=500)
        score_low  = self._score(accuracy=0.90, spikes=500, spikes_default=500)
        assert score_high > score_low

    def test_fewer_spikes_gives_higher_score(self):
        """
        All else being equal, fewer spikes should give a higher score.
        Fewer spikes = less energy on the chip. This is the whole point
        of doing calibration rather than just accepting the default thresholds.
        Here accuracy is held constant so we're isolating the effect of
        spike count on the score.
        """
        score_efficient = self._score(accuracy=0.95, spikes=300, spikes_default=500)
        score_wasteful  = self._score(accuracy=0.95, spikes=700, spikes_default=500)
        assert score_efficient > score_wasteful

    def test_score_reflects_tradeoff(self):
        """
        The most interesting case: a model with slightly lower accuracy
        but much fewer spikes can actually outscore a more accurate model.
        This is the fundamental insight behind calibration — we're not
        just maximizing accuracy, we're finding the best balance between
        accuracy and energy efficiency.

        With λ=0.3: losing 3% accuracy is worth it if it cuts spike count
        by more than 10% of the baseline. The numbers here are chosen to
        make the efficient model win by a clear margin.
        """
        # 98% accuracy but 60% more spikes than baseline
        score_accurate  = self._score(accuracy=0.98, spikes=800, spikes_default=500)
        # 95% accuracy but 60% fewer spikes than baseline
        score_efficient = self._score(accuracy=0.95, spikes=200, spikes_default=500)
        assert score_efficient > score_accurate, (
            "Expected the efficient (low-spike) model to win with lambda=0.3"
        )

    def test_score_is_deterministic(self):
        """
        The scoring formula is pure arithmetic — no randomness, no state.
        The same inputs must always produce exactly the same score.
        If this fails, something non-deterministic (like a random number or
        a hash-based operation) has been introduced into the scoring.
        """
        s1 = self._score(accuracy=0.97, spikes=450, spikes_default=500)
        s2 = self._score(accuracy=0.97, spikes=450, spikes_default=500)
        assert s1 == s2

    def test_score_with_zero_spikes(self):
        """
        If a model produces zero spikes (no neurons ever fire), there is
        no energy cost at all — the spike penalty is zero. In this edge case,
        the score should equal the accuracy alone. This is an extreme but
        valid scenario: a model with very high thresholds might produce zero
        spikes on some inputs (and would also likely have terrible accuracy).
        """
        score = self._score(accuracy=0.95, spikes=0, spikes_default=500)
        # score should be exactly 0.95 - 0.3 * (0/500) = 0.95 - 0 = 0.95
        assert abs(score - 0.95) < 1e-9

    def test_lambda_zero_ignores_spikes(self):
        """
        If lambda is set to 0, the spike count has zero weight in the score —
        we're purely maximizing accuracy and ignoring energy cost entirely.
        In that limit, the score must equal the accuracy regardless of how
        many spikes were produced. This tests that lambda correctly controls
        the accuracy-vs-efficiency tradeoff rather than hardcoding any
        specific penalty value.
        """
        score = self._score(accuracy=0.93, spikes=9999, spikes_default=500, lam=0.0)
        # score = 0.93 - 0.0 * (9999/500) = 0.93 - 0 = 0.93
        assert abs(score - 0.93) < 1e-9

    def test_best_threshold_has_highest_score(self):
        """
        Simulates the final step of the calibration search: given a list of
        evaluated threshold combinations with their scores, the one with the
        highest score should be selected as the winner.

        In the real pipeline this is done with Python's max() and a lambda
        key. This test verifies that selection logic picks the correct entry,
        catching any bug where the wrong metric is being compared (e.g., sorting
        by accuracy instead of score, or finding the minimum instead of maximum).
        """
        results = [
            {'thresholds': [1.0,  1.0, 1.0, 1.0],  'score': 0.62},
            {'thresholds': [0.75, 1.0, 1.5, 0.75], 'score': 0.71},  # should win
            {'thresholds': [1.25, 1.0, 1.0, 1.0],  'score': 0.58},
        ]
        best = max(results, key=lambda r: r['score'])

        assert best['thresholds'] == [0.75, 1.0, 1.5, 0.75], (
            "Wrong threshold combination selected as best"
        )
        assert best['score'] == 0.71, (
            "Best score value is incorrect"
        )