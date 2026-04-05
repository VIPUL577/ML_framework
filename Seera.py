from Seera_Engine import Tensor, np, autograd4nn
import matplotlib.pyplot as plt
import pickle

# ─────────────────────────────────────────────────────────────
# Base Layer
# ─────────────────────────────────────────────────────────────
class Layer:
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.activations = {
            "relu": Tensor.relu,
            "sigmoid": Tensor.sigmoid,
            "softmax": Tensor.softmax,
            "tanh": Tensor.tanh,
        }


# ─────────────────────────────────────────────────────────────
# Input Layer
# ─────────────────────────────────────────────────────────────
class Input(Layer):
    def __init__(self, shape):
        """shape: per-sample shape, e.g. (3,) for 3 features or (1, 28, 28) for images.
        During forward, input tensor has leading batch dim: (N, *shape)."""
        super().__init__()
        self.shape = shape
        X = Tensor.random((1, *shape) if isinstance(shape, tuple) else (1, shape))
        self.weights = 0
        self.bais = 0
        self.inputs = X

    def forward(self):
        self.outputs = self.inputs
        return self.outputs

    def __repr__(self):
        return f"Input Layer with shape {self.shape}"


# ─────────────────────────────────────────────────────────────
# Dense Layer (batched row-vector convention)
# ─────────────────────────────────────────────────────────────
class Dense(Layer):
    def __init__(
        self,
        in_units,
        out_units,
        activation=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
    ):
        super().__init__()
        initializers = {
            "zeros": 0,
            "ones": 1,
            "random_normal": 1,
            "random_uniform": 0.05,
            "he_normal": np.sqrt(2 / in_units) * 0.1,
            "he_uniform": np.sqrt(6 / in_units) * 0.1,
            "glorot_normal": np.sqrt(2 / (in_units + out_units)) * 0.1,
            "glorot_uniform": np.sqrt(6 / (in_units + out_units)) * 0.1,
            "lecun_normal": np.sqrt(1 / in_units) * 0.1,
            "lecun_uniform": np.sqrt(3 / in_units) * 0.1,
        }

        if activation not in self.activations:
            raise ValueError('Activation must be "relu", "sigmoid", "softmax", or "tanh"')
        if kernel_initializer not in initializers:
            raise ValueError(f"Initializer must be one of {list(initializers.keys())}")

        self.layeract = activation
        self.in_units = in_units
        self.out_units = out_units
        # Weights: (in_units, out_units) — row-vector convention
        self.weights = Tensor.random((in_units, out_units)) * initializers[kernel_initializer]
        # Bias: (1, out_units) — broadcasts over batch
        self.bais = Tensor.random((1, out_units)) * initializers[bias_initializer]

    def __call__(self, input_layer):
        if not isinstance(input_layer, Layer):
            raise TypeError("The input should be a Layer")
        self.inpobj = input_layer
        return self

    def forward(self):
        self.inputs = self.inpobj.outputs
        # x: (N, in_units) @ W: (in_units, out_units) + b: (1, out_units) → (N, out_units)
        self.z = self.inputs.matmul(self.weights) + self.bais
        self.outputs = self.activations[self.layeract](self.z)
        return self.outputs

    def update_params(self, vW, vB):
        self.weights.value -= vW
        self.bais.value -= vB

    def get_weights(self):
        return self.weights, self.bais

    def set_weights(self, W, B):
        if isinstance(W, Tensor):
            self.weights = W
            self.bais = B
        elif isinstance(W, np.ndarray):
            self.weights = Tensor(W, is_leaf=True)
            self.bais = Tensor(B, is_leaf=True)
        else:
            raise TypeError("Weights must be Tensor or numpy array")

    def __repr__(self):
        total_params = self.in_units * self.out_units + self.out_units
        return f"Dense({self.in_units}→{self.out_units}, act={self.layeract}, params={total_params})"


# ─────────────────────────────────────────────────────────────
# Conv2D Layer (batched)
# ─────────────────────────────────────────────────────────────
class Conv2D(Layer):
    def __init__(
        self,
        out_channels,
        in_channels,
        kernel_size,
        activation,
        stride=1,
        zero_padding=0,
        kernel_initializer="he_normal",
        bias_initializer="zeros",
    ):
        super().__init__()
        initializers = {
            "zeros": 0,
            "ones": 1,
            "random_normal": 1,
            "random_uniform": 0.05,
            "he_normal": np.sqrt(2 / (in_channels * kernel_size[0] * kernel_size[1])) * 0.1,
            "he_uniform": np.sqrt(6 / (in_channels * kernel_size[0] * kernel_size[1])) * 0.1,
            "glorot_normal": np.sqrt(2 / ((in_channels + out_channels) * kernel_size[0] * kernel_size[1])) * 0.1,
            "glorot_uniform": np.sqrt(6 / ((in_channels + out_channels) * kernel_size[0] * kernel_size[1])) * 0.1,
            "lecun_normal": np.sqrt(1 / (in_channels * kernel_size[0] * kernel_size[1])) * 0.1,
            "lecun_uniform": np.sqrt(3 / (in_channels * kernel_size[0] * kernel_size[1])) * 0.1,
        }

        if activation not in self.activations:
            raise ValueError('Activation must be "relu", "sigmoid", "softmax", or "tanh"')
        if kernel_initializer not in initializers:
            raise ValueError(f"Initializer must be one of {list(initializers.keys())}")

        # Normalize stride/padding to (h, w) tuples
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(zero_padding, int):
            zero_padding = (zero_padding, zero_padding)

        self.layeract = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride          # (strideh, stridew)
        self.zero_padding = zero_padding  # (padh, padw)
        # Bias: (1, F, 1, 1) broadcasts over (N, F, OH, OW)
        self.bais = 0
        self.weights = Tensor(
            np.random.normal(
                loc=0, scale=0.5,
                size=(out_channels, in_channels, kernel_size[0], kernel_size[1]),
            ),
            is_leaf=True,
        ) * initializers[kernel_initializer]

    def __call__(self, input_layer):
        if not isinstance(input_layer, Layer):
            raise TypeError("The input should be a Layer")
        self.inpobj = input_layer
        return self

    def forward(self):
        self.inputs = self.inpobj.outputs
        inp = self.inputs.value
        if inp.ndim == 3:
            inp = inp[np.newaxis]

        N, C, H, W_in = inp.shape
        strideh, stridew = self.stride
        padh, padw = self.zero_padding
        OH = (H - self.kernel_size[0] + 2 * padh) // strideh + 1
        OW = (W_in - self.kernel_size[1] + 2 * padw) // stridew + 1

        # Lazy-init bias with correct spatial dims
        if isinstance(self.bais, int) or self.bais.shape != (1, self.out_channels, 1, 1):
            self.bais = Tensor.zeros((1, self.out_channels, 1, 1))

        self.z = Tensor.conv2d(
            self.inputs, self.weights, stride=self.stride, padding=self.zero_padding
        ) + self.bais
        self.outputs = self.activations[self.layeract](self.z)
        return self.outputs

    def set_weights(self, W, B):
        if isinstance(W, np.ndarray):
            self.weights = Tensor(W, is_leaf=True)
            self.bais = Tensor(B, is_leaf=True)
        elif isinstance(W, Tensor):
            self.weights = W
            self.bais = B

    def update_params(self, vW, vB):
        self.weights.value -= vW
        self.bais.value -= vB

    def get_weights(self):
        return self.weights, self.bais

    def __repr__(self):
        return (
            f"Conv2D({self.in_channels}→{self.out_channels}, "
            f"kernel={self.kernel_size}, act={self.layeract})"
        )


# ─────────────────────────────────────────────────────────────
# Flatten Layer (batch-preserving)
# ─────────────────────────────────────────────────────────────
class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, input_layer):
        if not isinstance(input_layer, Layer):
            raise TypeError("Input must be a Layer instance.")
        self.inpobj = input_layer
        return self

    def forward(self):
        self.inputs = self.inpobj.outputs
        self.outputs = Tensor.flatten(self.inputs)
        return self.outputs

    def __repr__(self):
        return "Flatten Layer"


# ─────────────────────────────────────────────────────────────
# Unpool2D_Nearest (nearest-neighbor unpooling, no learnable params)
# ─────────────────────────────────────────────────────────────
class Unpool2D_Nearest(Layer):
    def __init__(self, size=(2, 2)):
        super().__init__()
        self.size = size

    def __call__(self, input_layer):
        if not isinstance(input_layer, Layer):
            raise TypeError("Input must be a Layer instance.")
        self.inpobj = input_layer
        return self

    def forward(self):
        self.inputs = self.inpobj.outputs
        self.outputs = Tensor.Unpool2Dnearest(self.inputs, size=self.size)
        return self.outputs

    def __repr__(self):
        return f"Unpool2D_Nearest(size={self.size})"


# ─────────────────────────────────────────────────────────────
# ConvTranspose2D (Transposed Convolution — learnable upsampling)
# ─────────────────────────────────────────────────────────────
class ConvTranspose2D(Layer):
    def __init__(
        self,
        out_channels,
        in_channels,
        kernel_size,
        activation,
        stride=1,
        zero_padding=0,
        kernel_initializer="he_normal",
    ):
        super().__init__()
        initializers = {
            "zeros": 0,
            "ones": 1,
            "random_normal": 1,
            "random_uniform": 0.05,
            "he_normal": np.sqrt(2 / (in_channels * kernel_size[0] * kernel_size[1])) * 0.1,
            "he_uniform": np.sqrt(6 / (in_channels * kernel_size[0] * kernel_size[1])) * 0.1,
            "glorot_normal": np.sqrt(2 / ((in_channels + out_channels) * kernel_size[0] * kernel_size[1])) * 0.1,
            "glorot_uniform": np.sqrt(6 / ((in_channels + out_channels) * kernel_size[0] * kernel_size[1])) * 0.1,
        }

        if activation not in self.activations:
            raise ValueError('Activation must be "relu", "sigmoid", "softmax", or "tanh"')
        if kernel_initializer not in initializers:
            raise ValueError(f"Initializer must be one of {list(initializers.keys())}")

        # Normalize stride/padding to (h, w) tuples
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(zero_padding, int):
            zero_padding = (zero_padding, zero_padding)

        self.layeract = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride          # (strideh, stridew)
        self.zero_padding = zero_padding  # (padh, padw)
        # Weight: (Cin, Cout, KH, KW)
        self.weights = Tensor(
            np.random.normal(
                loc=0, scale=0.5,
                size=(in_channels, out_channels, kernel_size[0], kernel_size[1]),
            ),
            is_leaf=True,
        ) * initializers[kernel_initializer]
        # Bias: (1, Cout, 1, 1) broadcasts over (N, Cout, Hout, Wout)
        self.bais = 0

    def __call__(self, input_layer):
        if not isinstance(input_layer, Layer):
            raise TypeError("The input should be a Layer")
        self.inpobj = input_layer
        return self

    def forward(self):
        self.inputs = self.inpobj.outputs
        inp = self.inputs.value
        if inp.ndim == 3:
            inp = inp[np.newaxis]

        N, Cin, H, W_in = inp.shape
        strideh, stridew = self.stride
        padh, padw = self.zero_padding
        Hout = (H - 1) * strideh - 2 * padh + self.kernel_size[0]
        Wout = (W_in - 1) * stridew - 2 * padw + self.kernel_size[1]

        # Lazy-init bias
        if isinstance(self.bais, int) or self.bais.shape != (1, self.out_channels, 1, 1):
            self.bais = Tensor.zeros((1, self.out_channels, 1, 1))

        self.z = Tensor.conv_transpose2d(
            self.inputs, self.weights,
            stride=self.stride, padding=self.zero_padding,
        ) + self.bais
        self.outputs = self.activations[self.layeract](self.z)
        return self.outputs

    def set_weights(self, W, B):
        if isinstance(W, np.ndarray):
            self.weights = Tensor(W, is_leaf=True)
            self.bais = Tensor(B, is_leaf=True)
        elif isinstance(W, Tensor):
            self.weights = W
            self.bais = B

    def update_params(self, vW, vB):
        self.weights.value -= vW
        self.bais.value -= vB

    def get_weights(self):
        return self.weights, self.bais

    def __repr__(self):
        return (
            f"ConvTranspose2D({self.in_channels}→{self.out_channels}, "
            f"kernel={self.kernel_size}, stride={self.stride}, act={self.layeract})"
        )


# ─────────────────────────────────────────────────────────────
# MaxPool2D (batched)
# ─────────────────────────────────────────────────────────────
class MaxPool2D(Layer):
    def __init__(self, pool_size=(2, 2), stride=1, padding=0):
        super().__init__()
        self.pool_size = pool_size
        # Normalize stride/padding to (h, w) tuples
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.stride = stride      # (strideh, stridew)
        self.padding = padding    # (padh, padw)

    def __call__(self, input_layer):
        if not isinstance(input_layer, Layer):
            raise TypeError("Input must be a Layer instance.")
        self.inpobj = input_layer
        return self

    def forward(self):
        self.inputs = self.inpobj.outputs
        self.outputs = Tensor.maxpool2d(
            self.inputs, kernelsize=self.pool_size,
            stride=self.stride, padding=self.padding,
        )
        return self.outputs

    def __repr__(self):
        return f"MaxPool2D(pool={self.pool_size}, stride={self.stride})"


# ─────────────────────────────────────────────────────────────
# Concatenate (batched, along channels)
# ─────────────────────────────────────────────────────────────
class Concatenate(Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, *input_layers):
        if len(input_layers) != 2:
            raise ValueError("Only 2 layers supported for concatenation.")
        for layer in input_layers:
            if not isinstance(layer, Layer):
                raise TypeError("All inputs must be Layer instances.")
        self.input_layers = input_layers
        return self

    def forward(self):
        self.inputs = [layer.outputs for layer in self.input_layers]
        self.outputs = Tensor.concatenete(self.inputs[0], self.inputs[1])
        return self.outputs

    def __repr__(self):
        return "Concatenate Layer"


# ─────────────────────────────────────────────────────────────
# BatchNorm1d  (for Dense layers)
# ─────────────────────────────────────────────────────────────
class BatchNorm1d(Layer):
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        """Batch normalization for Dense layers.
        Input: (N, num_features)   Output: (N, num_features)
        """
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.training = True

        # Learnable parameters
        self.gamma = Tensor.ones((num_features,))
        self.beta = Tensor.zeros((num_features,))

        # Running statistics (plain numpy, not part of comp graph)
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)

        # For param update interface
        self.weights = self.gamma
        self.bais = self.beta

    def __call__(self, input_layer):
        if not isinstance(input_layer, Layer):
            raise TypeError("Input must be a Layer instance.")
        self.inpobj = input_layer
        return self

    def forward(self):
        self.inputs = self.inpobj.outputs
        self.outputs = Tensor.batchnorm(
            self.inputs, self.gamma, self.beta,
            self.running_mean, self.running_var,
            training=self.training, momentum=self.momentum,
            eps=self.eps, mode="1d",
        )
        return self.outputs

    def update_params(self, vGamma, vBeta):
        self.gamma.value -= vGamma
        self.beta.value -= vBeta

    def get_weights(self):
        return self.gamma, self.beta

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __repr__(self):
        return f"BatchNorm1d({self.num_features})"


# ─────────────────────────────────────────────────────────────
# BatchNorm2d  (for Conv layers)
# ─────────────────────────────────────────────────────────────
class BatchNorm2d(Layer):
    def __init__(self, num_channels, momentum=0.1, eps=1e-5):
        """Batch normalization for Conv layers.
        Input: (N, C, H, W)   Output: (N, C, H, W)
        """
        super().__init__()
        self.num_channels = num_channels
        self.momentum = momentum
        self.eps = eps
        self.training = True

        # Learnable parameters (per channel)
        self.gamma = Tensor.ones((num_channels,))
        self.beta = Tensor.zeros((num_channels,))

        # Running statistics
        self.running_mean = np.zeros(num_channels, dtype=np.float32)
        self.running_var = np.ones(num_channels, dtype=np.float32)

        # For param update interface
        self.weights = self.gamma
        self.bais = self.beta

    def __call__(self, input_layer):
        if not isinstance(input_layer, Layer):
            raise TypeError("Input must be a Layer instance.")
        self.inpobj = input_layer
        return self

    def forward(self):
        self.inputs = self.inpobj.outputs
        self.outputs = Tensor.batchnorm(
            self.inputs, self.gamma, self.beta,
            self.running_mean, self.running_var,
            training=self.training, momentum=self.momentum,
            eps=self.eps, mode="2d",
        )
        return self.outputs

    def update_params(self, vGamma, vBeta):
        self.gamma.value -= vGamma
        self.beta.value -= vBeta

    def get_weights(self):
        return self.gamma, self.beta

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __repr__(self):
        return f"BatchNorm2d({self.num_channels})"


# ─────────────────────────────────────────────────────────────
# Sequential Model (with batch training support)
# ─────────────────────────────────────────────────────────────
class Sequential:
    def __init__(self, layers):
        if not isinstance(layers, list):
            raise TypeError("Layers must be a list")
        if not isinstance(layers[0], Input):
            raise TypeError("First layer must be an Input layer")
        layerlis = [layers[0]]
        for i, layer in enumerate(layers):
            if i == 0:
                continue
            layerlis.append(layer(layers[i - 1]))
        self.model = layerlis

    def forward(self, X=None):
        if X is not None:
            if not isinstance(X, Tensor):
                raise TypeError("Input must be a Tensor.")
            self.model[0].inputs = X
        output = None
        for layer in self.model:
            output = layer.forward()
        return output

    def predict(self, X):
        """Run forward in eval mode (disables BatchNorm training stats)."""
        self._set_mode(training=False)
        out = self.forward(X)
        self._set_mode(training=True)
        return out

    def _set_mode(self, training=True):
        for layer in self.model:
            if hasattr(layer, "training"):
                layer.training = training

    def zero_grad(self):
        """Reset all parameter gradients to zero."""
        for layer in self.model:
            if hasattr(layer, "update_params"):
                layer.weights.node.cp = np.zeros_like(layer.weights.value)
                layer.bais.node.cp = np.zeros_like(layer.bais.value)

    def get_params(self):
        for layer in self.model:
            if hasattr(layer, "update_params"):
                print(layer.get_weights())

    def summary(self):
        print("=" * 60)
        print("Model Summary")
        print("=" * 60)
        for idx, layer in enumerate(self.model):
            print(f"  Layer {idx}: {layer}")
        print("=" * 60)

    def fit(self, X_train, y, Optimizer, Loss, Epochs, batch_size=1, Loss_interval=20):
        """Train the model.
        X_train: numpy array (num_samples, *sample_shape)
        y:       numpy array (num_samples, *target_shape)
        batch_size: number of samples per batch (default 1 for backward compat)
        """
        history = []
        num_samples = X_train.shape[0]

        for epoch in range(Epochs):
            # Shuffle data at each epoch
            perm = np.random.permutation(num_samples)
            X_shuffled = X_train[perm]
            y_shuffled = y[perm]

            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                X_batch = Tensor(X_shuffled[start:end], is_leaf=True)
                y_batch = Tensor(y_shuffled[start:end])

                # Forward
                ypred = self.forward(X_batch)

                # Compute loss
                loss = Loss(ypred, y_batch)

                # For batch: take mean loss across batch for single scalar
                if loss.value.ndim > 0 and loss.value.size > 1:
                    loss = loss.mean()

                batch_loss = float(np.sum(loss.value))
                epoch_loss += batch_loss
                num_batches += 1

                # Backward
                self.zero_grad()
                autograd4nn(loss)

                # Optimizer step
                Optimizer.step()

            avg_loss = epoch_loss / num_batches
            history.append(avg_loss)
            if epoch % Loss_interval == 0 or epoch == Epochs - 1:
                print(f"Epoch {epoch+1}/{Epochs} — Loss: {avg_loss:.6f}")

        return np.array(history)

    def save(self, filepath):
        model_data = []
        for layer in self.model:
            layer_info = {
                "class": layer.__class__.__name__,
                "config": {},
            }
            # Save relevant config
            if isinstance(layer, Input):
                layer_info["config"]["shape"] = layer.shape
            elif isinstance(layer, Dense):
                layer_info["config"] = {
                    "in_units": layer.in_units,
                    "out_units": layer.out_units,
                    "layeract": layer.layeract,
                }
            elif isinstance(layer, Conv2D):
                layer_info["config"] = {
                    "out_channels": layer.out_channels,
                    "in_channels": layer.in_channels,
                    "kernel_size": layer.kernel_size,
                    "layeract": layer.layeract,
                    "stride": layer.stride,
                    "zero_padding": layer.zero_padding,
                }
            elif isinstance(layer, BatchNorm1d):
                layer_info["config"] = {
                    "num_features": layer.num_features,
                    "momentum": layer.momentum,
                    "eps": layer.eps,
                    "running_mean": layer.running_mean.copy(),
                    "running_var": layer.running_var.copy(),
                }
            elif isinstance(layer, BatchNorm2d):
                layer_info["config"] = {
                    "num_channels": layer.num_channels,
                    "momentum": layer.momentum,
                    "eps": layer.eps,
                    "running_mean": layer.running_mean.copy(),
                    "running_var": layer.running_var.copy(),
                }
            elif isinstance(layer, MaxPool2D):
                layer_info["config"] = {
                    "pool_size": layer.pool_size,
                    "stride": layer.stride,
                    "padding": layer.padding,
                }
            elif isinstance(layer, Unpool2D_Nearest):
                layer_info["config"] = {"size": layer.size}
            elif isinstance(layer, ConvTranspose2D):
                layer_info["config"] = {
                    "out_channels": layer.out_channels,
                    "in_channels": layer.in_channels,
                    "kernel_size": layer.kernel_size,
                    "layeract": layer.layeract,
                    "stride": layer.stride,
                    "zero_padding": layer.zero_padding,
                }

            if hasattr(layer, "get_weights"):
                w, b = layer.get_weights()
                layer_info["weights"] = w.value if hasattr(w, "value") else w
                layer_info["bais"] = b.value if hasattr(b, "value") else b

            model_data.append(layer_info)

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        layers = []
        for i, info in enumerate(model_data):
            class_name = info["class"]
            config = info["config"]

            if class_name == "Input":
                layer = Input(config["shape"])
            elif class_name == "Dense":
                layer = Dense(config["in_units"], config["out_units"], activation=config["layeract"])
                if "weights" in info:
                    layer.set_weights(info["weights"], info["bais"])
            elif class_name == "Conv2D":
                layer = Conv2D(
                    config["out_channels"], config["in_channels"],
                    config["kernel_size"], activation=config["layeract"],
                    stride=config.get("stride", 1),
                    zero_padding=config.get("zero_padding", 0),
                )
                if "weights" in info:
                    layer.set_weights(info["weights"], info["bais"])
            elif class_name == "BatchNorm1d":
                layer = BatchNorm1d(config["num_features"], config["momentum"], config["eps"])
                layer.running_mean = config["running_mean"]
                layer.running_var = config["running_var"]
                if "weights" in info:
                    layer.gamma = Tensor(info["weights"], is_leaf=True)
                    layer.beta = Tensor(info["bais"], is_leaf=True)
                    layer.weights = layer.gamma
                    layer.bais = layer.beta
            elif class_name == "BatchNorm2d":
                layer = BatchNorm2d(config["num_channels"], config["momentum"], config["eps"])
                layer.running_mean = config["running_mean"]
                layer.running_var = config["running_var"]
                if "weights" in info:
                    layer.gamma = Tensor(info["weights"], is_leaf=True)
                    layer.beta = Tensor(info["bais"], is_leaf=True)
                    layer.weights = layer.gamma
                    layer.bais = layer.beta
            elif class_name == "Flatten":
                layer = Flatten()
            elif class_name == "Unpool2D_Nearest" or class_name == "UpSampling2D_Nearest":
                layer = Unpool2D_Nearest(size=config["size"])
            elif class_name == "ConvTranspose2D":
                layer = ConvTranspose2D(
                    config["out_channels"], config["in_channels"],
                    config["kernel_size"], activation=config["layeract"],
                    stride=config.get("stride", 1),
                    zero_padding=config.get("zero_padding", 0),
                )
                if "weights" in info:
                    layer.set_weights(info["weights"], info["bais"])
            elif class_name == "MaxPool2D":
                layer = MaxPool2D(
                    pool_size=config["pool_size"],
                    stride=config["stride"],
                    padding=config["padding"],
                )
            elif class_name == "Concatenate":
                layer = Concatenate()
            else:
                raise ValueError(f"Unknown layer type: {class_name}")

            layers.append(layer)

        # Reconnect layers
        for i in range(1, len(layers)):
            if hasattr(layers[i], "inpobj"):
                layers[i].inpobj = layers[i - 1]

        return cls(layers)


# ─────────────────────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────────────────────
class Loss:
    def mse(self, y_pred, y):
        """Mean Squared Error: mean((y_pred - y)^2)  over all elements."""
        return ((y_pred - y) ** 2).mean()

    def mae(self, y_pred, y):
        """Mean Absolute Error."""
        return (y_pred - y).abs().mean()

    def binary_cross_entropy(self, y_pred, y, epsilon=1e-15):
        """Binary Cross-Entropy (element-wise, averaged over batch)."""
        return (-y * (y_pred + epsilon).log() - (1 - y) * ((1 - y_pred + epsilon).log())).mean()

    def categorical_cross_entropy(self, y_pred, y, epsilon=1e-15):
        """Categorical Cross-Entropy (averaged over batch).
        y_pred: (N, C)  y: (N, C) one-hot"""
        return (-y * ((y_pred + epsilon).log())).sum(axis=-1).mean()


# ─────────────────────────────────────────────────────────────
# SGD Optimizer
# ─────────────────────────────────────────────────────────────
class SGD:
    def __init__(self, model, lr, momentum=0):
        if not isinstance(model, Sequential):
            raise TypeError("Model should be Sequential")
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}
        for layer in self.model.model:
            self.velocity[layer] = [0, 0]

    def step(self):
        for layer in self.model.model:
            if hasattr(layer, "update_params"):
                g0 = layer.weights.node.cp
                g1 = layer.bais.node.cp
                self.velocity[layer][0] = self.momentum * self.velocity[layer][0] + self.lr * g0
                self.velocity[layer][1] = self.momentum * self.velocity[layer][1] + self.lr * g1
                layer.update_params(self.velocity[layer][0], self.velocity[layer][1])


# ─────────────────────────────────────────────────────────────
# Adam Optimizer
# ─────────────────────────────────────────────────────────────
class Adam:
    def __init__(self, model, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        if not isinstance(model, Sequential):
            raise TypeError("Model should be Sequential")
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 1

        self.m = {}
        self.v = {}
        for layer in self.model.model:
            self.m[layer] = [0, 0]
            self.v[layer] = [0, 0]

    def step(self):
        for layer in self.model.model:
            if hasattr(layer, "update_params"):
                g0 = layer.weights.node.cp
                g1 = layer.bais.node.cp

                self.m[layer][0] = self.beta1 * self.m[layer][0] + (1 - self.beta1) * g0
                self.m[layer][1] = self.beta1 * self.m[layer][1] + (1 - self.beta1) * g1

                self.v[layer][0] = self.beta2 * self.v[layer][0] + (1 - self.beta2) * (g0 ** 2)
                self.v[layer][1] = self.beta2 * self.v[layer][1] + (1 - self.beta2) * (g1 ** 2)

                m_hat0 = self.m[layer][0] / (1 - self.beta1 ** self.t)
                m_hat1 = self.m[layer][1] / (1 - self.beta1 ** self.t)

                v_hat0 = self.v[layer][0] / (1 - self.beta2 ** self.t)
                v_hat1 = self.v[layer][1] / (1 - self.beta2 ** self.t)

                update_w = (self.lr * m_hat0) / (v_hat0 ** 0.5 + self.eps)
                update_b = (self.lr * m_hat1) / (v_hat1 ** 0.5 + self.eps)

                layer.update_params(update_w, update_b)

        self.t += 1
