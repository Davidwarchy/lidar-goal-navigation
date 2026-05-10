import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod

# ---------------------------------------------------------------------------
# Abstract Neural Network Interface
# ---------------------------------------------------------------------------

class NeuralNetwork(ABC):
    """Abstract base class for neural networks used in neuroevolution."""
    
    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Process inputs and return output activations."""
        pass
    
    @abstractmethod
    def reset_state(self):
        """Reset any internal state (e.g., for RNNs/SNNs)."""
        pass
    
    @abstractmethod
    def get_weights(self) -> torch.Tensor:
        """Get flat array of all trainable weights for all agents."""
        pass
    
    @abstractmethod
    def set_weights(self, weights: torch.Tensor):
        """Set weights from flat array."""
        pass
    
    @property
    @abstractmethod
    def input_size(self) -> int:
        """Return input dimension."""
        pass
    
    @property
    @abstractmethod
    def output_size(self) -> int:
        """Return output dimension."""
        pass


# ---------------------------------------------------------------------------
# Vectorized Feedforward Neural Network with PyTorch
# ---------------------------------------------------------------------------

class VectorFeedforwardNetwork(NeuralNetwork):
    """
    Vectorized standard multi-layer perceptron with ReLU activations using PyTorch.
    
    Architecture:
    Input -> Hidden Layer(s) (ReLU) -> Output Layer (Linear)
    
    All agents are evaluated in parallel using vectorized operations.
    """
    
    def __init__(self, num_envs: int, input_size: int, hidden_sizes: list, output_size: int, device='cuda'):
        """
        Parameters:
        -----------
        num_envs : int
            Number of parallel agents
        input_size : int
            Dimensionality of input observations
        hidden_sizes : list of int
            Number of neurons in each hidden layer (e.g., [64, 32])
        output_size : int
            Number of output neurons (actions)
        device : str
            Device to run on ('cuda' or 'cpu')
        """
        self._num_envs = num_envs
        self._input_size = input_size
        self._output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        
        # Build layer sizes
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        # Initialize weights and biases for all agents as PyTorch parameters
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        
        for i in range(len(layer_sizes) - 1):
            # He initialization for ReLU
            # Shape: (num_envs, layer_sizes[i], layer_sizes[i+1])
            w = torch.randn(num_envs, layer_sizes[i], layer_sizes[i+1], device=self.device) * np.sqrt(2.0 / layer_sizes[i])
            b = torch.zeros(num_envs, layer_sizes[i+1], device=self.device)
            self.weights.append(nn.Parameter(w))
            self.biases.append(nn.Parameter(b))

        self._total_params = sum(
            layer_in * layer_out + layer_out
            for layer_in, layer_out in zip([input_size] + hidden_sizes, hidden_sizes + [output_size])
        )
    
    @property
    def input_size(self) -> int:
        return self._input_size
    
    @property
    def output_size(self) -> int:
        return self._output_size
    
    @property
    def num_envs(self) -> int:
        return self._num_envs
    
    def to(self, device):
        """Move network to specified device."""
        self.device = device
        for i in range(len(self.weights)):
            self.weights[i] = nn.Parameter(self.weights[i].to(device))
            self.biases[i] = nn.Parameter(self.biases[i].to(device))
        return self
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for all agents.
        
        Parameters:
        -----------
        inputs : (num_envs, input_size) tensor on device
        
        Returns:
        --------
        output : (num_envs, output_size) tensor
        """
        x = inputs
        for i in range(len(self.weights) - 1):
            # x: (num_envs, in_i) -> (num_envs, 1, in_i) @ (num_envs, in_i, out_i) -> (num_envs, 1, out_i) -> (num_envs, out_i)
            x = F.relu(torch.bmm(x.unsqueeze(1), self.weights[i]).squeeze(1) + self.biases[i])
        # Output layer (linear)
        x = torch.bmm(x.unsqueeze(1), self.weights[-1]).squeeze(1) + self.biases[-1]
        return x
        
    def reset_state(self):
        """Feedforward networks have no state to reset."""
        pass
    
    def get_weights(self) -> torch.Tensor:
        """Return (num_envs, total_params) tensor."""
        parts = []
        for w, b in zip(self.weights, self.biases):
            # w: (num_envs, in, out) -> flatten last two dims
            parts.append(w.reshape(self.num_envs, -1))
            parts.append(b)                     # (num_envs, out)
        return torch.cat(parts, dim=1)
    
    def set_weights(self, weights: torch.Tensor):
        """Assign from (num_envs, total_params) tensor."""
        start = 0
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            w_size = w.shape[1] * w.shape[2]
            b_size = b.shape[1]

            # Weight matrix
            w_flat = weights[:, start:start + w_size]
            self.weights[i] = nn.Parameter(w_flat.reshape(self.num_envs, w.shape[1], w.shape[2]))
            start += w_size

            # Bias vector
            self.biases[i] = nn.Parameter(weights[:, start:start + b_size])
            start += b_size


# ---------------------------------------------------------------------------
# Vectorized LIF Layer with PyTorch
# ---------------------------------------------------------------------------

class VectorLIFLayer(nn.Module):
    def __init__(self, num_envs, n_in, n_out, leak=0.9, threshold=1.0, device='cuda'):
        super().__init__()
        self.num_envs = num_envs
        self.n_in = n_in
        self.n_out = n_out
        self.leak = leak
        self.threshold = threshold
        self.device = device
        
        # Unique weights for every robot: (num_envs, n_in, n_out)
        limit = np.sqrt(2.0 / n_in)
        self.W = nn.Parameter(torch.randn(num_envs, n_in, n_out, device=device) * limit)
        self.b = nn.Parameter(torch.zeros(num_envs, n_out, device=device))
        self.register_buffer('V', torch.zeros(num_envs, n_out, device=device))

        self._total_params = n_in * n_out + n_out

    def reset_state(self):
        self.V.zero_()

    def forward(self, x):
        # x shape: (num_envs, n_in)
        # Vectorized batch matrix multiplication for unique weights per env
        # x: (num_envs, n_in)
        z = torch.bmm(x.unsqueeze(1), self.W).squeeze(1) + self.b
        self.V = self.leak * self.V + z
        spikes = (self.V >= self.threshold).float()
        self.V[spikes == 1.0] = 0.0
        return spikes

    def get_weights(self):
        """Return (num_envs, total_params) tensor."""
        w_flat = self.W.reshape(self.num_envs, -1)          # (num_envs, n_in*n_out)
        return torch.cat([w_flat, self.b], dim=1)           # (num_envs, total_params)

    def set_weights(self, weights):
        """Set from (num_envs, total_params) tensor."""
        w_size = self.n_in * self.n_out
        with torch.no_grad():
            self.W.copy_(weights[:, :w_size].reshape(self.num_envs, self.n_in, self.n_out))
            self.b.copy_(weights[:, w_size:])


# ---------------------------------------------------------------------------
# Vectorized Spiking Neural Network with PyTorch
# ---------------------------------------------------------------------------

class VectorSpikingNetwork(NeuralNetwork):
    """
    Multi‑layer vectorized Leaky Integrate‑and‑Fire Spiking Neural Network using PyTorch.
    
    Architecture:
    Input (rate‑coded) → LIF hidden layer(s) → ... → LIF output layer (binary spikes → argmax action)

    The network is run for `n_steps` internal time‑steps per call so that spiking dynamics
    can accumulate meaningful activity from a single observation vector.
    """
    
    def __init__(self, num_envs, input_size=100, hidden_sizes=[64], output_size=4, n_steps=5, device='cuda'):
        """
        Parameters
        ----------
        num_envs : int
            Number of parallel agents
        input_size : int
            Input dimension (number of lidar rays)
        hidden_sizes : list of int
            Number of neurons in each hidden LIF layer (e.g., [64] or [64, 32])
        output_size : int
            Number of output neurons (actions)
        n_steps : int
            Number of internal simulation steps per forward call
        device : str
            Device to run on ('cuda' or 'cpu')
        """
        self._num_envs = num_envs
        self._input_size = input_size
        self.hidden_sizes = hidden_sizes if isinstance(hidden_sizes, list) else [hidden_sizes]
        self._output_size = output_size
        self.n_steps = n_steps
        self.device = device
        
        # Build layers: input -> hidden layer 1 -> hidden layer 2 -> ... -> output
        self.layers = nn.ModuleList()
        prev_size = input_size
        for hsize in self.hidden_sizes:
            self.layers.append(VectorLIFLayer(num_envs, prev_size, hsize, device=device))
            prev_size = hsize
        # Output layer (action neurons)
        self.layers.append(VectorLIFLayer(num_envs, prev_size, output_size, device=device))
    
    @property
    def input_size(self) -> int:
        return self._input_size
    
    @property
    def output_size(self) -> int:
        return self._output_size
    
    @property
    def num_envs(self) -> int:
        return self._num_envs

    def to(self, device):
        """Move network to specified device."""
        self.device = device
        for layer in self.layers:
            layer.to(device)
        return self

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Run the SNN for `n_steps` time‑steps and return accumulated output spike counts.

        Parameters
        ----------
        obs : (num_envs, input_size) tensor on device, values in [0, 1]
              Treated as a constant Poisson rate stimulus.

        Returns
        -------
        spike_counts : (num_envs, output_size) tensor
                       Larger value → neuron fired more → preferred action.
        """
        spike_counts = torch.zeros(self.num_envs, self._output_size, device=self.device)
        for _ in range(self.n_steps):
            # Poisson / rate encoding: spike with probability = input value
            encoded = (torch.rand_like(obs) < obs).float()
            x = encoded
            for layer in self.layers:
                x = layer.forward(x)
            spike_counts += x   # x after output layer contains spikes
        return spike_counts

    def reset_state(self):
        """Reset all neuron membrane potentials (call between episodes)."""
        for layer in self.layers:
            layer.reset_state()

    def get_weights(self) -> torch.Tensor:
        """Return (num_envs, total_params) tensor."""
        return torch.cat([layer.get_weights() for layer in self.layers], dim=1)

    def set_weights(self, weights: torch.Tensor):
        """Set from (num_envs, total_params) tensor."""
        start = 0
        for layer in self.layers:
            w_size = layer._total_params
            layer.set_weights(weights[:, start:start + w_size])
            start += w_size