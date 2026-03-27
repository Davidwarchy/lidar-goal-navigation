"""
Spiking Neural Network (SNN) with Genetic Algorithm (Neuroevolution)

This strategy implements a population-based neuroevolution system where
multiple neural network agents compete in each generation. The best performing
agents are selected for reproduction through natural selection.

Key Features:
- Population of neural networks (not just one)
- Genetic algorithm: selection, crossover, mutation
- Energy system: each step costs 1 energy point
- Exploration-based fitness: reward based on coverage and distance traveled

Neural Network:
- Leaky Integrate-and-Fire (LIF) Spiking Neural Network
- Neurons accumulate membrane potential over time
- Fire binary spikes when potential crosses threshold
- Membrane potential leaks toward resting state each timestep
- Spike traces used for weight representation in GA genome

Fitness Metrics Tracked:
- Generation: Current iteration number
- % of Generation Successful: Percentage of agents that achieved good coverage
- Average Success Path Length: Average steps taken by successful agents
- Average Success Energy Remaining: Average energy left for successful agents
- Average Success Health: Same as energy remaining
- Average Distance to Reward: Average distance traveled
- Weights Directory: Location where weights are saved
"""

import numpy as np
import os
import json
import csv
from datetime import datetime


# ---------------------------------------------------------------------------
# Leaky Integrate-and-Fire Spiking Neural Network
# ---------------------------------------------------------------------------

class LIFNeuronLayer:
    """
    A layer of Leaky Integrate-and-Fire (LIF) neurons.

    Each neuron maintains a membrane potential V.  On every call to
    `forward()` the potential is updated according to:

        V[t] = leak * V[t-1] + W @ x[t] + b

    A neuron fires (output = 1) when V >= threshold, after which its
    potential is reset to `reset_potential`.

    Parameters
    ----------
    n_in  : number of pre-synaptic inputs
    n_out : number of neurons in this layer
    leak  : membrane leak factor  (0 < leak < 1); default 0.9
    threshold : firing threshold; default 1.0
    reset_potential : post-spike reset value; default 0.0
    """

    def __init__(self, n_in: int, n_out: int,
                 leak: float = 0.9,
                 threshold: float = 1.0,
                 reset_potential: float = 0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.leak = leak
        self.threshold = threshold
        self.reset_potential = reset_potential

        # Synaptic weight matrix  (n_in x n_out) and bias (n_out,)
        self.W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        self.b = np.zeros(n_out)

        # Runtime state (reset between episodes)
        self.V = np.zeros(n_out)   # membrane potential

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Process one time-step of input spikes / rates.

        Parameters
        ----------
        x : (n_in,) array – input activations (spikes or rates)

        Returns
        -------
        spikes : (n_out,) binary array  (1 = fired, 0 = silent)
        """
        # Leak + integrate
        self.V = self.leak * self.V + x @ self.W + self.b

        # Fire
        spikes = (self.V >= self.threshold).astype(np.float32)

        # Reset fired neurons
        self.V = np.where(spikes == 1.0, self.reset_potential, self.V)

        return spikes

    def reset_state(self):
        """Reset membrane potentials to zero (call between episodes)."""
        self.V[:] = 0.0

    # ------------------------------------------------------------------
    # Genome helpers (flat weight vector for the GA)
    # ------------------------------------------------------------------

    def get_weights(self) -> np.ndarray:
        return np.concatenate([self.W.flatten(), self.b])

    def set_weights(self, weights: np.ndarray):
        size_W = self.n_in * self.n_out
        self.W = weights[:size_W].reshape(self.n_in, self.n_out)
        self.b = weights[size_W: size_W + self.n_out]

    @property
    def n_params(self) -> int:
        return self.n_in * self.n_out + self.n_out


class SpikingNeuralNetwork:
    """
    Two-layer Leaky Integrate-and-Fire Spiking Neural Network.

    Architecture
    ------------
    Input  (n_in  neurons, rate-coded)
      ↓  LIF hidden layer
    Hidden (hidden_size neurons, binary spikes)
      ↓  LIF output layer
    Output (n_out neurons, binary spikes → argmax action)

    The network is run for `n_steps` internal time-steps per call so that
    spiking dynamics can accumulate meaningful activity from a single
    observation vector.

    Parameters
    ----------
    input_size  : dimensionality of the observation vector
    hidden_size : number of hidden LIF neurons
    output_size : number of action neurons
    n_steps     : internal simulation steps per observation (default 5)
    leak        : membrane leak for all layers (default 0.9)
    threshold   : firing threshold for all layers (default 1.0)
    """

    def __init__(self,
                 input_size: int = 100,
                 hidden_size: int = 64,
                 output_size: int = 4,
                 n_steps: int = 5,
                 leak: float = 0.9,
                 threshold: float = 1.0):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_steps = n_steps
        self.leak = leak
        self.threshold = threshold

        self.hidden_layer = LIFNeuronLayer(input_size, hidden_size,
                                           leak=leak, threshold=threshold)
        self.output_layer = LIFNeuronLayer(hidden_size, output_size,
                                           leak=leak, threshold=threshold)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run the SNN for `n_steps` time-steps and return accumulated output
        spike counts (one value per action neuron).

        Parameters
        ----------
        inputs : (input_size,) float array, values in [0, 1]
                 Treated as a constant Poisson rate stimulus.

        Returns
        -------
        spike_counts : (output_size,) float array
                       Larger value → neuron fired more → preferred action.
        """
        # Encode continuous input as Bernoulli spikes each time-step
        spike_counts = np.zeros(self.output_size)

        for _ in range(self.n_steps):
            # Poisson / rate encoding: spike with probability = input value
            encoded = (np.random.rand(self.input_size) < inputs).astype(np.float32)

            h_spikes = self.hidden_layer.forward(encoded)
            o_spikes = self.output_layer.forward(h_spikes)
            spike_counts += o_spikes

        return spike_counts

    def reset_state(self):
        """Reset all neuron membrane potentials (call between episodes)."""
        self.hidden_layer.reset_state()
        self.output_layer.reset_state()

    # ------------------------------------------------------------------
    # Genome helpers
    # ------------------------------------------------------------------

    def get_weights(self) -> np.ndarray:
        return np.concatenate([
            self.hidden_layer.get_weights(),
            self.output_layer.get_weights()
        ])

    def set_weights(self, weights: np.ndarray):
        n_h = self.hidden_layer.n_params
        self.hidden_layer.set_weights(weights[:n_h])
        self.output_layer.set_weights(weights[n_h: n_h + self.output_layer.n_params])

    def copy(self) -> "SpikingNeuralNetwork":
        new_net = SpikingNeuralNetwork(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            n_steps=self.n_steps,
            leak=self.leak,
            threshold=self.threshold,
        )
        new_net.set_weights(self.get_weights().copy())
        return new_net

    def mutate(self, rate: float = 0.1, magnitude: float = 0.5):
        """In-place Gaussian weight mutation."""
        weights = self.get_weights()
        mask = np.random.random(len(weights)) < rate
        weights[mask] += np.random.randn(mask.sum()) * magnitude
        np.clip(weights, -3.0, 3.0, out=weights)
        self.set_weights(weights)

