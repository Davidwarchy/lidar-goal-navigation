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


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """
    Individual agent in the population
    
    Each agent has:
    - Neural network (brain)
    - Energy level (starts with predefined amount)
    - Fitness score
    - Success status
    - Exploration tracking (coverage, distance traveled)
    """
    
    def __init__(self, network=None, energy: int = 100):
        if network is None:
            # Use 100 inputs to match actual lidar rays
            self.network = SpikingNeuralNetwork(input_size=100,
                                                hidden_size=64,
                                                output_size=4)
        else:
            self.network = network
            
        self.energy = energy
        self.start_energy = energy
        self.steps_taken = 0
        self.survived = False  # True iff energy ran to zero without dying

    def reset(self, energy: int = 100):
        """Reset agent for a new episode."""
        self.energy = energy
        self.start_energy = energy
        self.steps_taken = 0
        self.survived = False
        self.network.reset_state()

    def take_action(self, lidar_readings) -> int:
        """
        Query the SNN for an action.
        Softmax over spike counts → stochastic action selection.
        """
        inputs = np.array(lidar_readings, dtype=np.float32)

        n = self.network.input_size
        if len(inputs) < n:
            inputs = np.pad(inputs, (0, n - len(inputs)))
        elif len(inputs) > n:
            inputs = inputs[:n]

        spike_counts = self.network.forward(inputs)

        counts_shifted = spike_counts - spike_counts.max()
        probs = np.exp(counts_shifted)
        probs /= probs.sum() + 1e-9
        return int(np.random.choice(len(probs), p=probs))

    def update(self):
        """Deduct one energy unit and count the step."""
        self.energy -= 1
        self.steps_taken += 1

    def copy(self) -> "Agent":
        new_agent = Agent(network=self.network.copy(), energy=self.start_energy)
        return new_agent


# ---------------------------------------------------------------------------
# Genetic Algorithm  —  survival-only selection
# ---------------------------------------------------------------------------

class GeneticAlgorithm:
    """
    Survival-only GA: no ranking, no elitism, no fitness score.

    Next generation is bred from randomly chosen survivor pairs.
    If there are no survivors the population is re-seeded from scratch.
    """

    def __init__(self, population_size: int = 50,
                 mutation_rate: float = 0.1,
                 energy_per_agent: int = 100):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.energy_per_agent = energy_per_agent

    def crossover(self, parent1: Agent, parent2: Agent) -> Agent:
        """Uniform crossover of weight genomes."""
        w1 = parent1.network.get_weights()
        w2 = parent2.network.get_weights()
        min_len = min(len(w1), len(w2))
        mask = np.random.rand(min_len) > 0.5
        child_weights = np.where(mask, w1[:min_len], w2[:min_len])

        net = SpikingNeuralNetwork(
            input_size=parent1.network.input_size,
            hidden_size=parent1.network.hidden_size,
            output_size=parent1.network.output_size,
            n_steps=parent1.network.n_steps,
            leak=parent1.network.leak,
            threshold=parent1.network.threshold,
        )
        net.set_weights(child_weights)
        return Agent(network=net, energy=self.energy_per_agent)

    def mutate(self, agent: Agent) -> Agent:
        agent.network.mutate(rate=self.mutation_rate, magnitude=0.5)
        return agent

    def evolve(self, survivors: list) -> list:
        """
        Breed next generation from survivors.
        Parents are chosen uniformly at random — no ranking.
        If no survivors, return a freshly seeded random population.
        """
        if not survivors:
            print("[GA] No survivors — re-seeding population from scratch.")
            return self._fresh_population()

        print(f"[GA] {len(survivors)} survivor(s) → breeding next generation.")

        new_agents = []
        while len(new_agents) < self.population_size:
            if len(survivors) == 1:
                # Can only mutate the sole survivor
                child = survivors[0].copy()
            else:
                p1, p2 = np.random.choice(len(survivors), size=2, replace=False)
                child = self.crossover(survivors[p1], survivors[p2])
            child = self.mutate(child)
            new_agents.append(child)

        return new_agents

    def _fresh_population(self) -> list:
        template = self.population[0].network if hasattr(self, 'population') else None
        agents = []
        for _ in range(self.population_size):
            if template is not None:
                net = SpikingNeuralNetwork(
                    input_size=template.input_size,
                    hidden_size=template.hidden_size,
                    output_size=template.output_size,
                    n_steps=template.n_steps,
                    leak=template.leak,
                    threshold=template.threshold,
                )
            else:
                net = SpikingNeuralNetwork()
            agents.append(Agent(network=net, energy=self.energy_per_agent))
        return agents


# ---------------------------------------------------------------------------
# Main Strategy
# ---------------------------------------------------------------------------

class SpikeNNGeneticStrategy:
    """
    Neuroevolution strategy with survival as the only selection pressure.

    An agent is a survivor if it uses its full energy budget without dying
    (i.e. the environment never rejected its position due to a collision).
    No oracle metrics (coverage, distance, health, fitness) influence
    which agents reproduce.
    """

    def __init__(self,
                 population_size: int = 50,
                 generations: int = 20,
                 energy_per_agent: int = 100,
                 mutation_rate: float = 0.2,
                 num_rays: int = 100,
                 weights_dir: str = "spike_weights",
                 hidden_size: int = 64,
                 n_steps: int = 5,
                 leak: float = 0.9,
                 threshold: float = 1.0):

        self.name = "spike_nn_genetic"
        self.parameters = dict(
            population_size=population_size,
            generations=generations,
            energy_per_agent=energy_per_agent,
            mutation_rate=mutation_rate,
            num_rays=num_rays,
            weights_dir=weights_dir,
            hidden_size=hidden_size,
            n_steps=n_steps,
            leak=leak,
            threshold=threshold,
        )

        self.population_size = population_size
        self.generations = generations
        self.energy_per_agent = energy_per_agent
        self.mutation_rate = mutation_rate
        self.num_rays = num_rays
        self.weights_dir = weights_dir
        self.hidden_size = hidden_size
        self.n_steps = n_steps
        self.leak = leak
        self.threshold = threshold

        self.ga = GeneticAlgorithm(population_size, mutation_rate, energy_per_agent)

        self.population = [
            Agent(network=SpikingNeuralNetwork(
                input_size=num_rays,
                hidden_size=hidden_size,
                output_size=4,
                n_steps=n_steps,
                leak=leak,
                threshold=threshold,
            ), energy=energy_per_agent)
            for _ in range(population_size)
        ]

        self.metrics = {
            "generation": 0,
            "n_survivors": 0,
            "percent_survived": 0.0,
            "weights_directory": weights_dir,
        }

        self.all_generation_metrics = []
        os.makedirs(weights_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _preprocess_lidar(self, lidar_readings) -> np.ndarray:
        """Normalise lidar to [0, 1] and match network input size."""
        arr = np.array(lidar_readings, dtype=np.float32)
        if arr.max() > 0:
            arr = arr / arr.max()
        n = self.population[0].network.input_size if self.population else self.num_rays
        if len(arr) < n:
            arr = np.pad(arr, (0, n - len(arr)))
        elif len(arr) > n:
            arr = arr[:n]
        return np.clip(arr + 0.01, 0.0, 1.0)

    def _run_agent(self, agent: Agent, env) -> None:
        """
        Run one agent for a full episode.

        Survival = the agent depletes its energy without the environment
        blocking it (wall collision freezes position; we detect a stuck
        agent by checking whether it made any progress at all in the last
        N steps — but we do NOT measure coverage or distance as a metric).
        The simplest definition we use here: the episode ends naturally
        (done flag from env) OR energy hits zero. The agent survives if
        energy reaches zero (it kept moving until budget ran out).
        """
        agent.reset(self.energy_per_agent)
        obs = env.reset()
        done = False

        prev_x, prev_y = env.robot_x, env.robot_y
        stuck_steps = 0
        STUCK_LIMIT = 20  # if robot doesn't move for this many steps, it's dead

        while not done and agent.energy > 0:
            lidar = obs[0] if isinstance(obs, tuple) else obs
            inputs = self._preprocess_lidar(lidar)
            action = agent.take_action(inputs)
            obs, _reward, done, _info = env.step(action)
            agent.update()

            # Detect stuck (collision loop): position unchanged
            if abs(env.robot_x - prev_x) < 0.01 and abs(env.robot_y - prev_y) < 0.01:
                stuck_steps += 1
            else:
                stuck_steps = 0
            prev_x, prev_y = env.robot_x, env.robot_y

            if stuck_steps >= STUCK_LIMIT:
                # Robot is permanently blocked — it does not survive
                return

        # Survived if energy ran to zero (not killed by collision loop)
        if agent.energy <= 0:
            agent.survived = True

    def _evaluate_population(self, env) -> dict:
        """Run all agents; collect survivors. No fitness computed."""
        survivors = []

        for agent in self.population:
            self._run_agent(agent, env)
            if agent.survived:
                survivors.append(agent)

        n_ok = len(survivors)
        self.metrics["n_survivors"] = n_ok
        self.metrics["percent_survived"] = (n_ok / self.population_size) * 100.0

        return survivors

    def _save_weights(self, generation: int, survivors: list) -> str:
        """
        Save a randomly chosen survivor's weights.
        If no survivors, skip saving (nothing to save).
        """
        if not survivors:
            print(f"[Gen {generation}] No survivors — skipping weight save.")
            return ""

        chosen = survivors[np.random.randint(len(survivors))]
        filename = os.path.join(self.weights_dir, f"gen_{generation:03d}.json")
        data = {
            "generation": generation,
            "steps": chosen.steps_taken,
            "weights": chosen.network.get_weights().tolist(),
            "snn_params": {
                "input_size": chosen.network.input_size,
                "hidden_size": chosen.network.hidden_size,
                "output_size": chosen.network.output_size,
                "n_steps": chosen.network.n_steps,
                "leak": chosen.network.leak,
                "threshold": chosen.network.threshold,
            }
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
        return filename

    def _print_metrics(self, metrics: dict):
        print("\n" + "=" * 60)
        print(f"Generation: {metrics['generation']}")
        print("=" * 60)
        print(f"  Survivors:        {metrics['n_survivors']} / "
              f"{self.population_size}  "
              f"({metrics['percent_survived']:.1f}%)")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, env):
        """
        Run the neuroevolution training loop.

        Returns
        -------
        total_steps : int
        """
        total_steps = 0

        print("\n" + "=" * 60)
        print("Starting SNN Neuroevolution  —  survival selection only")
        print("=" * 60)
        for k, v in self.parameters.items():
            print(f"  {k:<28} {v}")
        print("=" * 60)

        for gen in range(self.generations):
            self.metrics["generation"] = gen + 1

            survivors = self._evaluate_population(env)
            self.metrics["weights_directory"] = self.weights_dir
            self.all_generation_metrics.append(self.metrics.copy())
            self._print_metrics(self.metrics)

            total_steps += sum(a.steps_taken for a in self.population)

            weight_file = self._save_weights(gen, survivors)
            if weight_file:
                print(f"Saved weights to: {weight_file}")

            if gen < self.generations - 1:
                self.population = self.ga.evolve(survivors)
                print(f"\nEvolving to generation {gen + 2}…")

        print("\n" + "=" * 60)
        print("Neuroevolution Training Complete!")
        print("=" * 60)

        summary_file = os.path.join(self.weights_dir, "training_summary.json")
        with open(summary_file, 'w') as f:
            json.dump({
                "total_generations": self.generations,
                "final_metrics": self.metrics,
                "total_steps": total_steps,
            }, f, indent=2)

        csv_file = os.path.join(self.weights_dir, "training_metrics.csv")
        fieldnames = [
            'Generation',
            'N Survivors',
            '% Survived',
            'Weights Directory',
        ]
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for gm in self.all_generation_metrics:
                writer.writerow({
                    'Generation': gm.get('generation', 0),
                    'N Survivors': gm.get('n_survivors', 0),
                    '% Survived': f"{gm.get('percent_survived', 0):.1f}%",
                    'Weights Directory': gm.get('weights_directory', ''),
                })

        print(f"Training summary saved to: {summary_file}")
        print(f"CSV metrics saved to:      {csv_file}")

        return total_steps


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_spike_nn_strategy(**kwargs) -> SpikeNNGeneticStrategy:
    """Factory function — drop-in replacement for the original."""
    return SpikeNNGeneticStrategy(**kwargs)