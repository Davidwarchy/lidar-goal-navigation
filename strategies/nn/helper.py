import torch
from .network import NeuralNetwork, VectorSpikingNetwork, VectorFeedforwardNetwork


# ---------------------------------------------------------------------------
# Network Factory
# ---------------------------------------------------------------------------

def create_vector_neural_network(network_type: str, num_envs: int, input_size: int, 
                                  output_size: int, device='cuda', **kwargs) -> NeuralNetwork:
    """
    Factory function to create different types of vectorized neural networks.
    
    Parameters:
    -----------
    network_type : str
        Type of network to create. Options: "spiking", "feedforward"
    num_envs : int
        Number of parallel agents
    input_size : int
        Input dimension
    output_size : int
        Output dimension
    device : str
        Device to run on ('cuda' or 'cpu')
    **kwargs : additional parameters
        For spiking: hidden_sizes (list of int, default [64]), n_steps (int, default 5)
        For feedforward: hidden_sizes (list of int, default [64])
    
    Returns:
    --------
    NeuralNetwork instance
    """
    if network_type == "spiking":
        hidden_sizes = kwargs.get("hidden_sizes", [64])
        n_steps = kwargs.get("n_steps", 5)
        return VectorSpikingNetwork(num_envs, input_size, hidden_sizes, output_size, n_steps, device)
    
    elif network_type == "feedforward":
        hidden_sizes = kwargs.get("hidden_sizes", [64])
        return VectorFeedforwardNetwork(num_envs, input_size, hidden_sizes, output_size, device)
    
    else:
        raise ValueError(f"Unknown network type: {network_type}. Choose from 'spiking' or 'feedforward'")
    

# ---------------------------------------------------------------------------
# Recombination Function (Crossover) for Genetic Algorithm
# ---------------------------------------------------------------------------
def _average_recombination(genomes: torch.Tensor) -> torch.Tensor:
    """
    Pairwise average recombination.
    Input: (population_size, total_params)
    Output: same shape, where each pair of consecutive individuals is replaced
            by their element-wise average (both children become that average).
    If population_size is odd, the last individual is left unchanged.
    """
    pop_size = genomes.shape[0]
    if pop_size < 2:
        return genomes
    # Make even number of individuals for pairing
    even_pop = pop_size if pop_size % 2 == 0 else pop_size - 1
    if even_pop > 0:
        # Pair consecutive indices: (0,1), (2,3), ...
        paired = genomes[:even_pop].view(-1, 2, genomes.shape[-1])
        avg = paired.mean(dim=1, keepdim=True)          # (pairs, 1, params)
        paired[:] = avg.expand_as(paired)               # both children = average
    return genomes
