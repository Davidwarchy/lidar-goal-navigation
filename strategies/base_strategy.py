from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    def __init__(self, name, parameters=None, parallel_trials=False, 
             fitness_proxy="none", fitness_pool="survivors_only"):
            self.name = name
            self.parameters = parameters or {}
            self.parallel_trials = parallel_trials
            self.fitness_proxy = fitness_proxy
            self.fitness_pool = fitness_pool
            
    @abstractmethod
    def run(self, env_params, base_output_dir):
        """Run the strategy.
        
        Args:
            env_params: dict of parameters to create an environment (without output_dir).
            base_output_dir: root directory for this strategy run.
        """
        pass
    
    def get_metadata(self):
        """Get strategy metadata for saving"""
        return {
            "strategy_name": self.name, 
            "strategy_parameters": self.parameters,
            "fitness_proxy": self.fitness_proxy,
            "fitness_pool": self.fitness_pool
        }