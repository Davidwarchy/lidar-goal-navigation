from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    def __init__(self, name, parameters=None, parallel_trials=False):
        self.name = name
        self.parameters = parameters or {}
        self.parallel_trials = parallel_trials
    
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
            "strategy_parameters": self.parameters 
        }