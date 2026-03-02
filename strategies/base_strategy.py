from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    def __init__(self, name, parameters=None):
        self.name = name
        self.parameters = parameters or {}
    
    @abstractmethod
    async def run(self, env):
        """Run the strategy with the given environment"""
        pass
    
    def get_metadata(self):
        """Get strategy metadata for saving"""
        return {
            "strategy_name": self.name, 
            "strategy_parameters": self.parameters 
        }