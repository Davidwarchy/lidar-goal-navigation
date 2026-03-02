import random
import asyncio
import numpy as np
from .base_strategy import BaseStrategy

class UniformRunLengthStrategy(BaseStrategy):
    """
    Choose a direction (0..3) and run for a length sampled uniformly
    from [min_step, max_step] (integers). Then pick a new direction
    uniformly and sample another run length.
    """
    def __init__(self, min_step=1, max_step=200):
        params = {"min_step": int(min_step), "max_step": int(max_step)}
        super().__init__("uniform", params)

        self.min_step = int(min_step)
        self.max_step = int(max_step)
        if self.min_step < 1:
            raise ValueError("min_step must be >= 1")
        if self.max_step < self.min_step:
            raise ValueError("max_step must be >= min_step")

    def _sample_uniform_runlength(self):
        # inclusive integer uniform sample
        return int(random.randint(self.min_step, self.max_step))

    async def run(self, env):
        obs = env.reset()
        done = False

        current_run_length = 0
        current_direction = random.randint(0, 3)

        while not done:
            extra_info = {}

            # Start a new run if finished the last one
            if current_run_length <= 0:
                current_direction = random.randint(0, 3)
                current_run_length = self._sample_uniform_runlength()
                
                # Store run start info
                extra_info['run_start'] = 1
                extra_info['run_length'] = current_run_length
            else:
                extra_info['run_start'] = 0
                extra_info['run_length'] = 0

            # Step environment (action is direction 0..3)
            obs, reward, done, info = env.step(current_direction, extra_info=extra_info)

            # Render if requested
            if env.render_flag:
                env.render()

            # Update run counter
            current_run_length -= 1

            # Occasional status print
            if env.current_step % 100 == 0:
                print(
                    f"Step {env.current_step}, "
                    f"Direction: {info['action']}, "
                    f"Run steps left: {current_run_length}, "
                    f"Coverage: {info['coverage']:.2f}%"
                )

            # Optional slowdown (commented)
            # await asyncio.sleep(1.0 / env.fps)

        return env.current_step, env._get_coverage()