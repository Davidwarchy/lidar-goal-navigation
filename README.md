# Robot Exploration 

A tiny two-wheel robot with a 100-ray lidar learns objecthood from pure sensorimotor experience. With no labels or tasks, it samples simple actions and tracks how they distort the lidar stream, revealing stable “things” in the world—structures it can use to predict, navigate, and act coherently.

A lightweight Gym-style setup for 2D robot navigation and exploration using lidar.

![alt text](media/self-grounded-agent.gif)

Figure interpretation. Black - obstacles. White - free space. Green - explored regions. Yellow - lidar rays. Maroon - detected obstacles. 

Overall, the project explores how autonomous agents can bootstrap internal representations from sensorimotor experience, aiming toward the goal of developing self-sustaining computational systems.

## Environments

Maps are stored as bitmap images in `environments/`.

* **White** = free space
* **Black** = obstacles

## Data Collection

The agent is a differential-drive robot (two wheels) with **100 lidar rays**.

Ensure that requirements are installed 

```bash
!pip install -r requirements.txt
``` 

You can run exploration with predefined strategies. Example: random walk.

```bash
python main.py --strategy random --max_steps 10000 --env 6.png
```

Results are saved automatically to an `output/` folder with timestamped directories. Other strategies include manual control, uniformly distributed random run lengths, levy walks, etc. 

## Performance Comparison Analysis

After running experiments, compare strategies (including baseline-relative deltas) with:

```bash
python -m analysis.performance.run_performance_analysis --root output --baselines random,levy,uniform --success-threshold 25 --plots
```

Outputs are written to `analysis/performance/results/`:
- `per_generation_comparison.csv`
- `per_trial_summary.csv`
- `baseline_deltas.csv`
- `report.json`
