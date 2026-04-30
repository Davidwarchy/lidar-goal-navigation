# Script for testing whether our systems break after making changes 
python main.py --strategy random  --max_steps 1000 --population 10 --generations 3 

# OMEN 
py -3.11 main.py --strategy spiking --mutation_rate 0.1 --mutation_mag 0.25 --trials 1 --generations 3 --population 1000  