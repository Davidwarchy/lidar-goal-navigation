Short-Run Validation: Create a test that runs each strategy (random, levy, ga, uniform) for exactly 10 steps. If the simulation completes and saves a metadata.json, the integration is successful.

Output Schema Validation: Automate a check to ensure that after a run, the expected files (metadata.json, generation_results.csv, and the gen_x_weights folder) actually exist and contain valid JSON/CSV data.

