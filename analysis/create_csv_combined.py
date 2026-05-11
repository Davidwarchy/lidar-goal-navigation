import pandas as pd
import glob
import os

def combine_csv_files(directory='.', output_filename='combined_results.csv'):
    # Get all csv files in the specified directory
    all_files = glob.glob(os.path.join(directory, "*.csv"))
    
    # Exclude the output file if it already exists
    all_files = [f for f in all_files if os.path.basename(f) != output_filename]
    
    if not all_files:
        print("No CSV files found.")
        return None

    df_list = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            # Add a column to track the source file (optional but useful)
            df['source_file'] = os.path.basename(filename)
            df_list.append(df)
            print(f"Loaded {filename} with shape {df.shape}")
        except Exception as e:
            print(f"Could not read {filename}: {e}")

    # Concatenate all dataframes. 
    # Pandas handles different columns by performing an outer join.
    combined_df = pd.concat(df_list, axis=0, ignore_index=True, sort=False)
    
    # Save the combined dataframe to a CSV
    combined_df.to_csv(output_filename, index=False)
    print(f"\nCombined CSV saved to {output_filename}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Total columns: {len(combined_df.columns)}")
    
    return combined_df

# Run the function on the current directory
dir_in = "output/experiments"
path_out = os.path.join("output", "combined_results.csv")
combined_df = combine_csv_files(dir_in, path_out)
if combined_df is not None:
    print("\nColumns in the combined file:")
    print(combined_df.columns.tolist())
    print("\nFirst few rows:")
    print(combined_df.head())