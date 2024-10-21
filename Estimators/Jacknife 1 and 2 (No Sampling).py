import pandas as pd
import os

def jackknife_order_1(S_obs, Q1):
    return S_obs + Q1

def jackknife_order_2(S_obs, Q1, Q2):
    return S_obs + 2 * Q1 - Q2

def calculate_jackknife_from_dataframe(df, count_col):
    S_obs = len(df)
    Q1 = len(df[df[count_col] == 1])
    Q2 = len(df[df[count_col] == 2])
    return S_obs, Q1, Q2

data_dir = "/kaggle/input/dear-lord-kill-me-now/Sepsis"

files_of_interest = {
    'activity_species.csv': 'count',
    'directly_follows_species.csv': 'count',
    'exponential_duration_species_zte2.csv': 'Count',
    'trace_variant_species.csv': 'count',
    'uniform_duration_species_zt1.csv': 'Count',
    'uniform_duration_species_zt5.csv': 'Count',
    'uniform_duration_species_zt30.csv': 'Count'
}

results = []

for file, count_col in files_of_interest.items():
    file_path = os.path.join(data_dir, file)
    df = pd.read_csv(file_path)
    S_obs, Q1, Q2 = calculate_jackknife_from_dataframe(df, count_col)

    jackknife1 = jackknife_order_1(S_obs, Q1)
    jackknife2 = jackknife_order_2(S_obs, Q1, Q2)

    jackknife1_rounded = round(jackknife1)
    jackknife2_rounded = round(jackknife2)

    results.append([file, S_obs, jackknife1_rounded, jackknife2_rounded])

results_df = pd.DataFrame(results, columns=['File', 'S_obs', 'Jackknife_Order_1', 'Jackknife_Order_2'])

output_file = "/kaggle/working/jackknife_results.csv"
results_df.to_csv(output_file, index=False)

print(results_df)
print(f"Results saved to: {output_file}")
