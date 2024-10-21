import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

def jackknife_order_1(S_obs, Q1):
    return S_obs + Q1

def jackknife_order_2(S_obs, Q1, Q2):
    return S_obs + 2 * Q1 - Q2

def calculate_s_obs_q1_q2(df, count_col):
    counts = df[count_col].values
    S_obs = len(df)
    Q1 = np.sum(counts == 1)
    Q2 = np.sum(counts == 2)
    return S_obs, Q1, Q2

def jackknife_resampling(df, count_col):
    counts = df[count_col].values
    S_obs, Q1, Q2 = calculate_s_obs_q1_q2(df, count_col)

    S_obs_resampled = S_obs - 1
    Q1_resampled = Q1 - (counts == 1)
    Q2_resampled = Q2 - (counts == 2)

    jackknife1_estimates = S_obs_resampled + Q1_resampled
    jackknife2_estimates = S_obs_resampled + 2 * Q1_resampled - Q2_resampled

    jackknife1_estimate = np.mean(jackknife1_estimates)
    jackknife2_estimate = np.mean(jackknife2_estimates)

    return round(jackknife1_estimate), round(jackknife2_estimate)

def process_file(file_info):
    file, count_col, data_dir = file_info
    file_path = os.path.join(data_dir, file)
    df = pd.read_csv(file_path)

    jackknife1_resample, jackknife2_resample = jackknife_resampling(df, count_col)
    S_obs, _, _ = calculate_s_obs_q1_q2(df, count_col)

    return [file, S_obs, jackknife1_resample, jackknife2_resample]

data_dir = "/kaggle/input/dear-lord-kill-me-now/BPI-2018"

files_of_interest = {
    'activity_species.csv': 'count',
    'directly_follows_species.csv': 'count',
    'exponential_duration_species_zte2.csv': 'Count',
    'trace_variant_species.csv': 'count',
    'uniform_duration_species_zt1.csv': 'Count',
    'uniform_duration_species_zt5.csv': 'Count',
    'uniform_duration_species_zt30.csv': 'Count'
}

file_info_list = [(file, count_col, data_dir) for file, count_col in files_of_interest.items()]

results = []

with ProcessPoolExecutor() as executor:
    future_to_file = {executor.submit(process_file, file_info): file_info[0] for file_info in file_info_list}

    for future in as_completed(future_to_file):
        file_name = future_to_file[future]
        try:
            result = future.result()
            results.append(result)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

results_df = pd.DataFrame(results,
                          columns=['File', 'S_obs', 'Jackknife_Order_1_Resampling', 'Jackknife_Order_2_Resampling'])

output_file = "/kaggle/working/jacknife_resampling_BPI-2018.csv"
results_df.to_csv(output_file, index=False)

print(results_df)
print(f"Results saved to: {output_file}")
