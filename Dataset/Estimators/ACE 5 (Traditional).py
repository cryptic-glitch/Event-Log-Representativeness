import pandas as pd
import os

def ace_estimator(df, count_col):
    rare_species = df[df[count_col] <= 5]
    abundant_species = df[df[count_col] > 5]

    S_obs = len(df)
    S_rare = len(rare_species)
    n_rare = rare_species[count_col].sum()
    F1 = len(df[df[count_col] == 1])

    if n_rare > 0:
        C_ACE = 1 - (F1 / n_rare)
    else:
        C_ACE = 0

    sum_i_i_minus_1_Fi = sum(i * (i - 1) * len(df[df[count_col] == i]) for i in range(1, 11))
    if n_rare > 1:
        gamma_sq_ACE = (S_rare / C_ACE) * (sum_i_i_minus_1_Fi / (n_rare * (n_rare - 1)))
    else:
        gamma_sq_ACE = 0

    if C_ACE > 0:
        S_ACE = S_rare + (F1 / C_ACE) * gamma_sq_ACE
    else:
        S_ACE = S_obs

    return round(S_ACE)

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

    S_ACE = ace_estimator(df, count_col)
    S_obs = len(df)

    results.append([file, S_obs, S_ACE])

results_df = pd.DataFrame(results, columns=['File', 'S_obs', 'ACE_Estimate'])

output_file = "/kaggle/working/ace_Rare5_Sepsis.csv"
results_df.to_csv(output_file, index=False)

print(results_df)
print(f"Results saved to: {output_file}")
