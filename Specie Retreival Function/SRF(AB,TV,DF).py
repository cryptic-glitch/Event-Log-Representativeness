CASE_ID_KEY = 'case:concept:name'
ACTIVITY_ID_KEY = 'concept:name'
EVENT_INSTANCE_KEY = 'concept:instance'
TIMESTAMP = 'time:timestamp'
LIFECYCLE = 'lifecycle:transition'
FROM_KEY = 'time:from'
UNTIL_KEY = 'time:until'
DURATION_KEY = 'time:duration'
EVENT_ID_KEY = 'concept:instance'

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def has_lifecycle_info(df):
    return (LIFECYCLE in df.columns) and df[LIFECYCLE].nunique() > 1

def lclog_to_durationlog(df):
    df = df[(df[LIFECYCLE].str.lower() == "start") | (df[LIFECYCLE].str.lower() == "complete")]
    df = df.sort_values(by=[CASE_ID_KEY, ACTIVITY_ID_KEY, TIMESTAMP]).reset_index(drop=True)

    df['prev_timestamp'] = df[TIMESTAMP].shift(1)
    df['prev_activity'] = df[ACTIVITY_ID_KEY].shift(1)
    df['prev_case'] = df[CASE_ID_KEY].shift(1)

    df[DURATION_KEY] = df[TIMESTAMP] - df['prev_timestamp']

    df = df[(df[LIFECYCLE].str.lower() == 'complete') &
            (df['prev_case'] == df[CASE_ID_KEY]) &
            (df['prev_activity'] == df[ACTIVITY_ID_KEY])]

    return df.drop(columns=['prev_timestamp', 'prev_activity', 'prev_case'])

def log_to_durationlog(df):
    df = df.sort_values(by=[CASE_ID_KEY, TIMESTAMP]).reset_index(drop=True)

    df['prev_timestamp'] = df[TIMESTAMP].shift(1)
    df['prev_case'] = df[CASE_ID_KEY].shift(1)

    df[DURATION_KEY] = df[TIMESTAMP] - df['prev_timestamp']
    df = df[(df['prev_case'] == df[CASE_ID_KEY])]

    return df.drop(columns=['prev_timestamp', 'prev_case'])

def convert_to_duration_log(df):
    if has_lifecycle_info(df):
        return lclog_to_durationlog(df)
    else:
        return log_to_durationlog(df)

def compute_species_in_parallel(df, func, *args):
    with ThreadPoolExecutor() as executor:
        future = executor.submit(func, df, *args)
        result = future.result()
    return result

def activity_based_species(df):
    unique_activities = df[ACTIVITY_ID_KEY].value_counts().reset_index(name='count')
    unique_activities.columns = ['species', 'count']
    return unique_activities, len(unique_activities)

def directly_follows_species(df, respect_lifecycle=True, include_startend=True):
    if respect_lifecycle and has_lifecycle_info(df):
        df = df[df[LIFECYCLE].str.lower() == 'complete']

    df_sorted = df.sort_values(by=[CASE_ID_KEY, TIMESTAMP]).reset_index(drop=True)
    df_sorted['next_activity'] = df_sorted.groupby(CASE_ID_KEY)[ACTIVITY_ID_KEY].shift(-1)

    directly_follows = df_sorted[[ACTIVITY_ID_KEY, 'next_activity']].dropna()

    if include_startend:
        start_rows = pd.DataFrame({ACTIVITY_ID_KEY: '$START$', 'next_activity': df_sorted.groupby(CASE_ID_KEY)[ACTIVITY_ID_KEY].first()})
        end_rows = pd.DataFrame({ACTIVITY_ID_KEY: df_sorted.groupby(CASE_ID_KEY)[ACTIVITY_ID_KEY].last(), 'next_activity': '$END$'})
        directly_follows = pd.concat([start_rows, directly_follows, end_rows])

    counts = directly_follows.groupby([ACTIVITY_ID_KEY, 'next_activity']).size().reset_index(name='count')
    counts.columns = ['species', 'next_species', 'count']
    return counts, counts.shape[0]

def trace_variant_based_species(df):
    trace_variants = df.groupby(CASE_ID_KEY)[ACTIVITY_ID_KEY].apply(lambda x: ','.join(x)).value_counts().reset_index(name='count')
    trace_variants.columns = ['species', 'count']
    return trace_variants, len(trace_variants)

def calculate_Q1_Q2(counts):
    frequency_counts = counts['count'].value_counts()
    Q1 = frequency_counts.get(1, 0)
    Q2 = frequency_counts.get(2, 0)
    return Q1, Q2

df = pd.read_csv('/kaggle/input/bpi-2019-sample-grouped/BPI-2019_grouped_sample_50 (1).csv')

df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], format='ISO8601', errors='coerce')

df_sorted = df.sort_values(by=[CASE_ID_KEY, TIMESTAMP])

df_sorted[DURATION_KEY] = df_sorted.groupby(CASE_ID_KEY)[TIMESTAMP].diff().dt.total_seconds().fillna(0)

activity_species, activity_count = compute_species_in_parallel(df_sorted, activity_based_species)
Q1_act, Q2_act = compute_species_in_parallel(activity_species, calculate_Q1_Q2)

directly_follows_result, directly_follows_count = compute_species_in_parallel(df_sorted, directly_follows_species)
Q1_df, Q2_df = compute_species_in_parallel(directly_follows_result, calculate_Q1_Q2)

trace_variant_species, trace_variant_count = compute_species_in_parallel(df_sorted, trace_variant_based_species)
Q1_tv, Q2_tv = compute_species_in_parallel(trace_variant_species, calculate_Q1_Q2)

activity_species.to_csv('/kaggle/working/50R_activity_species_BPI-2019.csv', index=False)
directly_follows_result.to_csv('/kaggle/working/50R_directly_follows_species_BPI-2019.csv', index=False)
trace_variant_species.to_csv('/kaggle/working/50R_trace_variant_species_BPI-2019.csv', index=False)

print(f"Activity-based Species (ζact): {activity_count} species found")
print(f"Q1 (Singletons) for ζact: {Q1_act}")
print(f"Q2 (Doubletons) for ζact: {Q2_act}")

print(f"Directly-Follows Relation-based Species (ζdf): {directly_follows_count} species found")
print(f"Q1 (Singletons) for ζdf: {Q1_df}")
print(f"Q2 (Doubletons) for ζdf: {Q2_df}")

print(f"Trace Variant-based Species (ζtv): {trace_variant_count} species found")
print(f"Q1 (Singletons) for ζtv: {Q1_tv}")
print(f"Q2 (Doubletons) for ζtv: {Q2_tv}")
