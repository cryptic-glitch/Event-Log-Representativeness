import pandas as pd
import numpy as np
from pandas import Series, DataFrame, Timedelta
import time

CASE_ID_KEY = 'case:concept:name'
ACTIVITY_ID_KEY = 'concept:name'
TIMESTAMP = 'time:timestamp'
LIFECYCLE = 'lifecycle:transition'
DURATION_KEY = 'time:duration'
FROM_KEY = 'time:from'
UNTIL_KEY = 'time:until'

def has_lifecycle_info(log: DataFrame) -> bool:
    return (LIFECYCLE in log.columns) and log[LIFECYCLE].unique().shape[0] > 1

def log_to_durationlog(log: DataFrame) -> DataFrame:
    if LIFECYCLE in log.columns:
        log = log[log[LIFECYCLE].str.lower() == "complete"]
    sort_columns = [CASE_ID_KEY, TIMESTAMP]
    df: DataFrame = log.sort_values(sort_columns).reset_index()

    aux_case = 'aux_case_' + str(time.time())
    aux_timestamp = 'aux_time_' + str(time.time())

    df[aux_case] = df[CASE_ID_KEY].shift(1)
    df[aux_timestamp] = df[TIMESTAMP].shift(1)

    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])
    df[aux_timestamp] = pd.to_datetime(df[aux_timestamp])

    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP], format='ISO8601')
    df[aux_timestamp] = pd.to_datetime(df[aux_timestamp], format='ISO8601')

    at_border = ~(df[aux_case] == df[CASE_ID_KEY])
    df.loc[~at_border, FROM_KEY] = df[~at_border][aux_timestamp]
    df.loc[at_border, FROM_KEY] = df[at_border][TIMESTAMP]
    df[UNTIL_KEY] = df[TIMESTAMP]

    df = df.drop([aux_case, aux_timestamp], axis=1)

    df[FROM_KEY] = pd.to_datetime(df[FROM_KEY])
    df[UNTIL_KEY] = pd.to_datetime(df[UNTIL_KEY])
    df[DURATION_KEY] = df[UNTIL_KEY] - df[FROM_KEY]

    return df

def lclog_to_durationlog(log: DataFrame) -> DataFrame:
    with_id = ('concept:instance' in log.columns)
    log = log[(log[LIFECYCLE].str.lower() == "start") | (log[LIFECYCLE].str.lower() == "complete")]
    sort_columns = [CASE_ID_KEY, ACTIVITY_ID_KEY]
    if with_id:
        sort_columns += ['concept:instance']
    sort_columns += [TIMESTAMP]
    df: DataFrame = log.sort_values(sort_columns).reset_index()

    aux_act = 'aux_act_' + str(time.time())
    aux_case = 'aux_case_' + str(time.time())
    aux_timestamp = 'aux_time_' + str(time.time())
    aux_lifecycle = 'aux_lc_' + str(time.time())

    df[aux_act] = df[ACTIVITY_ID_KEY].shift(1, fill_value=df.at[0, ACTIVITY_ID_KEY])
    df[aux_case] = df[CASE_ID_KEY].shift(1, fill_value=df.at[0, CASE_ID_KEY])
    df[aux_timestamp] = df[TIMESTAMP].shift(1, fill_value=df.at[0, TIMESTAMP])
    df[aux_lifecycle] = df[LIFECYCLE].shift(1, fill_value=df.at[0, LIFECYCLE])

    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP], format='ISO8601')
    df[aux_timestamp] = pd.to_datetime(df[aux_timestamp], format='ISO8601')

    df = df[df[LIFECYCLE].str.lower() == 'complete']
    with_start = (~(df[aux_lifecycle].str.lower() == 'complete')
                  & (df[aux_case] == df[CASE_ID_KEY])
                  & (df[aux_act] == df[ACTIVITY_ID_KEY]))
    df.loc[with_start, FROM_KEY] = df[with_start][aux_timestamp]
    df.loc[~with_start, FROM_KEY] = df[~with_start][TIMESTAMP]
    df[UNTIL_KEY] = df[TIMESTAMP]

    df = df.drop([aux_act, aux_timestamp, aux_case, aux_lifecycle], axis=1)

    df[FROM_KEY] = pd.to_datetime(df[FROM_KEY])
    df[UNTIL_KEY] = pd.to_datetime(df[UNTIL_KEY])
    df[DURATION_KEY] = df[UNTIL_KEY] - df[FROM_KEY]

    return df

class TimedOneGramBag:
    def __init__(self, interval: Timedelta, exponential: bool = False):
        self._interval = interval
        self._exponential = exponential

    def classify(self, sample: DataFrame) -> Series:
        duration_factor = 'factor_column_' + str(time.time())
        duration_bag = 'bag_column_' + str(time.time())
        count = 'count_column_' + str(time.time())

        if not (DURATION_KEY in sample.columns):
            if has_lifecycle_info(sample):
                sample = lclog_to_durationlog(sample)
            else:
                sample = log_to_durationlog(sample)

        sample[duration_factor] = sample[DURATION_KEY] / self._interval
        if self._exponential:
            np.seterr(divide='ignore')
            sample[duration_bag] = np.ceil(np.log2(sample[duration_factor]))
            np.seterr(divide='warn')
            sample.loc[(sample[duration_bag] < 0) & ~(sample[duration_bag] == -np.inf), duration_bag] = 0
        else:
            sample[duration_bag] = np.ceil(sample[duration_factor])

        df: DataFrame = sample.groupby([CASE_ID_KEY, ACTIVITY_ID_KEY, duration_bag]).size() \
            .reset_index() \
            .rename(columns={0: count})
        df[count] = 1

        s = df.groupby([ACTIVITY_ID_KEY, duration_bag]).sum(numeric_only=True).squeeze(axis='columns')
        s.index = s.index.to_flat_index()
        return s

def load_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    return df

def convert_to_duration_log(df):
    if has_lifecycle_info(df):
        duration_log = lclog_to_durationlog(df)
    else:
        duration_log = log_to_durationlog(df)
    return duration_log

def classify_uniform_species(duration_log, lambda_value):
    classifier = TimedOneGramBag(interval=pd.Timedelta(minutes=lambda_value), exponential=False)
    species = classifier.classify(duration_log)
    return species

def classify_exponential_species(duration_log, lambda_value):
    classifier = TimedOneGramBag(interval=pd.Timedelta(minutes=lambda_value), exponential=True)
    species = classifier.classify(duration_log)
    return species

def save_species_to_csv(species, file_name):
    species_df = species.reset_index()

    if species_df.shape[1] == 2:
        species_df.columns = ['Activity', 'Duration Bin']
    elif species_df.shape[1] == 3:
        species_df.columns = ['Activity', 'Duration Bin', 'Count']

    species_df.to_csv(file_name, index=False)

def main(input_csv):
    event_log = load_data(input_csv)
    duration_log = convert_to_duration_log(event_log)

    lambda_values = [1, 5, 30]
    for lambda_value in lambda_values:
        species = classify_uniform_species(duration_log, lambda_value)
        num_species = len(species)
        print(f"Uniform Duration-based Species ζt{lambda_value}: {num_species} species found")
        save_species_to_csv(species, f'uniform_duration_species_zt{lambda_value}.csv')

    lambda_value = 1.5
    exp_species = classify_exponential_species(duration_log, lambda_value)
    num_exp_species = len(exp_species)
    print(f"Exponential Duration-based Species ζte2: {num_exp_species} species found")
    save_species_to_csv(exp_species, 'exponential_duration_species_zte2.csv')

if __name__ == "__main__":
    input_csv = "/kaggle/input/bpi-2019-grouped-sample/BPI-2019_grouped_sample_50 (1).csv"
    main(input_csv)
