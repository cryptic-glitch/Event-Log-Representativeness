import pandas as pd
import os
import random


class LSMbrBootstrapGeneralization:
    def __init__(self, input_csv, output_dir, sample_size, num_samples, generations, subtrace_length, breeding_prob):
        self.input_csv = input_csv
        self.output_dir = output_dir
        self.sample_size = sample_size
        self.num_samples = num_samples
        self.generations = generations
        self.subtrace_length = subtrace_length
        self.breeding_prob = breeding_prob
        self.event_log = pd.read_csv(input_csv)
        self.traces = self.event_log.groupby('case:concept:name')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def generate_samples(self):
        for i in range(self.num_samples):
            sampled_log = self.log_sampling_with_breeding()
            output_file = os.path.join(self.output_dir, f'sample_{i + 1}.csv')
            sampled_log.to_csv(output_file, index=False)
            print(f'Sample {i + 1} saved to {output_file}')

    def log_sampling_with_breeding(self):
        sampled_log = pd.DataFrame(columns=self.event_log.columns)
        for _ in range(self.generations):
            case_ids = random.sample(list(self.traces.groups.keys()), k=self.sample_size)
            traces = [self.traces.get_group(cid) for cid in case_ids]
            for j in range(0, len(traces) - 1, 2):
                if random.random() < self.breeding_prob:
                    trace1, trace2 = traces[j], traces[j + 1]
                    new_trace = self.crossover_subtrace(trace1, trace2)
                    sampled_log = pd.concat([sampled_log, new_trace])
                else:
                    sampled_log = pd.concat([sampled_log, traces[j]])
        return sampled_log

    def crossover_subtrace(self, trace1, trace2):
        trace1_events = trace1.to_dict('records')
        trace2_events = trace2.to_dict('records')
        len1 = min(len(trace1_events), self.subtrace_length)
        len2 = min(len(trace2_events), self.subtrace_length)
        subtrace1 = trace1_events[:len1]
        subtrace2 = trace2_events[:len2]
        new_trace_events = subtrace1 + subtrace2
        new_trace_df = pd.DataFrame(new_trace_events)
        return new_trace_df

    def run(self):
        print(f"Generating {self.num_samples} bootstrapped samples with breeding...")
        self.generate_samples()
        print("Sample generation complete.")


if __name__ == "__main__":
    input_csv = "/kaggle/input/final-dataset-thesis/dataset_csv/Sepsis.csv"
    output_dir = "/kaggle/working/"
    sample_size = 100
    num_samples = 10
    generations = 5
    subtrace_length = 10
    breeding_prob = 0.5
    bootstrap = LSMbrBootstrapGeneralization(input_csv, output_dir, sample_size, num_samples, generations,
                                             subtrace_length, breeding_prob)
    bootstrap.run()
