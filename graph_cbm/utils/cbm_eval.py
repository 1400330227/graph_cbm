import sklearn.metrics as metrics
import numpy as np


class CBMEvaluator:
    def __init__(self, n_tasks=200):
        self.result_dict = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        self.result_means = {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0
        }
        self.n_tasks = n_tasks
        self.average = 'binary' if n_tasks == 2 else 'macro'

    def reset(self):
        """Reset all stored metrics"""
        for key in self.result_dict:
            self.result_dict[key] = []

    def compute_bin_accuracy(self, y_probs, y_true):
        y_pred = y_probs.argmax(dim=-1).cpu().detach().numpy()
        y_true = y_true.astype(int)  # Ensure y_true is integer

        # Calculate metrics
        accuracy = metrics.accuracy_score(y_true, y_pred)
        precision = metrics.precision_score(y_true, y_pred, average=self.average, zero_division=0)
        recall = metrics.recall_score(y_true, y_pred, average=self.average, zero_division=0)
        f1 = metrics.f1_score(y_true, y_pred, average=self.average, zero_division=0)

        # Store results
        self.result_dict['accuracy'].append(accuracy)
        self.result_dict['precision'].append(precision)
        self.result_dict['recall'].append(recall)
        self.result_dict['f1'].append(f1)

    def print_results(self):
        """Print aggregated results to console with formatting"""
        if not any(len(v) > 0 for v in self.result_dict.values()):
            print("No results available. Run compute_bin_accuracy() first.")
            return

        print("Classification Metrics Summary:")
        # print("=" * 40)
        for metric, values in self.result_dict.items():
            if values:  # Only print if there are values
                avg_value = np.mean(values)
                self.result_means[metric] = avg_value
                std_value = np.std(values)
                print(f"{metric.capitalize():<10}: {avg_value:.4f} ± {std_value:.4f} (mean ± std)")
        # print("=" * 40)
