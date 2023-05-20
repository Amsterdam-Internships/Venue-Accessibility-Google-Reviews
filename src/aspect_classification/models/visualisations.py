import matplotlib.pyplot as plt
import pandas as pd
import os

def extract_metrics():
    eval_metrics = pd.read_json(save_path)
    # Extract the ROUGE metric names and corresponding scores
    metrics = eval_metrics.columns
    precision_scores = eval_metrics.loc[0, ['precision']]
    recall_scores = eval_metrics.loc[0, ['recall']]
    f1_scores = eval_metrics.loc[0, ['f1']]
    plot_metric(metrics, precision_scores, 'precision')
    plot_metric(metrics, recall_scores, 'recall')
    plot_metric(metrics, f1_scores, 'f1')

def plot_metric(metrics, metric_type, metric_name):
    # Create a bar plot for precision scores
    plt.figure(figsize=(8, 6))
    plt.bar(metrics, metric_type)
    plt.xlabel('ROUGE Metric')
    plt.ylabel('{metric_name}'.format(metric_name))
    plt.title('ROUGE {metric_name} Scores'.format(metric_name))
    plt.savefig('{metric_name}_scores.png'.format(metric_name))
    plt.close()

if __name__ == '__main__':
    save_path = os.get_env('LOCAL_ENV') + 'results/opinion_summarisation/'
    extract_metrics()