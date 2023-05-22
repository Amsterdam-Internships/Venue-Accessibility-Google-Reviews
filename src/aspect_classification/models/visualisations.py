import matplotlib.pyplot as plt
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def extract_metrics():
    eval_metrics = pd.read_csv(load_path)
    # Extract the ROUGE metric names and corresponding scores
    metrics = eval_metrics.columns
    print(metrics)
    precision_scores = eval_metrics['hamming loss']
    recall_scores = eval_metrics['jaccard_score']
    f1_scores = eval_metrics['f1']
    plot_metric(metrics, precision_scores, 'hamming loss')
    plot_metric(metrics, recall_scores, 'jaccard_score')
    plot_metric(metrics, f1_scores, 'f1')

def plot_metric(metrics, metric_type, metric_name):
    # Create a bar plot for precision scores
    plt.figure(figsize=(8, 6))
    plt.bar(metrics, metric_type)
    plt.xlabel('ROUGE Metric')
    plt.ylabel('{metric_name}'.format(metric_name))
    plt.title('ROUGE {metric_name} Scores'.format(metric_name))
    plt.savefig(save_path + '{metric_name}_scores.png'.format(metric_name))
    plt.close()

if __name__ == '__main__':
    save_path = os.getenv('LOCAL_ENV') + 'results/aspect_classification/'
    load_path = os.getenv('LOCAL_ENV') + 'data/interim/predicted_aspect_labels.csv'
    extract_metrics()