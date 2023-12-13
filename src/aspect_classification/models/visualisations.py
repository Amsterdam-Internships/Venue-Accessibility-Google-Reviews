import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from dotenv import load_dotenv
import yaml
import seaborn as sns


config_path = os.getenv('LOCAL_ENV') + '/src/aspect_classification/models/config.yml'

with open(config_path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            bert_params = params['bert_params']
# Load environment variables from .env file
load_dotenv(override=True)

def extract_metrics():
    eval_metrics = pd.read_csv(load_path)
    # Get the metric scores for the single row
    total_loss = eval_metrics['eval_loss'].values[0]
    precision_score = eval_metrics['eval_precision'].values[0]
    recall_score = eval_metrics['eval_recall'].values[0]
    f1_score = eval_metrics['eval_f1 score'].values[0]

    plot_metrics(total_loss, precision_score, recall_score, f1_score)

def plot_metrics(total_loss, precision_score, recall_score, f1_score):
    # Create a bar plot for all four metrics
    plt.figure(figsize=(8, 6))
    metrics = ['loss', 'Precision', 'Recall', 'F1-Score']
    scores = [total_loss, precision_score, recall_score, f1_score]
    plt.bar(metrics, scores)
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title(f'Evaluation Metrics of {names}')
    plt.savefig(save_path +f'{names}_evaluation_metrics.png')
    
    confusion_matrix_df = pd.read_csv(confusion_matrix_path)  # Assuming 'Class' is the index column

    # Plot a heatmap
    plt.figure(figsize=(8, 6))
    heatmap = sns.heatmap(confusion_matrix_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Multi-Label Confusion Matrix")

    # Save the heatmap data to a new CSV file
    heatmap_data = pd.DataFrame(heatmap.get_array(), index=confusion_matrix_df.index, columns=confusion_matrix_df.columns)
    heatmap_data.to_csv('heatmap_data.csv')

    plt.close()


if __name__ == '__main__':
    names = bert_params['model_name_or_path'].split("/")[-1] if "/" in bert_params['model_name_or_path'] else bert_params['model_name_or_path']
    save_path = os.getenv('LOCAL_ENV') + '/results/aspect_classification/'
    load_path = os.getenv('LOCAL_ENV') + f'/results/aspect_classification/{names}.csv'
    confusion_matrix_path = os.getenv('LOCAL_ENV') + '/logs/aspect_classification/confusion_matrix.csv'
    extract_metrics()