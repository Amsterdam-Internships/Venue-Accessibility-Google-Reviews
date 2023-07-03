import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from dotenv import load_dotenv
from train import my_pipeline

# Load environment variables from .env file
load_dotenv(override=True)

def extract_metrics():
    eval_metrics = pd.read_csv(load_path)

    # Get the metric scores for the single row
    accuracy_score = eval_metrics['Accuracy'].values[0]
    precision_score = eval_metrics['Precision'].values[0]
    recall_score = eval_metrics['Recall'].values[0]
    f1_score = eval_metrics['F1-Score'].values[0]

    plot_metrics(accuracy_score, precision_score, recall_score, f1_score)

def plot_metrics(accuracy_score, precision_score, recall_score, f1_score):
    # Create a bar plot for all four metrics
    plt.figure(figsize=(8, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    scores = [accuracy_score, precision_score, recall_score, f1_score]
    plt.bar(metrics, scores)
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title(f'Evaluation Metrics of {names}')
    #TODO Add the model names to the file name
    plt.savefig(save_path +f'{names}_evaluation_metrics.png')
    plt.close()


if __name__ == '__main__':
    names = my_pipeline.model_name.split("/")[-1] if "/" in my_pipeline.model_name else [my_pipeline.model_name]
    save_path = os.getenv('LOCAL_ENV') + '/results/aspect_classification/'
    load_path = os.getenv('LOCAL_ENV') + f'/results/aspect_classification/{names}.csv'
    extract_metrics()