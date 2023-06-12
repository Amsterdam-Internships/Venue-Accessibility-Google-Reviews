import matplotlib.pyplot as plt
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def extract_metrics():
    eval_metrics = pd.read_csv(load_path)
    eval_metrics = eval_metrics.drop('Unnamed: 0', axis=1)

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
    plt.title('Evaluation Metrics')
    plt.savefig(save_path + 'evaluation_metrics.png')
    plt.close()




if __name__ == '__main__':
    save_path = os.getenv('LOCAL_ENV') + 'results/sentiment_classification/'
    load_path = os.getenv('LOCAL_ENV') + 'results/sentiment_classification/TinyBERT_General_4L_312D.csv'
    extract_metrics()