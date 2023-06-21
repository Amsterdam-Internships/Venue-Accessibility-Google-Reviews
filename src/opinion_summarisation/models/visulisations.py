import matplotlib.pyplot as plt
import pandas as pd
import os
from dotenv import load_dotenv
import yaml
from evaluate import pipeline

# Load environment variables from .env file
load_dotenv()

config_path = os.getenv('LOCAL_ENV') + 'src/opinion_summarisation/models/config.yml'

with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
def extract_metrics():
    eval_metrics = pd.read_csv(load_path)
    # Extract the Rouge scores
    rouge_1_scores = eval_metrics['rouge-1']
    rouge_2_scores = eval_metrics['rouge-2']
    rouge_l_scores = eval_metrics['rouge-l']
    scores = [rouge_1_scores, rouge_2_scores, rouge_l_scores]
    plot_metrics(scores)

def plot_metrics(rouge_scores):
    # Set the bar positions
    bar_positions = range(len(rouge_scores[0]))

    # Set the bar widths
    bar_width = 0.2

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the Rouge scores
    ax.bar(bar_positions, rouge_scores[0], width=bar_width, label='Precision')
    ax.bar([pos + bar_width for pos in bar_positions], rouge_scores[1], width=bar_width, label='Recall')
    ax.bar([pos + 2 * bar_width for pos in bar_positions], rouge_scores[2], width=bar_width, label='F1-Score')

    # Customize the plot
    ax.set_ylabel('% of overalp between reference summaries and generated summaries', fontsize=12)
    ax.set_xlabel
    ax.set_title('The Recall-Oriented Understudy for Gisting Evaluation (ROUGE) Scores', fontsize=12)
    ax.legend()

    x_labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    plt.xticks([pos + bar_width for pos in bar_positions], x_labels, rotation=0, ha='center')  

    # Save the plot as a PNG file
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == '__main__':
    save_path = os.getenv('LOCAL_ENV') + f'results/opinion_summarisation/'+pipeline.model+'evaluation_metrics.png'
    load_path = os.getenv('LOCAL_ENV') + f'results/opinion_summarisation/'+pipeline.model+'_eval_metrics.csv'
    extract_metrics()
