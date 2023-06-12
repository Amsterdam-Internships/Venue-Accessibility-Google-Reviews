import matplotlib.pyplot as plt
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
    fig, ax = plt.subplots()

    # Plot the Rouge scores
    ax.bar(bar_positions, rouge_scores[0], width=bar_width, label='ROUGE-1')
    ax.bar([pos + bar_width for pos in bar_positions], rouge_scores[1], width=bar_width, label='ROUGE-2')
    ax.bar([pos + 2 * bar_width for pos in bar_positions], rouge_scores[2], width=bar_width, label='ROUGE-L')

    # Customize the plot
    ax.set_ylabel('%')
    ax.set_title('The Recall-Orieented Understudy for Gisting Evaluation (ROUGE) Scores')
    ax.legend()

    # Rotate the x-axis labels for better readability
    plt.xticks([pos + bar_width for pos in bar_positions], bar_positions, rotation=90)

    # Save the plot as a PNG file
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == '__main__':
    save_path = os.getenv('LOCAL_ENV') + 'results/opinion_summarisation/eval_metrics.png'
    load_path = os.getenv('LOCAL_ENV') + 'results/opinion_summarisation/eval_metrics.csv'
    extract_metrics()
