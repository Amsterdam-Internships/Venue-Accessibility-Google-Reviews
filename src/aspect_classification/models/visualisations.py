import matplotlib.pyplot as plt
import pandas as pd
import os

eval_metrics = pd.read_json(save_path)

# Extract the ROUGE metric names and corresponding scores
metrics = rouge_df.columns
precision_scores = rouge_df.loc[0, ['precision']]
recall_scores = rouge_df.loc[0, ['recall']]
f1_scores = rouge_df.loc[0, ['f1']]

# Create a bar plot for precision scores
plt.figure(figsize=(8, 6))
plt.bar(metrics, precision_scores)
plt.xlabel('ROUGE Metric')
plt.ylabel('Precision')
plt.title('ROUGE Precision Scores')
plt.savefig('precision_scores.png')
plt.close()

# Create a bar plot for recall scores
plt.figure(figsize=(8, 6))
plt.bar(metrics, recall_scores)
plt.xlabel('ROUGE Metric')
plt.ylabel('Recall')
plt.title('ROUGE Recall Scores')
plt.savefig('recall_scores.png')
plt.close()

# Create a bar plot for F1 scores
plt.figure(figsize=(8, 6))
plt.bar(metrics, f1_scores)
plt.xlabel('ROUGE Metric')
plt.ylabel('F1-Score')
plt.title('ROUGE F1-Scores')
plt.savefig('f1_scores.png')
plt.close()

if __name__ == '__main__':
    save_path = os.get_env('LOCAL_ENV') + 'results/opinion_summarisation/'