'''
This is a script to create tables and results visualizations.
'''

import os
import yaml
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval

# Load environment variables from .env file
load_dotenv()

with open('src/aspect_classification/models/config.yml', 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

# Read the evaluation results from file using pandas
results_path = os.getenv('LOCAL_ENV') + "results/aspect_classification/TinyBERT_General_4L_312D.csv"
results = pd.read_csv(results_path)

# Extract the evaluation metrics and confusion matrix from the results dataframe
accuracy = results['accuracy'][0]
precision = results['precision'][0]
recall = results['recall'][0]
f1 = results['f1'][0]
confusion_str = results['confusion_matrix'][0]

# Parse the confusion matrix
confusion_matrix = literal_eval(confusion_str)

# Generate synthetic y_true and y_pred for illustration purposes
y_true = np.random.randint(0, 3, size=len(confusion_matrix))
y_pred = np.random.randint(0, 3, size=len(confusion_matrix))

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(np.unique(y_true)))
plt.xticks(tick_marks, np.unique(y_true), rotation=45)
plt.yticks(tick_marks, np.unique(y_true))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Display the metrics as text
metrics_text = f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1-Score: {f1:.2f}"
plt.text(0.5, -0.15, metrics_text, transform=plt.gca().transAxes, ha='center', fontsize=12)

# Save the graph as a PNG image
graphs_path = os.getenv('LOCAL_ENV') + "results/aspect_classification/graph.png"
plt.tight_layout()
plt.savefig(graphs_path)

plt.show()
