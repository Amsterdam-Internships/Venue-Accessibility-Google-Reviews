'''
This is a script to use trained models to make predictions.
'''
from sklearn.metrics import classification_report, confusion_matrix
import sys
sys.path.append('/Users/mylene/BachelorsProject/Venue-Accessibility-Google-Reviews/src')
from aspect_classification.data.data_cleaning import bert_processing
from sklearn.preprocessing import LabelEncoder
from pipelines import MyPipeline
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import yaml
import os

# Load environment variables from .env file
load_dotenv()

with open('src/aspect_classification/models/config.yml', 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
    

    
my_pipeline = MyPipeline(pipeline_type=params['pipeline_type'], bert_model=params['model_name'])

def evaluate(y_true, y_pred, labels):
    # Check for missing labels
    missing_labels = set(labels) - set(y_true)
    if missing_labels:
        raise ValueError(f"The following labels are missing from y_true: {missing_labels}")
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    return report, confusion

def remove_rows(test_data):
    # Remove the redundant rows
    test_data = test_data.dropna(subset=['Aspect Label'])
    test_data = test_data[test_data['Aspect Label'] != 'Nonsense']
    return test_data

def encode_input(data):
    encoder = LabelEncoder()
    y_true = np.array(encoder.fit_transform(data))
    return y_true
    
def generate_results():
    test_data = pd.read_excel(test_data_path)
    selected_rows = remove_rows(test_data)
    # Select the input features
    google_reviews = selected_rows['Text'].values.tolist()
    # Select target labels
    google_labels = selected_rows['Sentiment Label'].values.tolist()
    # Process reviews
    processed_reviews = bert_processing(google_reviews)
    y_pred = my_pipeline.predict(processed_reviews)
    y_true = encode_input(google_labels)

    # Convert y_true and y_pred to lists of strings
    y_true = [str(label) for label in y_true]
    y_pred = [str(label) for label in y_pred]
    labels = ['Positive', 'Negative']
    eval_metrics, confusion_graph = evaluate(y_true, y_pred, labels)
    
    # Convert y_pred to a pandas DataFrame
    predicted_labels_df = pd.DataFrame({'Predicted Sentiment Labels': y_pred})

    # Save the predicted labels as a CSV file
    predicted_labels_path = interim_path + "/predicted_sentiment_labels.csv"
    predicted_labels_df.to_csv(predicted_labels_path, index=False)
    
    # # Save the classification report as a text file
    report_path = results_path+"_classification_report.tex"
    report_df = pd.DataFrame(eval_metrics).transpose()
    report_df.to_latex(report_path)

    print(f"Classification report saved as {results_path}")

    # Visualize the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_graph, annot=True, cmap='Blues', fmt='d', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(results_path + '_confusion_matrix.png')

if __name__ == '__main__':
    # Get the file paths from environment variables
    test_data_path = os.getenv('LOCAL_ENV') + 'data/interim/predicted_labels.csv'
    loaded_model_path = os.getenv('LOCAL_ENV') + 'models/sentiment_analysis/bert.joblib'
    # Define the directory path
    names = params['model_name'].split("/")
    results_path = os.getenv('LOCAL_ENV') + f"results/sentiment_classification/{names[1]}"
    interim_path = os.getenv('LOCAL_ENV') + 'data/interim'

    # Call the function to generate results
    generate_results()
