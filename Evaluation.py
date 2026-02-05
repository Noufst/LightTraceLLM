import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, fbeta_score
import re

def natural_sort_key(filename):
    """Helper function for natural sorting of filenames."""
    # Split the filename into text and numeric components, sort by numeric values if available
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

def evaluate_results(results_directory):
    """
    Evaluate the results for a specific experiment, plot an HD bar chart of selected metrics, 
    and return both average and standard deviation metrics across all files.
    """
    print(f"[INFO] Starting evaluation of: {results_directory}")

    results = []

    # Loop through all CSV files in the specified directory
    for filename in os.listdir(results_directory):
        if filename.endswith('.csv') and 'performance_results.csv' not in filename and 'similarity_scores' not in filename and 'selected' not in filename:
            file_path = os.path.join(results_directory, filename)
            #
            # Read the CSV file
            df = pd.read_csv(file_path)
            print(df.columns)
            actual = df['label']
            predicted = df['predicted_label']
            
            
            # Calculate metrics
            accuracy = accuracy_score(actual, predicted)
            precision = precision_score(actual, predicted, average='binary', zero_division=1)
            recall = recall_score(actual, predicted, average='binary', zero_division=1)
            f1 = f1_score(actual, predicted, average='binary', zero_division=1)
            f2 = fbeta_score(actual, predicted, beta=2, average='binary', zero_division=1)

            # Append results to the list
            results.append({
                'filename': filename,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'f2_score': f2
            })
            print(f"[DEBUG] Metrics for {filename}: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1={f1}, F2={f2}")

    # Convert results list to a DataFrame
    results_df = pd.DataFrame(results)

    # Calculate averages and standard deviations for each metric
    summary_stats = []
    summary_stats.append({
        'filename': 'Average',
        'accuracy': results_df['accuracy'].mean(), 
        'precision': results_df['precision'].mean(),
        'recall': results_df['recall'].mean(),
        'f1_score': results_df['f1_score'].mean(),
        'f2_score': results_df['f2_score'].mean()
    })
    summary_stats.append({
        'filename': 'Std',
        'accuracy': results_df['accuracy'].std(), 
        'precision': results_df['precision'].std(),
        'recall': results_df['recall'].std(),
        'f1_score': results_df['f1_score'].std(),
        'f2_score': results_df['f2_score'].std()
    })

    results_df = pd.concat([results_df, pd.DataFrame(summary_stats)], ignore_index=True)

    # Save the results to a CSV file
    performance_results_file = os.path.join(results_directory, 'performance_results.csv')
    results_df.to_csv(performance_results_file, index=False)
    print(f"[INFO] Performance evaluation results saved to \"{performance_results_file}\".")

    # Bar chart visualization
    metrics = ['precision', 'recall', 'f1_score', 'f2_score']
    num_files = len(results_df)
    num_metrics = len(metrics)

    # Plotting grouped bar chart
    bar_width = 0.15
    index = np.arange(num_files)
    print("[INFO] Creating bar chart for performance metrics...")

    plt.figure(figsize=(16, 8), dpi=300)
    for i, metric in enumerate(metrics):
        bars = plt.bar(index + i * bar_width, results_df[metric], bar_width, label=metric.capitalize())
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    plt.xlabel('Filename', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.xticks(index + bar_width * (num_metrics / 2), results_df['filename'], rotation=45, ha='right', fontsize=10)
    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    plt.gca().set_ylim(0, 1)
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.05))
    plt.legend(fontsize=12)
    plt.tight_layout()

    plot_file = os.path.join(results_directory, 'results.png')
    plt.savefig(plot_file, dpi=300)
    plt.close()  # Avoid interruption by closing the figure
    print(f"[INFO] The results bar chart saved as \"{plot_file}\".")


# Example usage:
#evaluate_results("Results/TLC/_similarity/CM1_NASA")
#evaluate_results("Results/TLC/google/gemma-3-12b-it/diverse/EasyClinic_UC_TC_1764596679_5/role_1_shot_2")
#evaluate_results("Results/TLC/_ensemble_roles/CM1_NASA/Merged")

#evaluate_results("/Users/nouf/Library/CloudStorage/OneDrive-KFUPM/PhD_Thesis_Nouf/Coding/TraceLLM_Agents/Results/TLC/meta-llama/llama-4-maverick/diverse/EasyClinic_UC_TC_1763383431_1/role_1_shot_2")