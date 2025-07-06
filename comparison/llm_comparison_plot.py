import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuration: IMPORTANT! Replace with your actual JSON file paths ---
LLM_NAMES = ["gemini2.0", "llama3.2:8b", "mistral:7b"] # Give meaningful names to your LLMs
JSON_FILES = [
    "../results_english/gemini_results/visualizations/field_status_summary/aggregate_metrics.json", # <--- REPLACE THIS WITH THE ACTUAL PATH TO YOUR FIRST LLM's JSON
    "../results_english/llama_results/visualizations/field_status_summary/aggregate_metrics.json", # <--- REPLACE THIS WITH THE ACTUAL PATH TO YOUR SECOND LLM's JSON
    "../results_english/mistral_results/visualizations/field_status_summary/aggregate_metrics.json"  # <--- REPLACE THIS WITH THE ACTUAL PATH TO YOUR THIRD LLM's JSON
]

def load_and_flatten_data(json_files, llm_names):
    """
    Loads data from multiple JSON files and flattens it into a list of dictionaries.
    Each dictionary represents an LLM's evaluation results.
    """
    all_llm_data = []
    for i, file_path in enumerate(json_files):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: JSON file not found at {file_path}. Please check the path.")
            continue
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}. Check file format.")
            continue

        flat_data = {"LLM": llm_names[i]}

        # Directly extract top-level metrics
        for key, value in data.items():
            if isinstance(value, (int, float)):
                flat_data[key] = value
            elif isinstance(value, dict):
                # Handle nested dictionaries
                if key == "completeness_metrics":
                    flat_data["mean_completeness_score"] = value.get("mean_completeness_score")
                    # flat_data["std_completeness_score"] = value.get("std_completeness_score") # Can add if needed for error bars
                elif key == "hallucination_metrics":
                    flat_data["overall_hallucination_percentage"] = value.get("overall_hallucination_percentage")
                    # Field-wise hallucination will be handled separately due to its structure
                    # Store field-wise as a dictionary within the flat_data for later processing
                    flat_data["field_wise_hallucination_percentage"] = value.get("field_wise_hallucination_percentage", {})
                elif key == "missing_from_llm_metrics":
                    flat_data["total_missing_from_llm_fields"] = value.get("total_missing_from_llm_fields")
                    flat_data["mean_missing_from_llm_fields_per_record"] = value.get("mean_missing_from_llm_fields_per_record")
                elif key == "correct_fields_metrics":
                    flat_data["total_correct_fields"] = value.get("total_correct_fields")
                    flat_data["mean_correct_fields_per_record"] = value.get("mean_correct_fields_per_record")
                elif key == "incorrect_fields_metrics":
                    flat_data["total_incorrect_fields"] = value.get("total_incorrect_fields")
                    flat_data["mean_incorrect_fields_per_record"] = value.get("mean_incorrect_fields_per_record")
        all_llm_data.append(flat_data)
    return pd.DataFrame(all_llm_data)

def plot_metrics_comparison(df, metrics, title_prefix, ylabel, filename):
    """
    Plots a bar chart comparing LLMs for a given set of metrics.
    """
    # Ensure all specified metrics exist in the DataFrame
    valid_metrics = [m for m in metrics if m in df.columns]
    if not valid_metrics:
        print(f"Warning: No valid metrics found in DataFrame for {title_prefix}. Skipping plot.")
        return

    # Filter columns to only include the metrics of interest and 'LLM'
    plot_df = df[['LLM'] + valid_metrics].melt(id_vars='LLM', var_name='Metric', value_name='Score')

    plt.figure(figsize=(12, 7))
    sns.barplot(x='Metric', y='Score', hue='LLM', data=plot_df, palette='viridis')
    plt.title(f'{title_prefix} Comparison')
    plt.ylabel(ylabel)
    plt.xlabel('Metric')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='LLM')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{filename}.png")
    plt.close()

def plot_field_wise_hallucination(raw_llm_data, llm_names, filename="field_wise_hallucination_comparison"):
    """
    Plots field-wise hallucination percentages for all LLMs.
    Takes a list of individual LLM raw data dictionaries to handle nested structure.
    """
    all_field_hallucination_data = []
    for i, data_dict in enumerate(raw_llm_data):
        llm_name = llm_names[i]
        field_hallucination = data_dict.get("hallucination_metrics", {}).get("field_wise_hallucination_percentage", {})
        for field, percentage in field_hallucination.items():
            all_field_hallucination_data.append({
                "LLM": llm_name,
                "Field": field,
                "Hallucination Percentage": percentage
            })

    if not all_field_hallucination_data:
        print("No field-wise hallucination data to plot.")
        return

    plot_df = pd.DataFrame(all_field_hallucination_data)

    plt.figure(figsize=(15, 8))
    sns.barplot(x='Field', y='Hallucination Percentage', hue='LLM', data=plot_df, palette='magma')
    plt.title('Field-wise Hallucination Percentage Comparison')
    plt.ylabel('Hallucination Percentage (%)')
    plt.xlabel('Field')
    plt.xticks(rotation=90, ha='right')
    plt.legend(title='LLM')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{filename}.png")
    plt.close()


# --- Main Execution ---
if __name__ == "__main__":
    # Load all raw JSON data for special handling of nested structures like field_wise_hallucination
    raw_llm_data = []
    for file_path in JSON_FILES:
        try:
            with open(file_path, 'r') as f:
                raw_llm_data.append(json.load(f))
        except FileNotFoundError:
            print(f"Skipping {file_path} as it was not found.")
            # If a file is missing, we need to ensure the raw_llm_data and LLM_NAMES lists remain consistent
            # A more robust solution might involve removing the LLM from LLM_NAMES entirely
            # For simplicity here, we'll just skip and print a warning.
            # You might want to implement more robust error handling for production.
        except json.JSONDecodeError:
            print(f"Skipping {file_path} due to JSON decoding error.")

    if len(raw_llm_data) != len(JSON_FILES):
        print("Warning: Mismatch between number of loaded JSON files and expected files. Plots might be incomplete.")

    # Prepare data for general plotting (flattened version)
    # We pass JSON_FILES and LLM_NAMES directly for consistent handling
    df_comparison = load_and_flatten_data(JSON_FILES, LLM_NAMES)

    if df_comparison.empty:
        print("No data loaded for comparison. Exiting.")
    else:
        # --- Define Metric Categories for Plotting ---

        # Strict Accuracy Metrics
        strict_accuracy_metrics = [col for col in df_comparison.columns if "_strict_accuracy" in col]
        plot_metrics_comparison(df_comparison, strict_accuracy_metrics, "Strict Accuracy", "Accuracy Score", "strict_accuracy_comparison")

        # Fuzzy Accuracy Metrics
        fuzzy_accuracy_metrics = [col for col in df_comparison.columns if "_fuzzy_accuracy" in col]
        plot_metrics_comparison(df_comparison, fuzzy_accuracy_metrics, "Fuzzy Accuracy", "Accuracy Score", "fuzzy_accuracy_comparison")

        # Jaccard Similarity Metrics
        jaccard_metrics = [col for col in df_comparison.columns if "_jaccard_mean" in col]
        plot_metrics_comparison(df_comparison, jaccard_metrics, "Jaccard Similarity (Mean)", "Jaccard Score", "jaccard_similarity_comparison")

        # BLEU Similarity Metrics
        bleu_metrics = [col for col in df_comparison.columns if "_bleu_mean" in col]
        plot_metrics_comparison(df_comparison, bleu_metrics, "BLEU Similarity (Mean)", "BLEU Score", "bleu_similarity_comparison")

        # ROUGE-1 Similarity Metrics
        rouge1_metrics = [col for col in df_comparison.columns if "_rouge_1_mean" in col]
        plot_metrics_comparison(df_comparison, rouge1_metrics, "ROUGE-1 Similarity (Mean)", "ROUGE-1 Score", "rouge1_similarity_comparison")

        # ROUGE-2 Similarity Metrics
        rouge2_metrics = [col for col in df_comparison.columns if "_rouge_2_mean" in col]
        plot_metrics_comparison(df_comparison, rouge2_metrics, "ROUGE-2 Similarity (Mean)", "ROUGE-2 Score", "rouge2_similarity_comparison")

        # ROUGE-L Similarity Metrics
        rougel_metrics = [col for col in df_comparison.columns if "_rouge_l_mean" in col]
        plot_metrics_comparison(df_comparison, rougel_metrics, "ROUGE-L Similarity (Mean)", "ROUGE-L Score", "rougel_similarity_comparison")

        # LLM Similarity Metrics
        llm_similarity_metrics = [col for col in df_comparison.columns if "_llm_similarity_mean" in col and "binary" not in col]
        plot_metrics_comparison(df_comparison, llm_similarity_metrics, "LLM Similarity (Mean)", "LLM Similarity Score", "llm_similarity_comparison")

        # LLM Binary Similarity Metrics
        llm_binary_similarity_metrics = [col for col in df_comparison.columns if "_llm_binary_similarity_mean" in col]
        plot_metrics_comparison(df_comparison, llm_binary_similarity_metrics, "LLM Binary Similarity (Mean)", "LLM Binary Similarity Score", "llm_binary_similarity_comparison")

        # Completeness Metrics
        completeness_metrics = ["mean_completeness_score"]
        plot_metrics_comparison(df_comparison, completeness_metrics, "Completeness", "Score (%)", "completeness_comparison")

        # Hallucination Metrics (Overall)
        overall_hallucination_metrics = ["overall_hallucination_percentage"]
        plot_metrics_comparison(df_comparison, overall_hallucination_metrics, "Overall Hallucination Percentage", "Percentage (%)", "overall_hallucination_comparison")

        # Field-wise Hallucination (special handling due to nested dictionary)
        # We use the raw_llm_data here because 'field_wise_hallucination_percentage' is a nested dict
        plot_field_wise_hallucination(raw_llm_data, LLM_NAMES)

        # Summary Metrics (can be grouped or plotted individually as needed)
        summary_metrics = [
            "total_hallucinated_fields",
            "mean_hallucinated_fields_per_record",
            "total_missing_from_llm_fields",
            "mean_missing_from_llm_fields_per_record",
            "total_correct_fields",
            "mean_correct_fields_per_record",
            "total_incorrect_fields",
            "mean_incorrect_fields_per_record"
        ]
        plot_metrics_comparison(df_comparison, summary_metrics, "Summary Metrics", "Count/Score", "summary_metrics_comparison")

        print(f"Generated comparison plots for {len(LLM_NAMES)} LLMs.")
        print("Please check the current directory for the generated .png files.")