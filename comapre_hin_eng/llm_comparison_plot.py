import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re # Import regex for cleaner metric names

# --- Configuration: IMPORTANT! Replace with your actual JSON file paths ---
LLM_NAMES = ["English mistral:7b", "Hindi mistral:7b"] # Give meaningful names to your LLMs
JSON_FILES = [
    "../results_english/mistral_results/visualizations/field_status_summary/aggregate_metrics.json",  # <--- REPLACE THIS WITH THE ACTUAL PATH TO YOUR THIRD LLM's JSON
    "../results_hindi/mistral_results/visualizations/field_status_summary/aggregate_metrics.json"
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
            print(f"Error: JSON file not found at {file_path}. Please check the path and ensure the file exists.")
            # Skip this LLM if its file is not found, ensuring consistency between LLM_NAMES and loaded data
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
                    flat_data["std_completeness_score"] = value.get("std_completeness_score") # Added std for potential error bars
                elif key == "hallucination_metrics":
                    flat_data["overall_hallucination_percentage"] = value.get("overall_hallucination_percentage")
                    # Field-wise hallucination will be handled separately due to its structure
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
    
    # Create DataFrame from successfully loaded data
    df = pd.DataFrame(all_llm_data)

    # --- Clean up metric names for better plotting ---
    renamed_columns = {}
    for col in df.columns:
        if col not in ['LLM', 'field_wise_hallucination_percentage', 'std_completeness_score']: # Exclude special columns
            clean_name = col.replace('_strict_accuracy', ' (Strict Acc.)') \
                             .replace('_fuzzy_accuracy', ' (Fuzzy Acc.)') \
                             .replace('_jaccard_mean', ' (Jaccard Mean)') \
                             .replace('_bleu_mean', ' (BLEU Mean)') \
                             .replace('_rouge_1_mean', ' (ROUGE-1 Mean)') \
                             .replace('_rouge_2_mean', ' (ROUGE-2 Mean)') \
                             .replace('_rouge_l_mean', ' (ROUGE-L Mean)') \
                             .replace('_llm_similarity_mean', ' (LLM Sim. Mean)') \
                             .replace('_llm_binary_similarity_mean', ' (LLM Binary Sim. Mean)') \
                             .replace('mean_completeness_score', 'Completeness Score') \
                             .replace('overall_hallucination_percentage', 'Overall Hallucination %') \
                             .replace('total_hallucinated_fields', 'Total Hallucinated Fields') \
                             .replace('mean_hallucinated_fields_per_record', 'Mean Hallucinated Fields / Record') \
                             .replace('total_missing_from_llm_fields', 'Total Missing Fields') \
                             .replace('mean_missing_from_llm_fields_per_record', 'Mean Missing Fields / Record') \
                             .replace('total_correct_fields', 'Total Correct Fields') \
                             .replace('mean_correct_fields_per_record', 'Mean Correct Fields / Record') \
                             .replace('total_incorrect_fields', 'Total Incorrect Fields') \
                             .replace('mean_incorrect_fields_per_record', 'Mean Incorrect Fields / Record')

            # Capitalize the first letter of the actual field name (e.g., 'name' -> 'Name')
            clean_name = re.sub(r'(\b\w)', lambda x: x.group(1).upper(), clean_name)
            clean_name = clean_name.replace('_', ' ') # Replace remaining underscores with spaces
            clean_name = clean_name.strip() # Remove any leading/trailing spaces

            renamed_columns[col] = clean_name
    
    df.rename(columns=renamed_columns, inplace=True)
    return df

def plot_metrics_comparison(df, metrics, title_prefix, ylabel, filename, y_lim=None):
    """
    Plots a bar chart comparing LLMs for a given set of metrics.
    Legend is now placed inside the plot.
    """
    # Ensure all specified metrics exist in the DataFrame, using their *renamed* names
    valid_metrics = [m for m in metrics if m in df.columns]

    if not valid_metrics:
        print(f"Warning: No valid metrics found in DataFrame for {title_prefix}. Skipping plot.")
        return

    plot_df = df[['LLM'] + valid_metrics].melt(id_vars='LLM', var_name='Metric', value_name='Score')

    plt.figure(figsize=(14, 8)) # Increased figure size
    ax = sns.barplot(x='Metric', y='Score', hue='LLM', data=plot_df, palette='Paired', errorbar=None) 

    plt.title(f'{title_prefix} Comparison', fontsize=16) 
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel('Metric', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10) 
    plt.yticks(fontsize=10)
    
    # --- Legend inside the plot ---
    # Try upper right first, adjust ncol if more LLMs
    plt.legend(title='LLM', loc='upper right', ncol=1, fontsize=10, title_fontsize=12) # Changed loc and added ncol
    # You can experiment with other locations: 'upper left', 'upper center', 'lower right', etc.
    # For many LLMs, ncol can be increased (e.g., ncol=2) to make it wider but shorter.

    if y_lim:
        plt.ylim(y_lim)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', padding=3, fontsize=9) 

    plt.grid(axis='y', linestyle='--', alpha=0.6) 
    plt.tight_layout() # Removed rect as legend is now inside
    plt.savefig(f"{filename}.png", dpi=300) 
    plt.close()

def plot_field_wise_hallucination(raw_llm_data, llm_names, filename="field_wise_hallucination_comparison"):
    """
    Plots field-wise hallucination percentages for all LLMs.
    Legend is now placed inside the plot.
    """
    all_field_hallucination_data = []
    loaded_llm_names = [] 
    for i, file_path in enumerate(JSON_FILES):
        try:
            with open(file_path, 'r') as f:
                data_dict = json.load(f)
                llm_name = llm_names[i]
                field_hallucination = data_dict.get("hallucination_metrics", {}).get("field_wise_hallucination_percentage", {})
                for field, percentage in field_hallucination.items():
                    all_field_hallucination_data.append({
                        "LLM": llm_name,
                        "Field": field,
                        "Hallucination Percentage": percentage
                    })
                loaded_llm_names.append(llm_name) 
        except FileNotFoundError:
            print(f"Skipping {file_path} for field-wise hallucination due to FileNotFoundError.")
        except json.JSONDecodeError:
            print(f"Skipping {file_path} for field-wise hallucination due to JSONDecodeError.")


    if not all_field_hallucination_data:
        print("No field-wise hallucination data to plot.")
        return

    plot_df = pd.DataFrame(all_field_hallucination_data)

    field_order = plot_df.groupby('Field')['Hallucination Percentage'].mean().sort_values(ascending=False).index

    plt.figure(figsize=(16, 9)) 
    ax = sns.barplot(x='Field', y='Hallucination Percentage', hue='LLM', data=plot_df, palette='Paired', order=field_order)
    
    plt.title('Field-wise Hallucination Percentage Comparison', fontsize=16)
    plt.ylabel('Hallucination Percentage (%)', fontsize=12)
    plt.xlabel('Field', fontsize=12)
    plt.xticks(rotation=70, ha='right', fontsize=10) 
    plt.yticks(fontsize=10)
    
    # --- Legend inside the plot ---
    plt.legend(title='LLM', loc='upper right', ncol=1, fontsize=10, title_fontsize=12) # Changed loc and added ncol
    # For more fields, 'upper center' or 'upper left' might be better.

    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.ylim(0, 100) 

    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', label_type='edge', padding=3, fontsize=8) 

    plt.tight_layout() # Removed rect
    plt.savefig(f"{filename}.png", dpi=300)
    plt.close()


# --- Main Execution ---
if __name__ == "__main__":
    raw_json_data_for_field_wise = []
    actual_llm_names_loaded = [] 

    for i, file_path in enumerate(JSON_FILES):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                raw_json_data_for_field_wise.append(data)
                actual_llm_names_loaded.append(LLM_NAMES[i])
        except FileNotFoundError:
            print(f"Skipping {file_path} as it was not found. This LLM will be excluded from plots.")
            continue 
        except json.JSONDecodeError:
            print(f"Skipping {file_path} due to JSON decoding error. This LLM will be excluded from plots.")
            continue

    if not actual_llm_names_loaded:
        print("No LLM data successfully loaded. Exiting.")
    else:
        df_comparison = load_and_flatten_data([JSON_FILES[i] for i, name in enumerate(LLM_NAMES) if name in actual_llm_names_loaded], actual_llm_names_loaded)

        if df_comparison.empty:
            print("No data loaded for comparison. Exiting.")
        else:
            print(f"Successfully loaded data for LLMs: {', '.join(actual_llm_names_loaded)}")
            
            # --- Define Metric Categories for Plotting (using original names for selection) ---
            # NOTE: When selecting metrics for plotting, you should now use the *renamed* column names
            # from the DataFrame `df_comparison`. I've updated these lists below.

            # Strict Accuracy Metrics
            strict_accuracy_metrics = [col for col in df_comparison.columns if " (Strict Acc.)" in col]
            plot_metrics_comparison(df_comparison, strict_accuracy_metrics, "Strict Accuracy", "Accuracy Score", "strict_accuracy_comparison", y_lim=[0,1])

            # Fuzzy Accuracy Metrics
            fuzzy_accuracy_metrics = [col for col in df_comparison.columns if " (Fuzzy Acc.)" in col]
            plot_metrics_comparison(df_comparison, fuzzy_accuracy_metrics, "Fuzzy Accuracy", "Accuracy Score", "fuzzy_accuracy_comparison", y_lim=[0,1])

            # Jaccard Similarity Metrics
            jaccard_metrics = [col for col in df_comparison.columns if " (Jaccard Mean)" in col]
            plot_metrics_comparison(df_comparison, jaccard_metrics, "Jaccard Similarity (Mean)", "Jaccard Score", "jaccard_similarity_comparison", y_lim=[0,1])

            # BLEU Similarity Metrics
            bleu_metrics = [col for col in df_comparison.columns if " (BLEU Mean)" in col]
            plot_metrics_comparison(df_comparison, bleu_metrics, "BLEU Similarity (Mean)", "BLEU Score", "bleu_similarity_comparison", y_lim=[0,1])

            # ROUGE-1 Similarity Metrics
            rouge1_metrics = [col for col in df_comparison.columns if " (ROUGE-1 Mean)" in col]
            plot_metrics_comparison(df_comparison, rouge1_metrics, "ROUGE-1 Similarity (Mean)", "ROUGE-1 Score", "rouge1_similarity_comparison", y_lim=[0,1])

            # ROUGE-2 Similarity Metrics
            rouge2_metrics = [col for col in df_comparison.columns if " (ROUGE-2 Mean)" in col]
            plot_metrics_comparison(df_comparison, rouge2_metrics, "ROUGE-2 Similarity (Mean)", "ROUGE-2 Score", "rouge2_similarity_comparison", y_lim=[0,1])

            # ROUGE-L Similarity Metrics
            rougel_metrics = [col for col in df_comparison.columns if " (ROUGE-L Mean)" in col]
            plot_metrics_comparison(df_comparison, rougel_metrics, "ROUGE-L Similarity (Mean)", "ROUGE-L Score", "rougel_similarity_comparison", y_lim=[0,1])

            # LLM Similarity Metrics
            llm_similarity_metrics = [col for col in df_comparison.columns if " (LLM Sim. Mean)" in col]
            plot_metrics_comparison(df_comparison, llm_similarity_metrics, "LLM Similarity (Mean)", "LLM Similarity Score", "llm_similarity_comparison", y_lim=[0,1])

            # LLM Binary Similarity Metrics
            llm_binary_similarity_metrics = [col for col in df_comparison.columns if " (LLM Binary Sim. Mean)" in col]
            plot_metrics_comparison(df_comparison, llm_binary_similarity_metrics, "LLM Binary Similarity (Mean)", "LLM Binary Similarity Score", "llm_binary_similarity_comparison", y_lim=[0,1])

            # Completeness Metrics
            completeness_metrics = ["Completeness Score"] 
            plot_metrics_comparison(df_comparison, completeness_metrics, "Completeness", "Score (%)", "completeness_comparison", y_lim=[0,100])

            # Hallucination Metrics (Overall)
            overall_hallucination_metrics = ["Overall Hallucination %"] 
            plot_metrics_comparison(df_comparison, overall_hallucination_metrics, "Overall Hallucination Percentage", "Percentage (%)", "overall_hallucination_comparison", y_lim=[0,100])

            # Field-wise Hallucination (special handling due to nested dictionary)
            plot_field_wise_hallucination(raw_json_data_for_field_wise, actual_llm_names_loaded)

            # Summary Metrics (can be grouped or plotted individually as needed)
            summary_metrics = [
                "Total Hallucinated Fields",
                "Mean Hallucinated Fields / Record",
                "Total Missing Fields",
                "Mean Missing Fields / Record",
                "Total Correct Fields",
                "Mean Correct Fields / Record",
                "Total Incorrect Fields",
                "Mean Incorrect Fields / Record"
            ]
            plot_metrics_comparison(df_comparison, summary_metrics, "Summary Metrics", "Count/Score", "summary_metrics_comparison")

            print(f"Generated comparison plots for {len(actual_llm_names_loaded)} LLMs.")
            print("Please check the current directory for the generated .png files.")