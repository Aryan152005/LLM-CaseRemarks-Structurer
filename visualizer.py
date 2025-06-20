import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import json
from pathlib import Path
from loguru import logger
import os
import re
import shutil

# Remove existing handlers to avoid duplicate output if running multiple times in a session
logger.remove()
# Add a new handler with the desired level (e.g., "INFO" or "DEBUG")
logger.add(lambda msg: print(msg))

class Visualizer:
    def __init__(self, output_dir: str = "visualizations"):
        self.base_output_dir = Path(output_dir) # Renamed to clearly indicate base directory
        self.base_output_dir.mkdir(exist_ok=True, parents=True)
        self.current_plot_dir = self.base_output_dir # This will be updated based on plot type
        plt.style.use('default')
        sns.set_palette("husl")

    def _set_plot_directory(self, subdirectory: str):
        """Sets the current directory where plots will be saved."""
        self.current_plot_dir = self.base_output_dir / subdirectory
        self.current_plot_dir.mkdir(exist_ok=True, parents=True)

    def _save_plot(self, filename: str):
        """Save the current plot to the configured output directory."""
        plt.tight_layout()
        save_path = self.current_plot_dir / filename # Use current_plot_dir
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.debug(f"Plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save plot {filename}: {e}")
        finally:
            plt.close()

    def plot_categorical_metrics(self, metrics: Dict[str, Any]):
        """Plot categorical metrics (accuracy) using bar and pie charts."""
        self._set_plot_directory("categorical_metrics") # Set subdirectory for these plots

        # Adjusted to handle the new flat structure
        # We need to explicitly find accuracy metrics by name pattern
        accuracy_metrics = {}
        for key, value in metrics.items():
            if 'accuracy' in key:
                # Exclude specific comparison metrics if they are only for _plot_event_accuracy_comparison
                if 'event_type_' in key or 'event_sub_type_' in key or 'severity_' in key or 'victim_gender_' in key or 'state_of_victim_' in key:
                    accuracy_metrics[key] = value

        if not accuracy_metrics:
            logger.warning("No categorical accuracy metrics to plot (excluding comparison metrics for this specific function).")
            return
        
        # Prepare data for plotting (e.g., 'event_type_accuracy' -> 'Event Type Accuracy')
        fields = [key.replace('_', ' ').title() for key in accuracy_metrics.keys()]
        accuracies = list(accuracy_metrics.values())

        if not fields or not accuracies:
            logger.warning("No fields or accuracies found for categorical metrics to plot.")
            return
        
        logger.info(f"Plotting general categorical metrics: {fields}")

        # Bar Chart for Accuracies
        plt.figure(figsize=(12, 6))
        bars = plt.bar(fields, accuracies, color=sns.color_palette("husl", len(fields)))
        plt.title('Overall Categorical Accuracy by Field', fontsize=14, pad=20)
        plt.xlabel('Field', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.2f}',
                             ha='center', va='bottom')
        self._save_plot('overall_categorical_metrics_bar.png')

        # Pie Chart for Proportional Accuracy
        if accuracies and sum(accuracies) > 0:
            plt.figure(figsize=(9, 9))
            plt.pie(accuracies, labels=fields, autopct='%1.1f%%', startangle=90,
                             colors=sns.color_palette("husl", len(fields)))
            plt.title('Proportion of Overall Categorical Accuracy Across Fields', fontsize=14, pad=20)
            self._save_plot('overall_categorical_metrics_pie.png')
        else:
            logger.warning("Sum of accuracies is zero or no accuracies, skipping overall categorical pie chart.")

    def plot_text_similarity_metrics(self, metrics: Dict[str, Any]):
        """Plot text similarity metrics using bar and line charts."""
        self._set_plot_directory("text_similarity_metrics") # Set subdirectory

        # Adjusted to handle the new flat structure
        # Look for keys ending with '_jaccard_mean', '_bleu_mean', '_rouge_1_mean', etc.
        # This function should probably focus on mean_cosine_similarity or a generic 'mean similarity' if that's what's available
        # Given the new `aggregate_metrics.json`, it looks like everything is by field_metric_mean.
        # This function should probably average across all fields for a given metric (e.g., average jaccard across all fields).
        # Or, it should display per-field average for a specific type of similarity (e.g., mean Jaccard for each text field).
        
        # Let's try to aggregate all 'mean' values that are not accuracies or processing times
        # and not explicitly handled by rouge/jaccard/llm specific plots
        
        general_similarity_means = {}
        for key, value in metrics.items():
            if '_mean' in key and 'accuracy' not in key and 'processing_time' not in key:
                # Exclude specific rouge/jaccard/llm for the "overall" view this function implies
                # This makes it a bit tricky, let's refine its purpose for the new data.
                # Perhaps this function should iterate over specific "metric types" and plot their means across fields.

                # Let's focus this method on a broader "text similarity mean" if that's the intention.
                # Given the specific new structure, it seems more direct to plot specific metric types per field.

                # Re-thinking: The previous sample had 'mean_cosine_similarity'. The new one has 'field_jaccard_mean'.
                # This function might be better repurposed to show MEAN JACCARD/BLEU/ROUGE/LLM across ALL applicable fields.
                # Or, if that's too complex given the flat structure, we can skip it,
                # as _create_detailed_plots_from_df's bar chart and individual plots
                # might cover this adequately.

                # For now, let's keep it but adapt its data collection.
                # Collect average similarity scores for various text fields.
                # This will extract 'specified_matter_jaccard_mean' -> 'specified_matter' for 'jaccard'
                
                match = re.match(r'(.+)_(jaccard|bleu|rouge_1|rouge_2|rouge_l|llm_similarity)_mean$', key)
                if match:
                    field_name = match.group(1).replace('_', ' ').title()
                    metric_type = match.group(2).replace('_', ' ').title()
                    
                    if field_name not in general_similarity_means:
                        general_similarity_means[field_name] = {}
                    general_similarity_means[field_name][metric_type] = value
        
        if not general_similarity_means:
            logger.warning("No general text similarity mean metrics found to plot.")
            return

        logger.info("Plotting general text similarity metrics.")

        # Let's plot average Jaccard, Bleu, Rouge_1, Rouge_2, Rouge_L, LLM Similarity across all text fields.
        # This requires pivotting the data.
        plot_data = []
        for field, metrics_dict in general_similarity_means.items():
            for metric, value in metrics_dict.items():
                plot_data.append({'Field': field, 'Metric': metric, 'Value': value})
        
        if not plot_data:
            logger.warning("No data points for general text similarity plotting.")
            return

        df_plot = pd.DataFrame(plot_data)

        # Grouped Bar Chart
        plt.figure(figsize=(15, 8))
        ax = sns.barplot(x='Field', y='Value', hue='Metric', data=df_plot, palette='viridis')
        plt.title('Average Text Similarity Metrics by Field', fontsize=14, pad=20)
        plt.xlabel('Field', fontsize=12)
        plt.ylabel('Average Score', fontsize=12)
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Metric Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        self._save_plot('avg_text_similarity_metrics_grouped_bar.png')

        # Line Plot (maybe less useful here if too many metrics/fields)
        # For simplicity, we might skip the line plot for this aggregated view if it gets too cluttered.
        # If we wanted it, we'd need to consider what "trend" it represents.
        # For now, stick to the grouped bar.

    def plot_processing_time(self, metrics: Dict[str, Any]):
        """Plot processing time metrics using bar chart and histogram."""
        self._set_plot_directory("processing_time_metrics") # Set subdirectory

        mean_processing_time = metrics.get('mean_processing_time') # Assuming it's still top-level 'mean_processing_time'
        if mean_processing_time is None:
            logger.warning("Mean processing time not found in metrics.")
            return
        
        logger.info("Plotting processing time metrics.")

        # Bar Chart for Mean Processing Time
        plt.figure(figsize=(10, 6))
        times = [mean_processing_time]
        labels = ['Mean Processing Time']
        bars = plt.bar(labels, times, color=sns.color_palette("husl", 1)[0])
        plt.title('Average Processing Time', fontsize=14, pad=20)
        plt.ylabel('Time (seconds)', fontsize=12)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.2f}s',
                             ha='center', va='bottom')
        self._save_plot('processing_time_bar.png')

        # Histogram for Individual Processing Times (if available)
        # This data comes from 'detailed_results' typically, not 'aggregate_metrics'
        # So this part should be moved to a detailed plot method.
        # For now, commenting out as it assumes a different data source than 'metrics' dict
        # individual_times = metrics.get('individual_processing_times', [])
        # if individual_times:
        #     numeric_individual_times = pd.to_numeric(individual_times, errors='coerce').dropna()
        #     if not numeric_individual_times.empty:
        #         plt.figure(figsize=(10, 6))
        #         sns.histplot(numeric_individual_times, kde=True, bins=10, color=sns.color_palette("husl", 1)[0])
        #         plt.title('Distribution of Individual Processing Times', fontsize=14, pad=20)
        #         plt.xlabel('Time (seconds)', fontsize=12)
        #         plt.ylabel('Frequency', fontsize=12)
        #         self._save_plot('processing_time_distribution_hist.png')
        #     else:
        #         logger.warning("Individual processing times are not numeric or are empty after conversion.")
        # else:
        #     logger.warning("Individual processing times not found in metrics for histogram.")


    def plot_rouge_scores(self, metrics: Dict[str, Any]):
        """Plot ROUGE scores using grouped bar and line charts."""
        self._set_plot_directory("rouge_metrics") # Set subdirectory

        rouge_data_by_field = {}
        # Iterate through metrics to find ROUGE scores like 'field_rouge_1_mean'
        for key, value in metrics.items():
            match = re.match(r'(.+)_(rouge_1|rouge_2|rouge_l)_mean$', key)
            if match:
                field_name = match.group(1).replace('_', ' ').title()
                rouge_type = match.group(2).replace('_', ' ').upper()
                if field_name not in rouge_data_by_field:
                    rouge_data_by_field[field_name] = {}
                rouge_data_by_field[field_name][rouge_type] = value
        
        if not rouge_data_by_field:
            logger.warning("No ROUGE scores to plot from aggregate metrics.")
            return
        
        fields = list(rouge_data_by_field.keys())
        rouge_types = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L'] # Consistent order

        # Prepare data for plotting
        plot_data = []
        for field in fields:
            for rouge_type in rouge_types:
                score = rouge_data_by_field[field].get(rouge_type, 0) # Default to 0 if missing
                plot_data.append({'Field': field, 'ROUGE Type': rouge_type, 'Score': score})
        
        df_plot = pd.DataFrame(plot_data)

        logger.info(f"Plotting ROUGE scores for fields: {fields}")

        # Grouped Bar Chart for ROUGE Scores
        plt.figure(figsize=(14, 7))
        ax = sns.barplot(x='Field', y='Score', hue='ROUGE Type', data=df_plot, palette='viridis')
        plt.title('ROUGE Scores by Field', fontsize=14, pad=20)
        plt.xlabel('Field', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='ROUGE Type')
        plt.ylim(0, 1)
        # Add labels to bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=8)
        self._save_plot('rouge_scores_grouped_bar.png')

        # Line Plot for ROUGE Scores
        plt.figure(figsize=(14, 7))
        sns.lineplot(x='Field', y='Score', hue='ROUGE Type', data=df_plot, marker='o', palette='viridis')
        plt.title('ROUGE Scores by Field (Trend)', fontsize=14, pad=20)
        plt.xlabel('Field', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='ROUGE Type')
        plt.ylim(0, 1)
        self._save_plot('rouge_scores_line.png')

    def plot_keyword_similarity(self, metrics: Dict[str, Any]):
        """Plot keyword similarity metrics using a bar chart."""
        self._set_plot_directory("set_similarity_metrics") # Set subdirectory

        # Dynamically find mean Jaccard for any 'jaccard_mean' that might represent sets (like keywords)
        # This will need to be flexible. For now, let's assume one main "set similarity"
        # based on the example 'specified_matter_jaccard_mean' or 'keywords_jaccard_mean'
        
        # We need to find all keys ending in '_jaccard_mean' and process them.
        jaccard_means = {}
        for key, value in metrics.items():
            if key.endswith('_jaccard_mean'):
                field_name = key.replace('_jaccard_mean', '').replace('_', ' ').title()
                jaccard_means[field_name] = value

        if not jaccard_means:
            logger.warning("No Jaccard similarity metrics (ending with '_jaccard_mean') to plot.")
            return
        
        logger.info("Plotting Jaccard similarity metrics.")

        # Create DataFrame for plotting
        df_plot = pd.DataFrame(list(jaccard_means.items()), columns=['Field', 'Mean Jaccard Similarity'])

        # Bar Chart for Mean Jaccard Similarity
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Field', y='Mean Jaccard Similarity', data=df_plot, hue='Field', palette="husl", legend=False)
        plt.title('Mean Jaccard Similarity by Field', fontsize=14, pad=20)
        plt.ylabel('Jaccard Similarity', fontsize=12)
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f')
        self._save_plot('jaccard_similarity_bar.png')


    def create_all_visualizations(self, metrics: Dict[str, Any]):
        """Create all aggregate-level visualizations from the metrics."""
        self._set_plot_directory("aggregate_metrics_summary") # Main folder for aggregate plots
        logger.info("Creating all aggregate-level visualizations.")
        try:
            self.plot_categorical_metrics(metrics)
            self.plot_text_similarity_metrics(metrics)
            self.plot_processing_time(metrics)
            self.plot_rouge_scores(metrics)
            self.plot_keyword_similarity(metrics) # This now handles all Jaccard means

            # Save metrics as JSON for reference
            with open(self.current_plot_dir / 'aggregate_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Aggregate metrics saved to {self.current_plot_dir / 'aggregate_metrics.json'}")

        except Exception as e:
            logger.error(f"Error creating aggregate visualizations: {e}")
            raise

    def create_visualizations(self, evaluation_results: Dict[str, Any], output_dir: str):
        """Create visualizations for evaluation results (detailed and aggregate)."""
        # Store original base directory, then set the new one for this run
        original_base_output_dir = self.base_output_dir
        self.base_output_dir = Path(output_dir)
        self.base_output_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Visualizations will be saved to base directory: {self.base_output_dir}")

        try:
            df = pd.DataFrame(evaluation_results.get('detailed_results', []))
            # Directly use the provided 'aggregate_metrics' dictionary
            metrics = evaluation_results.get('aggregate_metrics', {})

            if df.empty:
                logger.warning("Detailed results DataFrame is empty. Skipping detailed visualizations.")
            
            if not metrics:
                logger.warning("Aggregate metrics are empty. Skipping aggregate visualizations.")
            
            # Call aggregate-level plots
            if metrics:
                self.create_all_visualizations(metrics) # This sets its own sub-folders

            # Call detailed-level plots, which will also set their own sub-folders
            if not df.empty:
                # These methods will set their own subdirectories using _set_plot_directory
                self._plot_event_accuracy_comparison(metrics) # Pass metrics for accurate parsing
                self._plot_text_similarity_distribution(df)
                self._plot_llm_similarity_scores(df)
                self._plot_correlation_heatmap(df)
                self._plot_processing_time_trend(df)
                self._create_detailed_plots_from_df(df) # Overall bar chart from detailed results mean

            logger.info(f"All visualizations created in {self.base_output_dir}")

        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            raise
        finally:
            # Restore the original base_output_dir for subsequent calls if needed
            self.base_output_dir = original_base_output_dir
            self.current_plot_dir = self.base_output_dir # Reset current plot dir

    def _plot_event_accuracy_comparison(self, metrics: Dict[str, Any]):
        """
        Plot comparison of strict vs fuzzy accuracy for event types and sub-types.
        Now adapted to new flat aggregate_metrics structure.
        """
        self._set_plot_directory("categorical_metrics/event_accuracy_comparison") # New subdirectory

        event_type_strict = metrics.get('event_type_strict_accuracy')
        event_type_fuzzy = metrics.get('event_type_fuzzy_accuracy')
        event_sub_type_strict = metrics.get('event_sub_type_strict_accuracy')
        event_sub_type_fuzzy = metrics.get('event_sub_type_fuzzy_accuracy')
        
        # Filter out None values to ensure only existing metrics are plotted
        plot_data = []
        if event_type_strict is not None: plot_data.append({'Metric': 'Event Type', 'Accuracy Type': 'Strict', 'Accuracy': event_type_strict})
        if event_type_fuzzy is not None: plot_data.append({'Metric': 'Event Type', 'Accuracy Type': 'Fuzzy', 'Accuracy': event_type_fuzzy})
        if event_sub_type_strict is not None: plot_data.append({'Metric': 'Event Sub-type', 'Accuracy Type': 'Strict', 'Accuracy': event_sub_type_strict})
        if event_sub_type_fuzzy is not None: plot_data.append({'Metric': 'Event Sub-type', 'Accuracy Type': 'Fuzzy', 'Accuracy': event_sub_type_fuzzy})

        df_plot = pd.DataFrame(plot_data)
        
        if df_plot.empty:
            logger.warning("No event accuracy metrics found to plot comparison or all values are None.")
            return

        logger.info("Plotting event accuracy comparison.")

        # Bar Chart
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Metric', y='Accuracy', hue='Accuracy Type', data=df_plot, palette="viridis")
        plt.title('Event Type and Sub-type Accuracy Comparison', fontsize=14, pad=20)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0,1)
        plt.xticks(rotation=0)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f')
        plt.tight_layout()
        self._save_plot('event_accuracy_comparison_bar.png')

        # Line chart
        plt.figure(figsize=(10, 6))
        ax = sns.lineplot(x='Accuracy Type', y='Accuracy', hue='Metric', data=df_plot, marker='o', linestyle='-', palette='viridis')
        plt.title('Event Type and Sub-type Accuracy Comparison (Line Plot)', fontsize=14, pad=20)
        plt.xlabel('Accuracy Type', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0,1)
        plt.legend(title='Metric Type')
        # Add labels to points
        for i, row in df_plot.iterrows():
            ax.annotate(f'{row["Accuracy"]:.2f}', (row['Accuracy Type'], row['Accuracy']), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        plt.tight_layout()
        self._save_plot('event_accuracy_comparison_line.png')

    def _plot_text_similarity_distribution(self, df: pd.DataFrame):
        """
        Plot distributions of text similarity metrics using KDE, Histogram, and Violin plots.
        Dynamically identifies text similarity columns.
        Ensures clear boundaries for similarity scores [0, 1].
        """
        self._set_plot_directory("detailed_distributions/text_similarity") # New subdirectory

        # Dynamically identify text similarity columns using regex patterns
        similarity_patterns = {
            'jaccard': r'_jaccard$',
            'bleu': r'_bleu$',
            'rouge_1': r'_rouge_1$',
            'rouge_2': r'_rouge_2$',
            'rouge_l': r'_rouge_l$',
            'llm_similarity': r'_llm_similarity$'
        }
        
        text_fields_and_metrics = {}

        for col in df.columns:
            for metric_name, pattern in similarity_patterns.items():
                if re.search(pattern, col):
                    field_name = col.replace(re.search(pattern, col).group(0), '')
                    if field_name not in text_fields_and_metrics:
                        text_fields_and_metrics[field_name] = []
                    text_fields_and_metrics[field_name].append(metric_name)
                    break

        if not text_fields_and_metrics:
            logger.warning("No text similarity metrics columns found in DataFrame for detailed distribution plots.")
            return
            
        logger.info(f"Dynamically identified text similarity metrics for: {list(text_fields_and_metrics.keys())}")

        for field, metrics_list in text_fields_and_metrics.items():
            field_present_metrics_data = [] # To collect data for violin plot
            
            # Create a sub-subdirectory for each field's detailed distributions
            self._set_plot_directory(f"detailed_distributions/text_similarity/{field.replace(' ', '_')}")
            logger.info(f"Plotting detailed text similarity distributions for field: {field}")

            for metric in metrics_list:
                col = f"{field}_{metric}"
                data_to_plot = pd.to_numeric(df[col], errors='coerce').dropna()

                if not data_to_plot.empty:
                    for val in data_to_plot:
                        field_present_metrics_data.append({'Metric': metric.replace('_', ' ').upper(), 'Score': val})

                    try:
                        plt.figure(figsize=(10, 5))
                        sns.kdeplot(data=data_to_plot, label=metric.replace('_', ' ').upper(), fill=True, alpha=0.6, clip=(0, 1))
                        plt.title(f'Distribution of {metric.replace("_", " ").upper()} - {field.replace("_", " ").title()} (KDE Plot)', fontsize=14)
                        plt.xlabel('Similarity Score')
                        plt.ylabel('Density')
                        plt.xlim(0, 1)
                        plt.legend()
                        plt.tight_layout()
                        self._save_plot(f"{field}_{metric}_distribution_kde.png")
                    except Exception as e:
                        logger.warning(f"KDE Plot failed for {col}: {e}")

                    try:
                        plt.figure(figsize=(10, 5))
                        plt.hist(data_to_plot, bins=np.linspace(0, 1, 21), edgecolor='black', alpha=0.7, color=sns.color_palette("husl", 1)[0])
                        plt.title(f'Distribution of {metric.replace("_", " ").upper()} - {field.replace("_", " ").title()} (Histogram)', fontsize=14)
                        plt.xlabel('Similarity Score')
                        plt.ylabel('Frequency')
                        plt.xlim(0, 1)
                        plt.tight_layout()
                        self._save_plot(f"{field}_{metric}_distribution_hist.png")
                    except Exception as e:
                        logger.warning(f"Histogram Plot failed for {col}: {e}")
                else:
                    logger.debug(f"Column '{col}' is empty or contains only non-numeric/null values — skipping individual plots.")

            # Reset plot directory to the parent for combined plots
            self._set_plot_directory(f"detailed_distributions/text_similarity/")

            if field_present_metrics_data:
                combined_df = pd.DataFrame(field_present_metrics_data)
                plt.figure(figsize=(12, 7))
                sns.violinplot(x='Metric', y='Score', data=combined_df, hue='Metric', palette="viridis", legend=False, cut=0)
                plt.title(f'Distribution of Similarity Metrics for {field.replace("_", " ").title()} (Violin Plot)', fontsize=14)
                plt.xlabel('Metric', fontsize=12)
                plt.ylabel('Similarity Score', fontsize=12)
                plt.ylim(0, 1)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                self._save_plot(f"{field}_combined_metrics_distribution_violin.png")
            else:
                logger.debug(f"No combined metric data for {field} to plot violin plot.")

            jaccard_col = f"{field}_jaccard"
            llm_sim_col = f"{field}_llm_similarity"
            
            if jaccard_col in df.columns and llm_sim_col in df.columns:
                jaccard_data = pd.to_numeric(df[jaccard_col], errors='coerce').dropna()
                llm_sim_data = pd.to_numeric(df[llm_sim_col], errors='coerce').dropna()

                aligned_df = pd.DataFrame({
                    'Jaccard': jaccard_data,
                    'LLMSimilarity': llm_sim_data
                }).dropna()

                if not aligned_df.empty:
                    self._set_plot_directory(f"detailed_distributions/text_similarity/{field.replace(' ', '_')}") # Set back to field's folder
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(x=aligned_df['Jaccard'], y=aligned_df['LLMSimilarity'], alpha=0.7, color=sns.color_palette("husl", 1)[0])
                    plt.title(f'Jaccard vs LLM Similarity for {field.replace("_", " ").title()} (Scatter Plot)', fontsize=14)
                    plt.xlabel(f'Jaccard Similarity for {field.replace("_", " ").title()}', fontsize=12)
                    plt.ylabel(f'LLM Similarity for {field.replace("_", " ").title()}', fontsize=12)
                    plt.xlim(0, 1)
                    plt.ylim(0, 1)
                    plt.tight_layout()
                    self._save_plot(f"{field}_jaccard_vs_llm_similarity_scatter.png")
                else:
                    logger.debug(f"Skipping Jaccard vs LLM Similarity scatter plot for {field}: No aligned non-empty data after cleaning.")
            else:
                logger.debug(f"Skipping Jaccard vs LLM Similarity scatter plot for {field}: Required columns not found or contain only non-numeric/null values.")


    def _plot_llm_similarity_scores(self, df: pd.DataFrame):
        """Plot LLM similarity scores for each text field using bar and pie charts."""
        self._set_plot_directory("detailed_distributions/llm_similarity") # New subdirectory

        llm_similarity_cols = [col for col in df.columns if col.endswith('_llm_similarity')]

        llm_similarity_data = []
        for col_name in llm_similarity_cols:
            cleaned_llm_data = pd.to_numeric(df[col_name], errors='coerce').dropna()
            if not cleaned_llm_data.empty:
                field_name = col_name.replace('_llm_similarity', '').replace('_', ' ').title()
                llm_similarity_data.append({'Field': field_name, 'Average Similarity Score': cleaned_llm_data.mean()})
            else:
                logger.debug(f"LLM similarity column '{col_name}' is empty or contains only non-numeric/null values — skipping for bar plot.")

        if not llm_similarity_data:
            logger.warning("No LLM similarity data available to plot.")
            return

        logger.info("Plotting LLM similarity scores.")
        mean_df = pd.DataFrame(llm_similarity_data)

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Field', y='Average Similarity Score', data=mean_df, hue='Field', palette="husl", legend=False)
        plt.title('Average LLM Similarity Scores by Field (Detailed Results)', fontsize=14, pad=20)
        plt.ylabel('Average Similarity Score', fontsize=12)
        plt.ylim(0,1)
        plt.xticks(rotation=45, ha='right')
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f')
        plt.tight_layout()
        self._save_plot('llm_similarity_scores_bar.png')

        if not mean_df.empty and mean_df['Average Similarity Score'].sum() > 0:
            plt.figure(figsize=(9, 9))
            plt.pie(mean_df['Average Similarity Score'], labels=mean_df['Field'], autopct='%1.1f%%', startangle=90,
                             colors=sns.color_palette("husl", len(mean_df)))
            plt.title('Proportion of Average LLM Similarity by Field (Detailed Results)', fontsize=14, pad=20)
            plt.tight_layout()
            self._save_plot('llm_similarity_scores_pie.png')
        else:
            logger.warning("No valid data for LLM similarity pie chart or sum of scores is zero.")


    def _plot_correlation_heatmap(self, df: pd.DataFrame):
        """Plot correlation heatmap of all dynamically identified numeric metrics."""
        self._set_plot_directory("correlations") # New subdirectory
        df_numeric_for_corr = df.copy()
        
        metric_patterns = r'(accuracy|jaccard|bleu|rouge_\d|rouge_l|llm_similarity|processing_time)$'
        potential_metric_cols = [
            col for col in df_numeric_for_corr.columns 
            if re.search(metric_patterns, col) or col == 'processing_time'
        ]
        
        df_numeric_for_corr = df_numeric_for_corr[potential_metric_cols]

        for col in df_numeric_for_corr.columns:
            df_numeric_for_corr[col] = pd.to_numeric(df_numeric_for_corr[col], errors='coerce')

        numeric_cols_for_corr = [col for col in df_numeric_for_corr.columns if not df_numeric_for_corr[col].dropna().empty]

        if not numeric_cols_for_corr:
            logger.warning("No valid numeric columns found for correlation heatmap after cleaning.")
            return

        logger.info("Plotting correlation heatmap.")
        corr_matrix = df_numeric_for_corr[numeric_cols_for_corr].corr()
        
        if corr_matrix.empty:
            logger.warning("Correlation matrix is empty after computing, likely no sufficient data points.")
            return

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f", linewidths=.5)
        plt.title('Correlation Heatmap of Evaluation Metrics', fontsize=16)
        plt.tight_layout()
        self._save_plot('correlation_heatmap.png')

    def _plot_processing_time_trend(self, df: pd.DataFrame):
        """
        Plots the processing time trend over individual records, similar to a stock graph.
        Assumes 'processing_time' column exists in the detailed results DataFrame.
        """
        self._set_plot_directory("detailed_distributions/processing_time_trend") # New subdirectory

        if 'processing_time' not in df.columns:
            logger.warning("Column 'processing_time' not found in detailed results for trend plot.")
            return

        times = pd.to_numeric(df['processing_time'], errors='coerce').dropna()
        if times.empty:
            logger.warning("No valid numeric processing time data for trend plot.")
            return
        
        logger.info("Plotting processing time trend (stock-like graph).")

        plt.figure(figsize=(14, 7))
        plt.plot(times.index, times.values, marker='o', linestyle='-', color='skyblue', label='Processing Time per Record')
        plt.title('Processing Time Trend per Record', fontsize=16, pad=20)
        plt.xlabel('Record Index', fontsize=12)
        plt.ylabel('Processing Time (seconds)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        self._save_plot('processing_time_trend.png')

        # Also add a histogram for detailed processing time distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(times, kde=True, bins=10, color=sns.color_palette("husl", 1)[0])
        plt.title('Distribution of Individual Processing Times (Detailed Results)', fontsize=14, pad=20)
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        self._save_plot('processing_time_distribution_hist.png')


    def _create_detailed_plots_from_df(self, df: pd.DataFrame) -> None:
        """
        Create various detailed visualizations from evaluation results using the DataFrame.
        Dynamically identifies relevant numerical metrics for an overall bar chart from detailed results.
        """
        self._set_plot_directory("detailed_overall_performance") # New subdirectory

        metric_keywords = ['accuracy', 'jaccard', 'bleu', 'rouge', 'similarity', 'match', 'score', 'time']
        
        numeric_metric_cols = []
        for col in df.columns:
            if any(keyword in col for keyword in metric_keywords):
                temp_series = pd.to_numeric(df[col], errors='coerce')
                if not temp_series.dropna().empty:
                    numeric_metric_cols.append(col)
        
        if not numeric_metric_cols:
            logger.warning("No valid numeric metric columns found in DataFrame for overall performance plotting from detailed results.")
            return

        logger.info("Creating overall performance metrics bar chart from detailed results (average across records).")
        valid_metrics_data = df[numeric_metric_cols].dropna(how='all')
        if not valid_metrics_data.empty:
            valid_metrics_data = valid_metrics_data.apply(pd.to_numeric, errors='coerce').dropna(how='all')
            
            if not valid_metrics_data.empty:
                plt.figure(figsize=(12, 6))
                ax = valid_metrics_data.mean().plot(kind='bar', color=sns.color_palette("husl", len(valid_metrics_data.columns)))
                plt.title('Overall Performance Metrics (Average from Detailed Results)', fontsize=14, pad=20)
                plt.ylabel('Average Score', fontsize=12)
                
                # Check if all relevant means are <= 1.05 for ylim (ignoring processing_time)
                means_for_ylim_check = valid_metrics_data.mean().drop(labels=['processing_time'], errors='ignore', axis='index')
                if not means_for_ylim_check.empty and all(val <= 1.05 for val in means_for_ylim_check):
                    plt.ylim(0,1)
                
                plt.xticks(rotation=45, ha='right')
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.2f')
                self._save_plot('detailed_results_overall_metrics_bar.png')
            else:
                logger.warning("No valid numeric overall performance metrics found to plot after type conversion and NaN removal for detailed results summary.")
        else:
            logger.warning("No valid numeric overall performance metrics found to plot (all NaN) for detailed results summary.")


# --- Test Data and Execution (for demonstration, won't be in the final module) ---
if __name__ == "__main__":
    test_output_dir = "test_visualizations_structured"
    
    # Clean up previous test run directory if it exists
    if Path(test_output_dir).exists():
        shutil.rmtree(test_output_dir)
        print(f"Cleaned up existing directory: {test_output_dir}")

    # Sample evaluation results - adapted to the flat aggregate_metrics.json structure
    sample_evaluation_results = {
        'aggregate_metrics': {
            # Categorical metrics (flat structure)
            'event_type_strict_accuracy': 0.6,
            'event_type_fuzzy_accuracy': 0.631,
            'event_sub_type_strict_accuracy': 0.4,
            'event_sub_type_fuzzy_accuracy': 0.643,
            'state_of_victim_strict_accuracy': 0.8,
            'victim_gender_strict_accuracy': 0.9,
            
            # Text similarity metrics (flat structure, with '_mean' suffix)
            'specified_matter_jaccard_mean': 0.547,
            'specified_matter_bleu_mean': 0.489,
            'specified_matter_rouge_1_mean': 0.614,
            'specified_matter_rouge_2_mean': 0.560,
            'specified_matter_rouge_l_mean': 0.595,
            'specified_matter_llm_similarity_mean': 0.920,
            
            'date_reference_jaccard_mean': 0.8,
            'date_reference_bleu_mean': 0.8,
            'date_reference_rouge_1_mean': 0.8,
            'date_reference_rouge_2_mean': 0.8,
            'date_reference_rouge_l_mean': 0.8,
            'date_reference_llm_similarity_mean': 0.8,

            'frequency_jaccard_mean': 0.9,
            'frequency_bleu_mean': 0.9,
            'frequency_rouge_1_mean': 0.9,
            'frequency_rouge_2_mean': 0.9,
            'frequency_rouge_l_mean': 0.9,
            'frequency_llm_similarity_mean': 0.9,

            # Add more text fields as needed for robustness
            'identification_jaccard_mean': 0.85,
            'identification_bleu_mean': 0.231, # Lower BLEU to test variability
            'identification_rouge_1_mean': 0.866,
            'identification_rouge_2_mean': 0.600,
            'identification_rouge_l_mean': 0.866,
            'identification_llm_similarity_mean': 0.895,

            'object_involved_jaccard_mean': 0.77,
            'object_involved_bleu_mean': 0.581,
            'object_involved_rouge_1_mean': 0.733,
            'object_involved_rouge_2_mean': 0.700,
            'object_involved_rouge_l_mean': 0.733,
            'object_involved_llm_similarity_mean': 0.870,
            
            # Processing metrics (assumed top-level)
            'mean_processing_time': 1.25,
            
            # Additional Jaccard-like metrics (for plot_keyword_similarity)
            'keywords_jaccard_mean': 0.70, # This was in old structure, adding back for demo
            'tags_jaccard_mean': 0.65 # Another example
        },
        'detailed_results': [
            {
                'id': 'rec1', 'event_type_accuracy': 1.0, 'event_sub_type_accuracy': 1.0, 'severity_accuracy': 1.0,
                'specified_matter_jaccard': 0.7, 'specified_matter_bleu': 0.65, 'specified_matter_rouge_1': 0.6, 'specified_matter_rouge_2': 0.5, 'specified_matter_rouge_l': 0.55, 'specified_matter_llm_similarity': 0.8,
                'date_reference_jaccard': 0.8, 'date_reference_bleu': 0.75, 'date_reference_rouge_1': 0.7, 'date_reference_rouge_2': 0.6, 'date_reference_rouge_l': 0.65, 'date_reference_llm_similarity': 0.85,
                'keywords_jaccard': 0.75, 'processing_time': 1.1
            },
            {
                'id': 'rec2', 'event_type_accuracy': 0.0, 'event_sub_type_accuracy': 1.0, 'severity_accuracy': 0.0,
                'specified_matter_jaccard': 0.6, 'specified_matter_bleu': 0.55, 'specified_matter_rouge_1': 0.5, 'specified_matter_rouge_2': 0.4, 'specified_matter_rouge_l': 0.45, 'specified_matter_llm_similarity': 0.7,
                'date_reference_jaccard': 0.7, 'date_reference_bleu': 0.65, 'date_reference_rouge_1': 0.6, 'date_reference_rouge_2': 0.5, 'date_reference_rouge_l': 0.55, 'date_reference_llm_similarity': 0.78,
                'keywords_jaccard': 0.65, 'processing_time': 1.3
            },
            {
                'id': 'rec3', 'event_type_accuracy': 1.0, 'event_sub_type_accuracy': 0.0, 'severity_accuracy': 1.0,
                'specified_matter_jaccard': 0.8, 'specified_matter_bleu': 0.75, 'specified_matter_rouge_1': 0.7, 'specified_matter_rouge_2': 0.6, 'specified_matter_rouge_l': 0.65, 'specified_matter_llm_similarity': 0.88,
                'date_reference_jaccard': 0.9, 'date_reference_bleu': 0.85, 'date_reference_rouge_1': 0.8, 'date_reference_rouge_2': 0.7, 'date_reference_rouge_l': 0.75, 'date_reference_llm_similarity': 0.92,
                'keywords_jaccard': 0.80, 'processing_time': 1.2
            },
            {
                'id': 'rec4', 'event_type_accuracy': 0.0, 'event_sub_type_accuracy': 0.0, 'severity_accuracy': 0.0,
                'specified_matter_jaccard': 'not specified', 'specified_matter_bleu': 0.45, 'specified_matter_rouge_1': 0.4, 'specified_matter_rouge_2': 0.3, 'specified_matter_rouge_l': 0.35, 'specified_matter_llm_similarity': 0.6,
                'date_reference_jaccard': 0.6, 'date_reference_bleu': 0.55, 'date_reference_rouge_1': 0.5, 'date_reference_rouge_2': 0.4, 'date_reference_rouge_l': 0.45, 'date_reference_llm_similarity': 0.7,
                'keywords_jaccard': 0.55, 'processing_time': 1.5
            },
            {
                'id': 'rec5', 'event_type_accuracy': 1.0, 'event_sub_type_accuracy': 1.0, 'severity_accuracy': 1.0,
                'specified_matter_jaccard': 0.9, 'specified_matter_bleu': 0.85, 'specified_matter_rouge_1': 0.8, 'specified_matter_rouge_2': 0.7, 'specified_matter_rouge_l': 0.75, 'specified_matter_llm_similarity': 0.95,
                'date_reference_jaccard': 0.95, 'date_reference_bleu': 0.9, 'date_reference_rouge_1': 0.85, 'date_reference_rouge_2': 0.75, 'date_reference_rouge_l': 0.8, 'date_reference_llm_similarity': 0.98,
                'keywords_jaccard': 0.90, 'processing_time': 1.0
            },
            {
                'id': 'rec6', 'event_type_accuracy': 1.0, 'event_sub_type_accuracy': 1.0, 'severity_accuracy': 1.0,
                'specified_matter_jaccard': 0.85, 'specified_matter_bleu': 0.8, 'specified_matter_rouge_1': 0.75, 'specified_matter_rouge_2': 0.65, 'specified_matter_rouge_l': 0.7, 'specified_matter_llm_similarity': 0.9,
                'date_reference_jaccard': 0.88, 'date_reference_bleu': 0.82, 'date_reference_rouge_1': 0.78, 'date_reference_rouge_2': 0.68, 'date_reference_rouge_l': 0.73, 'date_reference_llm_similarity': 0.93,
                'keywords_jaccard': 0.85, 'processing_time': 1.6
            },
            {
                'id': 'rec7', 'event_type_accuracy': 0.0, 'event_sub_type_accuracy': 0.0, 'severity_accuracy': 0.0,
                'specified_matter_jaccard': 0.5, 'specified_matter_bleu': 0.4, 'specified_matter_rouge_1': 0.3, 'specified_matter_rouge_2': 0.2, 'specified_matter_rouge_l': 0.25, 'specified_matter_llm_similarity': 0.5,
                'date_reference_jaccard': 0.55, 'date_reference_bleu': 0.45, 'date_reference_rouge_1': 0.4, 'date_reference_rouge_2': 0.3, 'date_reference_rouge_l': 0.35, 'date_reference_llm_similarity': 0.6,
                'keywords_jaccard': 0.45, 'processing_time': 0.9
            }
        ]
    }

    visualizer = Visualizer()

    try:
        print(f"\nAttempting to create visualizations in: {test_output_dir}")
        visualizer.create_visualizations(sample_evaluation_results, test_output_dir)
        print(f"\nVisualizations creation process completed. Checking generated files...")

        # List files in the output directory and its subdirectories
        generated_files = []
        for root, dirs, files in os.walk(test_output_dir):
            for file in files:
                generated_files.append(Path(root) / file)

        if generated_files:
            print(f"\nSuccessfully generated {len(generated_files)} files in '{test_output_dir}':")
            for f in generated_files:
                print(f" - {f.relative_to(test_output_dir)}")
            print("\nVerification successful: Plots were generated and likely structured.")
        else:
            print("\nVerification failed: No plots were generated.")
            
    except Exception as e:
        print(f"\nAn error occurred during visualization creation: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        print("Verification failed: Plots were NOT generated due to an error.")

    finally:
        # Optional: Keep the directory for manual inspection or remove it
        # shutil.rmtree(test_output_dir)
        # print(f"\nCleaned up test directory: {test_output_dir}")
        pass