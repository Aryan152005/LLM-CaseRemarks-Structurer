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
        logger.debug(f"Current plot directory set to: {self.current_plot_dir}")

    def _save_plot(self, filename: str):
        """Save the current plot to the configured output directory."""
        plt.tight_layout()
        save_path = self.current_plot_dir / filename
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.debug(f"Plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save plot {filename}: {e}")
        finally:
            plt.close()

    def _convert_numpy_types_to_python(self, obj: Any) -> Any:
        """
        Recursively converts numpy.int64 and numpy.float64 objects
        within a dictionary or list to standard Python int or float.
        """
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types_to_python(elem) for elem in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj

    def plot_categorical_metrics(self, metrics: Dict[str, Any]):
        """Plot categorical metrics (accuracy) using bar and pie charts."""
        self._set_plot_directory("categorical_metrics") # Set subdirectory for these plots

        accuracy_metrics_to_plot = {}
        
        # Iterate directly over the top-level metrics dictionary
        for key, value in metrics.items():
            if key.endswith('_strict_accuracy') or key.endswith('_fuzzy_accuracy'):
                # Construct a more readable display name
                display_name = key.replace('_strict_accuracy', ' Strict').replace('_fuzzy_accuracy', ' Fuzzy').replace('_', ' ').title()
                accuracy_metrics_to_plot[display_name] = value

        if not accuracy_metrics_to_plot:
            logger.warning("No categorical accuracy metrics to plot found in provided metrics.")
            return
        
        fields = list(accuracy_metrics_to_plot.keys())
        accuracies = list(accuracy_metrics_to_plot.values())

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
        """
        Generates a bar plot for average text similarity metrics across all comparable text fields.
        """
        self._set_plot_directory("text_similarity_metrics") # Set subdirectory

        similarity_patterns = [
            r'_jaccard_mean$', r'_bleu_mean$', r'_rouge_1_mean$', r'_rouge_2_mean$', r'_rouge_l_mean$', 
            r'_llm_similarity_mean$', r'_llm_binary_similarity_mean$'
        ]

        plot_data = []
        for key, value in metrics.items():
            for pattern in similarity_patterns:
                match = re.search(pattern, key)
                if match:
                    field_name = key.replace(match.group(0), '').replace('_', ' ').title()
                    metric_type = match.group(0).replace('_mean', '').replace('_', ' ').title()
                    plot_data.append({
                        'Field': field_name,
                        'Metric Type': metric_type,
                        'Average Score': value
                    })
                    break

        if not plot_data:
            logger.warning("No text similarity mean metrics found to plot.")
            return

        plot_df = pd.DataFrame(plot_data)
        
        logger.info("Plotting general text similarity metrics.")

        # Grouped Bar Chart
        plt.figure(figsize=(15, 8))
        ax = sns.barplot(x='Field', y='Average Score', hue='Metric Type', data=plot_df, palette='viridis')
        plt.title('Average Text Similarity Metrics by Field', fontsize=16)
        plt.xlabel('Field', fontsize=12)
        plt.ylabel('Average Score', fontsize=12)
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Metric Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        self._save_plot('avg_text_similarity_metrics_grouped_bar.png')

    def plot_llm_similarity_metrics(self, metrics: Dict[str, Any]):
        """
        Generates a bar plot and a pie chart for average LLM similarity metrics across all comparable text fields.
        This function specifically handles metrics ending with '_llm_similarity_mean' (continuous).
        """
        self._set_plot_directory("llm_similarity_metrics") # Set subdirectory for these plots

        llm_similarity_metrics_to_plot = {}
        for key, value in metrics.items():
            if key.endswith('_llm_similarity_mean') and not key.endswith('_llm_binary_similarity_mean'):
                field_name = key.replace('_llm_similarity_mean', '').replace('_', ' ').title()
                llm_similarity_metrics_to_plot[field_name] = value

        if not llm_similarity_metrics_to_plot:
            logger.warning("No continuous LLM similarity mean metrics found to plot.")
            return

        fields = list(llm_similarity_metrics_to_plot.keys())
        scores = list(llm_similarity_metrics_to_plot.values())

        if not fields or not scores:
            logger.warning("No fields or scores found for continuous LLM similarity metrics to plot.")
            return

        logger.info(f"Plotting aggregate continuous LLM similarity metrics for fields: {fields}")

        # Bar Chart for Average LLM Similarity
        plt.figure(figsize=(10, 6))
        bars = plt.bar(fields, scores, color=sns.color_palette("viridis", len(fields)))
        plt.title('Average LLM Similarity Scores by Field (Aggregate)', fontsize=14, pad=20)
        plt.xlabel('Field', fontsize=12)
        plt.ylabel('Average Score', fontsize=12)
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.2f}',
                             ha='center', va='bottom')
        self._save_plot('avg_llm_similarity_metrics_bar.png')

        # Pie Chart for Proportion of LLM Similarity
        if scores and sum(scores) > 0:
            plt.figure(figsize=(9, 9))
            plt.pie(scores, labels=fields, autopct='%1.1f%%', startangle=90,
                             colors=sns.color_palette("viridis", len(fields)))
            plt.title('Proportion of Average LLM Similarity by Field (Aggregate)', fontsize=14, pad=20)
            self._save_plot('proportion_llm_similarity_pie.png')
        else:
            logger.warning("Sum of continuous LLM similarity scores is zero or no scores, skipping aggregate LLM similarity pie chart.")

    def plot_processing_time_distribution(self, df_detailed: pd.DataFrame):
        """
        Generates a distribution plot (KDE or histogram) for processing time from detailed results.
        """
        self._set_plot_directory("processing_metrics") # Set subdirectory

        if 'processing_time' not in df_detailed.columns or df_detailed['processing_time'].isnull().all():
            logger.warning("No valid 'processing_time' data found in detailed results for plotting distribution. Skipping processing time plot.")
            return

        processing_times_data = df_detailed['processing_time'].dropna()

        if processing_times_data.empty:
            logger.warning("No non-null processing_time data after dropping missing values. Skipping processing time plot.")
            return

        logger.info(f"Plotting processing time distribution for {len(processing_times_data)} samples.")

        plt.figure(figsize=(10, 6))
        
        sns.histplot(processing_times_data, kde=True, bins=20, color='skyblue') 

        plt.title('Distribution of Processing Time', fontsize=16)
        plt.xlabel('Processing Time (seconds)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        mean_time = processing_times_data.mean()
        std_time = processing_times_data.std()
        
        if not np.isnan(mean_time):
            plt.axvline(mean_time, color='red', linestyle='--', label=f'Mean: {mean_time:.2f}s')
            if not np.isnan(std_time):
                plt.axvline(mean_time + std_time, color='orange', linestyle=':', label=f'Std Dev: {std_time:.2f}s')
                plt.axvline(mean_time - std_time, color='orange', linestyle=':')
        
        plt.legend()
        self._save_plot('processing_time_distribution.png')
        
    def plot_rouge_scores(self, metrics: Dict[str, Any]):
        """
        Generates bar and line plots for ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) across all comparable text fields.
        """
        self._set_plot_directory("rouge_metrics") # Set subdirectory

        rouge_data_by_field = {}
        rouge_types_suffix = ['_rouge_1_mean', '_rouge_2_mean', '_rouge_l_mean']

        for key, value in metrics.items():
            for suffix in rouge_types_suffix:
                if key.endswith(suffix):
                    field_name = key.replace(suffix, '').replace('_', ' ').title()
                    rouge_type = suffix.replace('_mean', '').lstrip('_').replace('_', '-').upper()
                    if field_name not in rouge_data_by_field:
                        rouge_data_by_field[field_name] = {}
                    rouge_data_by_field[field_name][rouge_type] = value
                    break

        if not rouge_data_by_field:
            logger.warning("No ROUGE scores to plot from aggregate metrics.")
            return
        
        fields = list(rouge_data_by_field.keys())
        rouge_types = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']

        plot_data = []
        for field in fields:
            for rouge_type in rouge_types:
                score = rouge_data_by_field[field].get(rouge_type, 0)
                plot_data.append({'Field': field, 'ROUGE Type': rouge_type, 'Score': score})
        
        df_plot = pd.DataFrame(plot_data)

        logger.info(f"Plotting ROUGE scores for fields: {fields}")

        # Grouped Bar Chart
        plt.figure(figsize=(14, 7))
        ax = sns.barplot(x='Field', y='Score', hue='ROUGE Type', data=df_plot, palette='viridis')
        plt.title('ROUGE Scores by Field', fontsize=14, pad=20)
        plt.xlabel('Field', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='ROUGE Type')
        plt.ylim(0, 1)
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
        self._save_plot('jaccard_similarity_bar.png');
    def plot_completeness_metrics(self, metrics: Dict[str, Any], df_detailed: pd.DataFrame):
        """
        Plots completeness metrics: mean completeness from aggregate and a categorized distribution.
        """
        self._set_plot_directory("completeness_metrics")

        # Plot mean completeness score (from aggregate metrics)
        completeness_aggregate = metrics.get('completeness_metrics', {})
        mean_completeness_score = completeness_aggregate.get('mean_completeness_score')

        if mean_completeness_score is not None:
            logger.info(f"Plotting mean completeness score: {mean_completeness_score:.2f}")
            plt.figure(figsize=(8, 5))
            bars = plt.bar(['Mean Completeness Score'], [mean_completeness_score], color='lightcoral')
            plt.title('Average Completeness Score', fontsize=14, pad=20)
            plt.ylabel('Score (0-100)', fontsize=12)
            plt.ylim(0, 100)
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                                 f'{height:.2f}',
                                 ha='center', va='bottom')
            self._save_plot('mean_completeness_score_bar.png')
        else:
            logger.warning("Mean completeness score not found in aggregate metrics. Skipping mean completeness plot.")

        # Plot categorized distribution of completeness scores (from detailed results)
        if 'completeness_score' in df_detailed.columns and not df_detailed['completeness_score'].isnull().all():
            completeness_data = pd.to_numeric(df_detailed['completeness_score'], errors='coerce').dropna() # Ensure numeric
            if not completeness_data.empty:
                logger.info(f"Plotting categorized completeness score distribution for {len(completeness_data)} samples.")
                
                bins = [0, 50, 75, 100.01]
                labels = ['Low (0-50)', 'Medium (50-75)', 'High (75-100)']
                
                completeness_categories = pd.cut(completeness_data, bins=bins, labels=labels, right=False)
                
                category_counts = completeness_categories.value_counts(sort=False)
                category_percentages = (category_counts / len(completeness_data)) * 100
                
                if not category_percentages.empty:
                    plt.figure(figsize=(10, 6))
                    bars = plt.bar(category_percentages.index, category_percentages.values, color=sns.color_palette("pastel"))
                    plt.title('Percentage of Records by Completeness Score Range', fontsize=16)
                    plt.xlabel('Completeness Score Range', fontsize=12)
                    plt.ylabel('Percentage of Records (%)', fontsize=12)
                    plt.ylim(0, 100)
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height,
                                 f'{height:.1f}%',
                                 ha='center', va='bottom')
                    
                    self._save_plot('completeness_score_categorized_distribution.png')
                else:
                    logger.warning("No completeness data after categorization. Skipping categorized distribution plot.")
            else:
                logger.warning("No non-null completeness_score data in detailed results for distribution plot.")
        else:
            logger.warning("'completeness_score' column not found or is all null in detailed results. Skipping completeness distribution plot.")

    def plot_hallucination_metrics(self, metrics: Dict[str, Any]):
        """
        Plots overall and field-wise hallucination percentages.
        """
        self._set_plot_directory("hallucination_metrics")

        hallucination_aggregate = metrics.get('hallucination_metrics', {})
        overall_hallucination_percentage = hallucination_aggregate.get('overall_hallucination_percentage')
        field_wise_hallucination_percentage = hallucination_aggregate.get('field_wise_hallucination_percentage', {})

        if overall_hallucination_percentage is not None:
            logger.info(f"Plotting overall hallucination percentage: {overall_hallucination_percentage:.2f}%")
            plt.figure(figsize=(8, 5))
            bars = plt.bar(['Overall Hallucination Percentage'], [overall_hallucination_percentage], color='indianred')
            plt.title('Overall Hallucination Percentage', fontsize=14, pad=20)
            plt.ylabel('Percentage (%)', fontsize=12)
            plt.ylim(0, 100)
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                                 f'{height:.2f}%',
                                 ha='center', va='bottom')
            self._save_plot('overall_hallucination_percentage_bar.png')
        else:
            logger.warning("Overall hallucination percentage not found in aggregate metrics. Skipping overall hallucination plot.")

        if field_wise_hallucination_percentage:
            logger.info("Plotting field-wise hallucination percentages.")
            df_field_wise = pd.DataFrame(list(field_wise_hallucination_percentage.items()), 
                                         columns=['Field', 'Hallucination Percentage'])
            df_field_wise = df_field_wise.sort_values(by='Hallucination Percentage', ascending=False)

            plt.figure(figsize=(12, 7))
            ax = sns.barplot(x='Hallucination Percentage', y='Field', data=df_field_wise, palette='rocket')
            plt.title('Field-Wise Hallucination Percentage', fontsize=14, pad=20)
            plt.xlabel('Hallucination Percentage (%)', fontsize=12)
            plt.ylabel('Field', fontsize=12)
            plt.xlim(0, 100)
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f%%')
            plt.tight_layout()
            self._save_plot('field_wise_hallucination_percentage_bar.png')
        else:
            logger.warning("No field-wise hallucination percentages to plot.")

    def plot_field_status_summary(self, metrics: Dict[str, Any]):
        """
        Plots a summary of correct, missing from LLM, incorrect, and hallucinated fields.
        """
        self._set_plot_directory("field_status_summary")

        correct_count = metrics.get('correct_fields_metrics', {}).get('total_correct_fields')
        missing_count = metrics.get('missing_from_llm_metrics', {}).get('total_missing_from_llm_fields')
        incorrect_count = metrics.get('incorrect_fields_metrics', {}).get('total_incorrect_fields')
        hallucinated_count = metrics.get('hallucination_metrics', {}).get('total_hallucinated_fields')

        plot_data = []
        if correct_count is not None: plot_data.append({'Category': 'Correct', 'Count': correct_count, 'Color': '#28a745'})
        if incorrect_count is not None: plot_data.append({'Category': 'Incorrect', 'Count': incorrect_count, 'Color': '#fd7e14'})
        if missing_count is not None: plot_data.append({'Category': 'Missing from LLM', 'Count': missing_count, 'Color': '#ffc107'})
        if hallucinated_count is not None: plot_data.append({'Category': 'Hallucinated by LLM', 'Count': hallucinated_count, 'Color': '#dc3545'})

        if not plot_data:
            logger.warning("No field status summary data to plot.")
            return

        df_plot = pd.DataFrame(plot_data)
        
        logger.info("Plotting overall field status summary.")

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Category', y='Count', data=df_plot, palette=df_plot['Color'].tolist())
        plt.title('Overall Field Status Summary (Total Counts)', fontsize=16, pad=20)
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Total Count of Fields', fontsize=12)
        plt.xticks(rotation=30, ha='right')
        for container in ax.containers:
            ax.bar_label(container, fmt='%d')
        plt.tight_layout()
        self._save_plot('overall_field_status_summary_bar.png')

        # Pie chart for proportions
        total_sum = df_plot['Count'].sum()
        if total_sum > 0:
            plt.figure(figsize=(9, 9))
            plt.pie(df_plot['Count'], labels=df_plot['Category'], autopct='%1.1f%%', startangle=90,
                             colors=df_plot['Color'].tolist())
            plt.title('Proportion of Overall Field Status', fontsize=14, pad=20)
            plt.tight_layout()
            self._save_plot('overall_field_status_summary_pie.png')
        else:
            logger.warning("Total count of fields is zero, skipping overall field status pie chart.")


    def create_all_visualizations(self, metrics: Dict[str, Any]):
        """Create all aggregate-level visualizations from the metrics."""
        self._set_plot_directory("aggregate_metrics_summary") # Main folder for aggregate plots
        logger.info("Creating all aggregate-level visualizations.")
        try:
            self.plot_categorical_metrics(metrics)
            self.plot_text_similarity_metrics(metrics)
            self.plot_llm_similarity_metrics(metrics)
            self.plot_rouge_scores(metrics)
            self.plot_keyword_similarity(metrics)
            # For completeness and hallucination, pass the relevant sub-dictionaries from aggregate_metrics
            self.plot_completeness_metrics(metrics=metrics, df_detailed=pd.DataFrame()) # df_detailed is empty here, will be populated later
            self.plot_hallucination_metrics(metrics)
            self.plot_field_status_summary(metrics) 

            # Convert numpy types to standard Python types before JSON serialization
            serializable_metrics = self._convert_numpy_types_to_python(metrics)

            with open(self.current_plot_dir / 'aggregate_metrics.json', 'w') as f:
                json.dump(serializable_metrics, f, indent=2) # Use the converted metrics
            logger.info(f"Aggregate metrics saved to {self.current_plot_dir / 'aggregate_metrics.json'}")

        except Exception as e:
            logger.error(f"Error creating aggregate visualizations: {e}")
            raise

    def create_visualizations(self, evaluation_results: Dict[str, Any], output_dir: str):
        """Create visualizations for evaluation results (detailed and aggregate)."""
        original_base_output_dir = self.base_output_dir
        self.base_output_dir = Path(output_dir)
        self.base_output_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Visualizations will be saved to base directory: {self.base_output_dir}")

        try:
            df = pd.DataFrame(evaluation_results.get('detailed_results', []))
            metrics = evaluation_results.get('aggregate_metrics', {})

            if 'processing_metrics' in df.columns:
                # Ensure processing_time is directly accessible as a column
                if not df['processing_metrics'].empty and isinstance(df['processing_metrics'].iloc[0], dict):
                    df['processing_time'] = df['processing_metrics'].apply(lambda x: x.get('processing_time') if isinstance(x, dict) else np.nan)
                    logger.info("Flattened 'processing_metrics' to 'processing_time' column for plotting.")
                else:
                    logger.warning("'processing_metrics' column found but does not contain dictionaries. Cannot extract 'processing_time'.")
            else:
                logger.warning("'processing_metrics' column not found in DataFrame for flattening. Processing time plots might be affected.")

            if df.empty:
                logger.warning("Detailed results DataFrame is empty. Skipping detailed visualizations.")
            
            if not metrics:
                logger.warning("Aggregate metrics are empty. Skipping aggregate visualizations.")
            
            if metrics:
                self.create_all_visualizations(metrics)

            if not df.empty:
                self.plot_processing_time_distribution(df)
                self.plot_completeness_metrics(metrics=metrics, df_detailed=df) # Pass full metrics and detailed df
                self._plot_llm_similarity_scores(df)
                self._plot_text_similarity_distribution(df) # Uncommented
                self._plot_correlation_heatmap(df) # Uncommented
                self._plot_processing_time_trend(df) # Uncommented
                self._create_detailed_plots_from_df(df) # Uncommented

        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            raise
        finally:
            self.base_output_dir = original_base_output_dir
            logger.info("Finished visualization creation process.")

    def _plot_event_accuracy_comparison(self, metrics: Dict[str, Any]):
        """
        Plot comparison of strict vs fuzzy accuracy for event types and sub-types.
        """
        self._set_plot_directory("categorical_metrics/event_accuracy_comparison")

        # Access metrics directly from the top-level aggregate_metrics
        event_type_strict = metrics.get('event_type_strict_accuracy')
        event_type_fuzzy = metrics.get('event_type_fuzzy_accuracy')
        event_sub_type_strict = metrics.get('event_sub_type_strict_accuracy')
        event_sub_type_fuzzy = metrics.get('event_sub_type_fuzzy_accuracy')
        
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

        plt.figure(figsize=(10, 6))
        ax = sns.lineplot(x='Accuracy Type', y='Accuracy', hue='Metric', data=df_plot, marker='o', linestyle='-', palette='viridis')
        plt.title('Event Type and Sub-type Accuracy Comparison (Line Plot)', fontsize=14, pad=20)
        plt.xlabel('Accuracy Type', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0,1)
        plt.legend(title='Metric Type')
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
        self._set_plot_directory("detailed_distributions/text_similarity")

        similarity_patterns = {
            'jaccard': r'_jaccard$',
            'bleu': r'_bleu$',
            'rouge_1': r'_rouge_1$',
            'rouge_2': r'_rouge_2$',
            'rouge_l': r'_rouge_l$',
            'llm_similarity': r'_llm_similarity$',
            'llm_binary_similarity': r'_llm_binary_similarity$'
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
            field_present_metrics_data = []
            
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
                        sns.kdeplot(data=data_to_plot, fill=True, alpha=0.6, clip=(0, 1), color='purple')
                        plt.title(f'Distribution of {metric.replace("_", " ").upper()} - {field.replace("_", " ").title()} (KDE Plot)', fontsize=14)
                        plt.xlabel('Similarity Score')
                        plt.ylabel('Density')
                        plt.xlim(0, 1)
                        # plt.legend() # Removed as it's a single KDE plot
                        plt.tight_layout()
                        self._save_plot(f"{field}_{metric}_distribution_kde.png")
                    except Exception as e:
                        logger.warning(f"KDE Plot failed for {col}: {e}")

                    try:
                        plt.figure(figsize=(10, 5))
                        plt.hist(data_to_plot, bins=np.linspace(0, 1, 21), edgecolor='black', alpha=0.7, color='purple')
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

            self._set_plot_directory(f"detailed_distributions/text_similarity/") # Reset to parent directory for combined plots

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
                    self._set_plot_directory(f"detailed_distributions/text_similarity/{field.replace(' ', '_')}") # Set subdirectory for field-specific scatter
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
        self._set_plot_directory("detailed_distributions/llm_similarity")

        llm_similarity_cols = [col for col in df.columns if col.endswith('_llm_similarity') and not col.endswith('_llm_binary_similarity')]

        llm_similarity_data = []
        for col_name in llm_similarity_cols:
            cleaned_llm_data = pd.to_numeric(df[col_name], errors='coerce').dropna()
            if not cleaned_llm_data.empty:
                field_name = col_name.replace('_llm_similarity', '').replace('_', ' ').title()
                llm_similarity_data.append({'Field': field_name, 'Average Similarity Score': cleaned_llm_data.mean()})
                
                try:
                    self._set_plot_directory(f"detailed_distributions/llm_similarity/{field_name.replace(' ', '_')}")
                    
                    plt.figure(figsize=(10, 5))
                    sns.kdeplot(data=cleaned_llm_data, fill=True, alpha=0.6, clip=(0, 1), color='purple')
                    plt.title(f'Distribution of LLM Similarity - {field_name} (KDE Plot)', fontsize=14)
                    plt.xlabel('Similarity Score')
                    plt.ylabel('Density')
                    plt.xlim(0, 1)
                    plt.tight_layout()
                    self._save_plot(f"{field_name}_llm_similarity_distribution_kde.png")
                    
                    plt.figure(figsize=(10, 5))
                    plt.hist(cleaned_llm_data, bins=np.linspace(0, 1, 21), edgecolor='black', alpha=0.7, color='purple')
                    plt.title(f'Distribution of LLM Similarity - {field_name} (Histogram)', fontsize=14)
                    plt.xlabel('Similarity Score')
                    plt.ylabel('Frequency')
                    plt.xlim(0, 1)
                    plt.tight_layout()
                    self._save_plot(f"{field_name}_llm_similarity_distribution_hist.png")

                except Exception as e:
                    logger.warning(f"Individual LLM Similarity plot failed for {col_name}: {e}")
            else:
                logger.debug(f"LLM similarity column '{col_name}' is empty or contains only non-numeric/null values — skipping for detailed plots.")

        self._set_plot_directory("detailed_distributions/llm_similarity") # Reset to parent directory for combined plots

        if not llm_similarity_data:
            logger.warning("No LLM similarity data available to plot overall detailed summary.")
            return

        logger.info("Plotting overall detailed LLM similarity scores (bar and pie charts).")
        mean_df = pd.DataFrame(llm_similarity_data)

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Field', y='Average Similarity Score', data=mean_df, hue='Field', palette="viridis", legend=False)
        plt.title('Average LLM Similarity Scores by Field (Detailed Results Summary)', fontsize=14, pad=20)
        plt.ylabel('Average Similarity Score', fontsize=12)
        plt.ylim(0,1)
        plt.xticks(rotation=45, ha='right')
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f')
        plt.tight_layout()
        self._save_plot('llm_similarity_scores_detailed_bar.png')

        if not mean_df.empty and mean_df['Average Similarity Score'].sum() > 0:
            plt.figure(figsize=(9, 9))
            plt.pie(mean_df['Average Similarity Score'], labels=mean_df['Field'], autopct='%1.1f%%', startangle=90,
                             colors=sns.color_palette("viridis", len(mean_df)))
            plt.title('Proportion of Average LLM Similarity by Field (Detailed Results Summary)', fontsize=14, pad=20)
            plt.tight_layout()
            self._save_plot('llm_similarity_scores_detailed_pie.png')
        else:
            logger.warning("No valid data for detailed LLM similarity pie chart or sum of scores is zero.")

    def _plot_correlation_heatmap(self, df: pd.DataFrame):
        """Plot correlation heatmap of all dynamically identified numeric metrics."""
        self._set_plot_directory("correlations")
        df_numeric_for_corr = df.copy()
        
        metric_patterns = r'(accuracy|jaccard|bleu|rouge_\d|rouge_l|llm_similarity|llm_binary_similarity|processing_time|completeness_score)$' # Added completeness_score
        potential_metric_cols = [
            col for col in df_numeric_for_corr.columns 
            if re.search(metric_patterns, col) or col == 'processing_time' or col == 'completeness_score' # Ensure completeness_score is included
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
        self._set_plot_directory("detailed_distributions/processing_time_trend")

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
        self._set_plot_directory("detailed_overall_performance")

        metric_keywords = ['accuracy', 'jaccard', 'bleu', 'rouge', 'similarity', 'match', 'score', 'time', 'binary', 'completeness'] # Added 'completeness'
        
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
    
    if Path(test_output_dir).exists():
        shutil.rmtree(test_output_dir)
        print(f"Cleaned up existing directory: {test_output_dir}")
