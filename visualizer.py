import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import json
from pathlib import Path
from loguru import logger
import os
import re # Import regex for dynamic column identification
import shutil # For cleaning up test directory

# Remove existing handlers to avoid duplicate output if running multiple times in a session
logger.remove()
# Add a new handler with the desired level (e.g., "INFO" or "DEBUG")
logger.add(lambda msg: print(msg)) # Changed level to INFO

class Visualizer:
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        # Ensure the base output directory exists. Use parents=True for nested paths.
        self.output_dir.mkdir(exist_ok=True, parents=True)
        plt.style.use('default')
        sns.set_palette("husl")

    def _save_plot(self, filename: str):
        """Save the current plot to the configured output directory."""
        plt.tight_layout()
        save_path = self.output_dir / filename
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.debug(f"Plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save plot {filename}: {e}")
        finally:
            plt.close()

    def plot_categorical_metrics(self, metrics: Dict[str, Any]):
        """Plot categorical metrics (accuracy) using bar and pie charts."""
        categorical_data = metrics.get('categorical_metrics', {})
        if not categorical_data:
            logger.warning("No categorical metrics to plot.")
            return

        fields = list(categorical_data.keys())
        accuracies = [data['accuracy'] for data in categorical_data.values() if 'accuracy' in data]

        if not fields or not accuracies:
            logger.warning("No fields or accuracies found for categorical metrics to plot.")
            return
        
        logger.info(f"Plotting categorical metrics for fields: {fields}")

        # Bar Chart for Accuracies
        plt.figure(figsize=(12, 6))
        bars = plt.bar(fields, accuracies, color=sns.color_palette("husl", len(fields)))
        plt.title('Accuracy by Field', fontsize=14, pad=20)
        plt.xlabel('Field', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, 1) # Accuracies are bounded by 0 and 1
        plt.xticks(rotation=45, ha='right')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.2f}',
                             ha='center', va='bottom')
        self._save_plot('categorical_metrics_bar.png')

        # Pie Chart for Proportional Accuracy
        if accuracies and sum(accuracies) > 0:
            plt.figure(figsize=(9, 9))
            plt.pie(accuracies, labels=fields, autopct='%1.1f%%', startangle=90,
                             colors=sns.color_palette("husl", len(fields)))
            plt.title('Proportion of Accuracy Across Fields', fontsize=14, pad=20)
            self._save_plot('categorical_metrics_pie.png')
        else:
            logger.warning("Sum of accuracies is zero or no accuracies, skipping pie chart.")

    def plot_text_similarity_metrics(self, metrics: Dict[str, Any]):
        """Plot text similarity metrics using bar and line charts."""
        text_data = metrics.get('text_similarity_metrics', {})
        if not text_data:
            logger.warning("No text similarity metrics to plot.")
            return

        fields = []
        similarities = []
        stds = []

        for field, data in text_data.items():
            mean_sim = data.get('mean_cosine_similarity')
            std_sim = data.get('std_cosine_similarity')
            if mean_sim is not None and std_sim is not None:
                fields.append(field)
                similarities.append(mean_sim)
                stds.append(std_sim)
            else:
                logger.warning(f"Skipping text similarity plot for {field}: Missing mean or std cosine similarity.")

        if not fields:
            logger.warning("No valid fields found for text similarity metrics to plot.")
            return
        
        logger.info(f"Plotting text similarity metrics for fields: {fields}")


        # Bar Chart with Error Bars for Mean Cosine Similarity
        plt.figure(figsize=(12, 6))
        bars = plt.bar(fields, similarities, yerr=stds, capsize=5, color=sns.color_palette("husl", len(fields)))
        plt.title('Mean Cosine Similarity by Field', fontsize=14, pad=20)
        plt.xlabel('Field', fontsize=12)
        plt.ylabel('Cosine Similarity', fontsize=12)
        plt.ylim(0, 1) # Cosine similarity is bounded by 0 and 1
        plt.xticks(rotation=45, ha='right')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.2f}',
                             ha='center', va='bottom')
        self._save_plot('text_similarity_metrics_bar.png')

        # Line Plot for Mean Cosine Similarity
        plt.figure(figsize=(12, 6))
        plt.plot(fields, similarities, marker='o', linestyle='-', color=sns.color_palette("husl", 1)[0])
        plt.title('Mean Cosine Similarity by Field (Trend)', fontsize=14, pad=20)
        plt.xlabel('Field', fontsize=12)
        plt.ylabel('Cosine Similarity', fontsize=12)
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        for i, txt in enumerate(similarities):
            plt.annotate(f'{txt:.2f}', (fields[i], similarities[i]), textcoords="offset points", xytext=(0,10), ha='center')
        self._save_plot('text_similarity_metrics_line.png')

    def plot_processing_time(self, metrics: Dict[str, Any]):
        """Plot processing time metrics using bar chart and histogram."""
        processing_data = metrics.get('processing_metrics', {})
        if not processing_data:
            logger.warning("No processing time metrics to plot.")
            return

        mean_processing_time = processing_data.get('mean_processing_time')
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
        individual_times = processing_data.get('individual_processing_times', [])
        if individual_times:
            # Ensure individual_times are numeric before plotting histogram
            numeric_individual_times = pd.to_numeric(individual_times, errors='coerce').dropna()
            if not numeric_individual_times.empty:
                plt.figure(figsize=(10, 6))
                sns.histplot(numeric_individual_times, kde=True, bins=10, color=sns.color_palette("husl", 1)[0])
                plt.title('Distribution of Individual Processing Times', fontsize=14, pad=20)
                plt.xlabel('Time (seconds)', fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                self._save_plot('processing_time_distribution_hist.png')
            else:
                logger.warning("Individual processing times are not numeric or are empty after conversion.")
        else:
            logger.warning("Individual processing times not found in metrics for histogram.")

    def plot_rouge_scores(self, metrics: Dict[str, Any]):
        """Plot ROUGE scores using grouped bar and line charts."""
        text_data = metrics.get('text_similarity_metrics', {})
        if not text_data:
            logger.warning("No ROUGE scores to plot from text similarity metrics.")
            return

        rouge_scores = {
            'ROUGE-1': [],
            'ROUGE-2': [],
            'ROUGE-L': []
        }

        fields_with_rouge_data = []
        for field, field_data in text_data.items():
            rouge_data = field_data.get('rouge_scores', {})
            r1 = rouge_data.get('rouge1')
            r2 = rouge_data.get('rouge2')
            rL = rouge_data.get('rougeL')

            if r1 is not None or r2 is not None or rL is not None:
                fields_with_rouge_data.append(field)
                rouge_scores['ROUGE-1'].append(r1 if r1 is not None else 0)
                rouge_scores['ROUGE-2'].append(r2 if r2 is not None else 0)
                rouge_scores['ROUGE-L'].append(rL if rL is not None else 0)
            else:
                logger.debug(f"No ROUGE scores found for field: {field}. Skipping for ROUGE plots.")
        
        fields = fields_with_rouge_data # Update fields to only include those with data
        if not fields:
            logger.warning("No fields with ROUGE data found to plot.")
            return
        
        logger.info(f"Plotting ROUGE scores for fields: {fields}")


        x = np.arange(len(fields))
        width = 0.25

        # Grouped Bar Chart for ROUGE Scores
        plt.figure(figsize=(14, 7))
        colors = sns.color_palette("husl", len(rouge_scores))
        for i, (metric, scores) in enumerate(rouge_scores.items()):
            bars = plt.bar(x + i*width, scores, width, label=metric, color=colors[i])
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                                     f'{height:.2f}',
                                     ha='center', va='bottom', fontsize=8)
        plt.title('ROUGE Scores by Field', fontsize=14, pad=20)
        plt.xlabel('Field', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.xticks(x + width, fields, rotation=45, ha='right')
        plt.legend(fontsize=10)
        plt.ylim(0, 1) # ROUGE scores are bounded by 0 and 1
        self._save_plot('rouge_scores_grouped_bar.png')

        # Line Plot for ROUGE Scores
        plt.figure(figsize=(14, 7))
        for metric, scores in rouge_scores.items():
            plt.plot(fields, scores, marker='o', linestyle='-', label=metric)
            for i, txt in enumerate(scores):
                plt.annotate(f'{txt:.2f}', (fields[i], scores[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        plt.title('ROUGE Scores by Field (Trend)', fontsize=14, pad=20)
        plt.xlabel('Field', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(fontsize=10)
        plt.ylim(0, 1)
        self._save_plot('rouge_scores_line.png')

    def plot_keyword_similarity(self, metrics: Dict[str, Any]):
        """Plot keyword similarity metrics using a bar chart."""
        keyword_data = metrics.get('set_similarity_metrics', {}).get('keywords', {})
        if not keyword_data:
            logger.warning("No keyword similarity metrics to plot.")
            return

        similarity = keyword_data.get('mean_jaccard_similarity')
        std = keyword_data.get('std_jaccard_similarity')

        if similarity is None or std is None:
            logger.warning("Mean or standard deviation for Jaccard similarity not found.")
            return
        
        logger.info("Plotting keyword similarity metrics.")

        # Bar Chart for Mean Jaccard Similarity
        plt.figure(figsize=(8, 6))
        bars = plt.bar(['Keywords'], [similarity], yerr=[std], capsize=5, color=sns.color_palette("husl", 1)[0])
        plt.title('Keyword Similarity (Mean Jaccard Similarity)', fontsize=14, pad=20)
        plt.ylabel('Jaccard Similarity', fontsize=12)
        plt.ylim(0, 1) # Jaccard similarity is bounded by 0 and 1
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.2f}',
                             ha='center', va='bottom')
        self._save_plot('keyword_similarity_bar.png')

    def create_all_visualizations(self, metrics: Dict[str, Any]):
        """Create all aggregate-level visualizations from the metrics."""
        logger.info("Creating all aggregate-level visualizations.")
        try:
            self.plot_categorical_metrics(metrics)
            self.plot_text_similarity_metrics(metrics)
            self.plot_processing_time(metrics)
            self.plot_rouge_scores(metrics)
            self.plot_keyword_similarity(metrics)

            # Save metrics as JSON for reference
            with open(self.output_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info("Aggregate metrics saved to metrics.json")

        except Exception as e:
            logger.error(f"Error creating aggregate visualizations: {e}")
            raise

    def create_visualizations(self, evaluation_results: Dict[str, Any], output_dir: str):
        """Create visualizations for evaluation results (detailed and aggregate)."""
        original_output_dir = self.output_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Visualizations will be saved to {self.output_dir}")

        try:
            df = pd.DataFrame(evaluation_results.get('detailed_results', []))
            metrics = evaluation_results.get('aggregate_metrics', {})

            if df.empty:
                logger.warning("Detailed results DataFrame is empty. Skipping detailed visualizations.")
            
            if not metrics:
                logger.warning("Aggregate metrics are empty. Skipping aggregate visualizations.")
            
            # Call aggregate-level plots using the original methods
            if metrics:
                self.create_all_visualizations(metrics) # This calls all your plot_* methods

            # Call detailed-level plots, ensuring they use the correct self.output_dir
            if not df.empty:
                self._plot_event_accuracy_comparison(metrics) # Pass metrics as expected by your function
                self._plot_text_similarity_distribution(df)
                self._plot_llm_similarity_scores(df)
                self._plot_correlation_heatmap(df)
                self._plot_processing_time_trend(df) # New plot for processing time trend
                self._create_detailed_plots_from_df(df) # Renamed to avoid confusion with create_visualizations wrapper

            logger.info(f"Visualizations created in {self.output_dir}")

        except Exception as e:
            logger.error(f"Error creating detailed visualizations: {e}")
            raise
        finally:
            # Restore the original output_dir for subsequent calls if needed
            self.output_dir = original_output_dir


    def _plot_event_accuracy_comparison(self, metrics: Dict[str, Any]):
        """Plot comparison of strict vs fuzzy accuracy for event types and sub-types."""
        event_type_strict = metrics.get('categorical_metrics', {}).get('event_type', {}).get('accuracy', 0)
        event_type_fuzzy = metrics.get('categorical_metrics', {}).get('event_type', {}).get('fuzzy_accuracy', 0)
        event_sub_type_strict = metrics.get('categorical_metrics', {}).get('event_sub_type', {}).get('accuracy', 0)
        event_sub_type_fuzzy = metrics.get('categorical_metrics', {}).get('event_sub_type', {}).get('fuzzy_accuracy', 0)

        event_metrics = {
            'Event Type': [event_type_strict, event_type_fuzzy],
            'Event Sub-type': [event_sub_type_strict, event_sub_type_fuzzy]
        }
        
        df_plot = pd.DataFrame(event_metrics, index=['Strict', 'Fuzzy'])
        
        if df_plot.empty or df_plot.isnull().all().all(): # Added check for all nulls
            logger.warning("No event accuracy metrics found to plot comparison or all values are null.")
            return

        logger.info("Plotting event accuracy comparison.")

        # Bar Chart
        plt.figure(figsize=(10, 6))
        ax = df_plot.plot(kind='bar', figsize=(10, 6), color=sns.color_palette("husl", df_plot.shape[1]), ax=plt.gca()) # Use gca() for current axes
        plt.title('Event Type and Sub-type Accuracy Comparison', fontsize=14, pad=20)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0,1)
        plt.xticks(rotation=0)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f')
        plt.tight_layout()
        self._save_plot('event_accuracy_comparison_bar.png')
        # plt.close() # Keep this if you don't want to show the plot immediately in an interactive env

        # Line chart
        plt.figure(figsize=(10, 6))
        for col in df_plot.columns:
            plt.plot(df_plot.index, df_plot[col], marker='o', linestyle='-', label=col)
            for i, val in enumerate(df_plot[col]):
                plt.annotate(f'{val:.2f}', (df_plot.index[i], val), textcoords="offset points", xytext=(0,10), ha='center')
        plt.title('Event Type and Sub-type Accuracy Comparison (Line Plot)', fontsize=14, pad=20)
        plt.xlabel('Accuracy Type', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0,1)
        plt.legend(title='Metric Type')
        plt.tight_layout()
        self._save_plot('event_accuracy_comparison_line.png')
        # plt.close() # Keep this if you don't want to show the plot immediately in an interactive env

    def _plot_text_similarity_distribution(self, df: pd.DataFrame):
        """
        Plot distributions of text similarity metrics using KDE, Histogram, and Violin plots.
        Dynamically identifies text similarity columns.
        Ensures clear boundaries for similarity scores [0, 1].
        """
        # Dynamically identify text similarity columns using regex patterns
        similarity_patterns = {
            'jaccard': r'_jaccard$',
            'bleu': r'_bleu$',
            'rouge_1': r'_rouge_1$',
            'rouge_2': r'_rouge_2$',
            'rouge_l': r'_rouge_l$',
            'llm_similarity': r'_llm_similarity$'
        }
        
        text_fields_and_metrics = {} # Stores {field: [metrics_present_for_field]}

        for col in df.columns:
            for metric_name, pattern in similarity_patterns.items():
                if re.search(pattern, col):
                    field_name = col.replace(re.search(pattern, col).group(0), '')
                    if field_name not in text_fields_and_metrics:
                        text_fields_and_metrics[field_name] = []
                    text_fields_and_metrics[field_name].append(metric_name)
                    break # Move to next column once a pattern is matched

        if not text_fields_and_metrics:
            logger.warning("No text similarity metrics columns found in DataFrame for detailed distribution plots.")
            return
            
        logger.info(f"Dynamically identified text similarity metrics for: {text_fields_and_metrics.keys()}")

        for field, metrics_list in text_fields_and_metrics.items():
            field_present_metrics_data = [] # To collect data for violin plot
            logger.info(f"Plotting detailed text similarity distributions for field: {field}")

            for metric in metrics_list: # Iterate through relevant metrics for this field
                col = f"{field}_{metric}"
                data_to_plot = pd.to_numeric(df[col], errors='coerce').dropna() # Convert to numeric, handle 'not specified'

                if not data_to_plot.empty:
                    # Store data for violin plot later
                    for val in data_to_plot:
                        field_present_metrics_data.append({'Metric': metric.replace('_', ' ').upper(), 'Score': val})

                    # KDE plot (with clipping for visual boundary adherence)
                    try:
                        plt.figure(figsize=(10, 5))
                        sns.kdeplot(data=data_to_plot, label=metric.replace('_', ' ').upper(), fill=True, alpha=0.6, clip=(0, 1))
                        plt.title(f'Distribution of {metric.replace("_", " ").upper()} - {field.replace("_", " ").title()} (KDE Plot)', fontsize=14)
                        plt.xlabel('Similarity Score')
                        plt.ylabel('Density')
                        plt.xlim(0, 1) # Explicitly set x-axis limits
                        plt.legend()
                        plt.tight_layout()
                        self._save_plot(f"{field}_{metric}_distribution_kde.png")
                    except Exception as e:
                        logger.warning(f"KDE Plot failed for {col}: {e}")

                    # Histogram for individual metric distribution
                    try:
                        plt.figure(figsize=(10, 5))
                        plt.hist(data_to_plot, bins=np.linspace(0, 1, 21), edgecolor='black', alpha=0.7, color=sns.color_palette("husl", 1)[0])
                        plt.title(f'Distribution of {metric.replace("_", " ").upper()} - {field.replace("_", " ").title()} (Histogram)', fontsize=14)
                        plt.xlabel('Similarity Score')
                        plt.ylabel('Frequency')
                        plt.xlim(0, 1) # Explicitly set x-axis limits
                        plt.tight_layout()
                        self._save_plot(f"{field}_{metric}_distribution_hist.png")
                    except Exception as e:
                        logger.warning(f"Histogram Plot failed for {col}: {e}")
                else:
                    logger.debug(f"Column '{col}' is empty or contains only non-numeric/null values — skipping individual plots.")

            # Violin Plot for combined metric distributions per field
            if field_present_metrics_data:
                combined_df = pd.DataFrame(field_present_metrics_data)
                plt.figure(figsize=(12, 7))
                sns.violinplot(x='Metric', y='Score', data=combined_df, hue='Metric', palette="viridis", legend=False, cut=0)
                plt.title(f'Distribution of Similarity Metrics for {field.replace("_", " ").title()} (Violin Plot)', fontsize=14)
                plt.xlabel('Metric', fontsize=12)
                plt.ylabel('Similarity Score', fontsize=12)
                plt.ylim(0, 1) # Explicitly set y-axis limits for clarity
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                self._save_plot(f"{field}_combined_metrics_distribution_violin.png")
            else:
                logger.debug(f"No combined metric data for {field} to plot violin plot.")

            # Scatter plot for Jaccard vs LLM Similarity (if both exist)
            jaccard_col = f"{field}_jaccard"
            llm_sim_col = f"{field}_llm_similarity"
            
            if jaccard_col in df.columns and llm_sim_col in df.columns:
                jaccard_data = pd.to_numeric(df[jaccard_col], errors='coerce').dropna()
                llm_sim_data = pd.to_numeric(df[llm_sim_col], errors='coerce').dropna()

                # Align indices to ensure plotting corresponding values
                aligned_df = pd.DataFrame({
                    'Jaccard': jaccard_data,
                    'LLMSimilarity': llm_sim_data
                }).dropna() # Drop rows where either is NaN after alignment

                if not aligned_df.empty:
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(x=aligned_df['Jaccard'], y=aligned_df['LLMSimilarity'], alpha=0.7, color=sns.color_palette("husl", 1)[0])
                    plt.title(f'Jaccard vs LLM Similarity for {field.replace("_", " ").title()} (Scatter Plot)', fontsize=14)
                    plt.xlabel(f'Jaccard Similarity for {field.replace("_", " ").title()}', fontsize=12)
                    plt.ylabel(f'LLM Similarity for {field.replace("_", " ").title()}', fontsize=12)
                    plt.xlim(0, 1) # Enforce x-axis limits
                    plt.ylim(0, 1) # Enforce y-axis limits
                    plt.tight_layout()
                    self._save_plot(f"{field}_jaccard_vs_llm_similarity_scatter.png")
                else:
                    logger.debug(f"Skipping Jaccard vs LLM Similarity scatter plot for {field}: No aligned non-empty data after cleaning.")
            else:
                logger.debug(f"Skipping Jaccard vs LLM Similarity scatter plot for {field}: Required columns not found or contain only non-numeric/null values.")


    def _plot_llm_similarity_scores(self, df: pd.DataFrame):
        """Plot LLM similarity scores for each text field using bar and pie charts."""
        # Dynamically identify LLM similarity columns
        llm_similarity_cols = [col for col in df.columns if col.endswith('_llm_similarity')]

        llm_similarity_data = []
        for col_name in llm_similarity_cols:
            cleaned_llm_data = pd.to_numeric(df[col_name], errors='coerce').dropna() # Convert to numeric, drop NaNs
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

        # Bar Plot for Average LLM Similarity
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Field', y='Average Similarity Score', data=mean_df, hue='Field', palette="husl", legend=False)
        plt.title('Average LLM Similarity Scores by Field', fontsize=14, pad=20)
        plt.ylabel('Average Similarity Score', fontsize=12)
        plt.ylim(0,1)
        plt.xticks(rotation=45, ha='right')
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f')
        plt.tight_layout()
        self._save_plot('llm_similarity_scores_bar.png')

        # Pie Chart for LLM Similarity Score Proportions
        if not mean_df.empty and mean_df['Average Similarity Score'].sum() > 0:
            plt.figure(figsize=(9, 9))
            plt.pie(mean_df['Average Similarity Score'], labels=mean_df['Field'], autopct='%1.1f%%', startangle=90,
                             colors=sns.color_palette("husl", len(mean_df)))
            plt.title('Proportion of Average LLM Similarity by Field', fontsize=14, pad=20)
            plt.tight_layout()
            self._save_plot('llm_similarity_scores_pie.png')
        else:
            logger.warning("No valid data for LLM similarity pie chart or sum of scores is zero.")


    def _plot_correlation_heatmap(self, df: pd.DataFrame):
        """Plot correlation heatmap of all dynamically identified numeric metrics."""
        df_numeric_for_corr = df.copy()
        
        # Dynamically identify columns that are likely to be numeric metrics.
        # This regex catches various accuracy, similarity, and score metrics.
        metric_patterns = r'(accuracy|jaccard|bleu|rouge_\d|rouge_l|llm_similarity|processing_time)$'
        potential_metric_cols = [
            col for col in df_numeric_for_corr.columns 
            if re.search(metric_patterns, col) or col == 'processing_time' # Include processing_time explicitly if not caught by pattern
        ]
        
        # Filter df_numeric_for_corr to only include these columns
        df_numeric_for_corr = df_numeric_for_corr[potential_metric_cols]

        # Convert all relevant columns to numeric, coercing errors, then drop NaNs for correlation
        for col in df_numeric_for_corr.columns:
            df_numeric_for_corr[col] = pd.to_numeric(df_numeric_for_corr[col], errors='coerce')

        # Drop columns that are entirely NaN after conversion, or original columns that were completely non-numeric
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
        if 'processing_time' not in df.columns:
            logger.warning("Column 'processing_time' not found in detailed results for trend plot.")
            return

        times = pd.to_numeric(df['processing_time'], errors='coerce').dropna()
        if times.empty:
            logger.warning("No valid numeric processing time data for trend plot.")
            return
        
        logger.info("Plotting processing time trend (stock-like graph).")

        plt.figure(figsize=(14, 7))
        # Using index as 'time' or 'record number' for a trend
        plt.plot(times.index, times.values, marker='o', linestyle='-', color='skyblue', label='Processing Time per Record')
        plt.title('Processing Time Trend per Record', fontsize=16, pad=20)
        plt.xlabel('Record Index', fontsize=12)
        plt.ylabel('Processing Time (seconds)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        self._save_plot('processing_time_trend.png')


    def _create_detailed_plots_from_df(self, df: pd.DataFrame) -> None: # Renamed this method
        """
        Create various detailed visualizations from evaluation results using the DataFrame.
        Dynamically identifies relevant numerical metrics.
        """
        # Dynamically identify columns that are numeric and represent metrics.
        # We can look for patterns that typically denote scores/accuracies.
        # Exclude 'id' or other non-metric numerical columns if they exist.
        metric_keywords = ['accuracy', 'jaccard', 'bleu', 'rouge', 'similarity', 'match', 'score', 'time']
        
        # Filter for columns that are numeric and likely represent metrics
        numeric_metric_cols = []
        for col in df.columns:
            if any(keyword in col for keyword in metric_keywords):
                # Attempt to convert to numeric. If it fails for all values, it's not a numeric metric.
                temp_series = pd.to_numeric(df[col], errors='coerce')
                if not temp_series.dropna().empty:
                    numeric_metric_cols.append(col)
        
        if not numeric_metric_cols:
            logger.warning("No valid numeric metric columns found in DataFrame for overall performance plotting.")
            return

        logger.info("Creating overall performance metrics bar chart from detailed results.")
        # 1. Overall Performance Metrics (Bar Chart)
        valid_metrics_data = df[numeric_metric_cols].dropna(how='all')
        if not valid_metrics_data.empty:
            # Ensure the columns are numeric before plotting their mean
            valid_metrics_data = valid_metrics_data.apply(pd.to_numeric, errors='coerce').dropna(how='all')
            
            if not valid_metrics_data.empty: # Re-check after numeric conversion
                plt.figure(figsize=(12, 6))
                ax = valid_metrics_data.mean().plot(kind='bar', color=sns.color_palette("husl", len(valid_metrics_data.columns)))
                plt.title('Overall Performance Metrics', fontsize=14, pad=20)
                plt.ylabel('Average Score', fontsize=12)
                # Set y-limit only if the maximum value is within a reasonable range (e.g., for scores 0-1)
                max_val = valid_metrics_data.mean().max()
                # Consider if processing time might skew this. For mixed metrics, 0-1 might not always apply.
                # If you have processing time in here, max_val could be > 1.
                # A more robust check might be to apply ylim only to metrics known to be [0,1]
                if all(val <= 1.05 for val in valid_metrics_data.mean().drop(labels=['processing_time'], errors='ignore')):
                    plt.ylim(0,1)
                plt.xticks(rotation=45, ha='right')
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.2f')
                self._save_plot('overall_metrics_bar.png')
            else:
                logger.warning("No valid numeric overall performance metrics found to plot after type conversion and NaN removal.")
        else:
            logger.warning("No valid numeric overall performance metrics found to plot (all NaN).")

        # You can add more dynamic detailed plots here if needed.
        # For instance, time series plots if 'timestamp' column exists.
        # Or individual record performance if 'record_id' exists and you want to show per-record scores.

# --- Test Data and Execution ---
if __name__ == "__main__":
    test_output_dir = "test_visualizations"
    
    # Clean up previous test run directory if it exists
    if Path(test_output_dir).exists():
        shutil.rmtree(test_output_dir)
        print(f"Cleaned up existing directory: {test_output_dir}")

    # Sample evaluation results
    sample_evaluation_results = {
        'aggregate_metrics': {
            'categorical_metrics': {
                'event_type': {'accuracy': 0.85, 'fuzzy_accuracy': 0.92},
                'event_sub_type': {'accuracy': 0.78, 'fuzzy_accuracy': 0.88},
                'severity': {'accuracy': 0.90}
            },
            'text_similarity_metrics': {
                'description': {
                    'mean_cosine_similarity': 0.75,
                    'std_cosine_similarity': 0.10,
                    'rouge_scores': {'rouge1': 0.65, 'rouge2': 0.55, 'rougeL': 0.60}
                },
                'summary': {
                    'mean_cosine_similarity': 0.82,
                    'std_cosine_similarity': 0.08,
                    'rouge_scores': {'rouge1': 0.70, 'rouge2': 0.62, 'rougeL': 0.68}
                }
            },
            'processing_metrics': {
                'mean_processing_time': 1.25,
                'individual_processing_times': [1.1, 1.3, 1.2, 1.5, 1.0, 1.4, 1.2, 1.1, 1.3, 1.2, 1.6, 1.0, 1.1, 1.2, 1.3]
            },
            'set_similarity_metrics': {
                'keywords': {
                    'mean_jaccard_similarity': 0.70,
                    'std_jaccard_similarity': 0.15
                }
            }
        },
        'detailed_results': [
            {
                'id': 'rec1', 'event_type_accuracy': 1.0, 'event_sub_type_accuracy': 1.0, 'severity_accuracy': 1.0,
                'description_jaccard': 0.7, 'description_bleu': 0.65, 'description_rouge_1': 0.6, 'description_rouge_2': 0.5, 'description_rouge_l': 0.55, 'description_llm_similarity': 0.8,
                'summary_jaccard': 0.8, 'summary_bleu': 0.75, 'summary_rouge_1': 0.7, 'summary_rouge_2': 0.6, 'summary_rouge_l': 0.65, 'summary_llm_similarity': 0.85,
                'keywords_jaccard': 0.75, 'processing_time': 1.1
            },
            {
                'id': 'rec2', 'event_type_accuracy': 0.0, 'event_sub_type_accuracy': 1.0, 'severity_accuracy': 0.0,
                'description_jaccard': 0.6, 'description_bleu': 0.55, 'description_rouge_1': 0.5, 'description_rouge_2': 0.4, 'description_rouge_l': 0.45, 'description_llm_similarity': 0.7,
                'summary_jaccard': 0.7, 'summary_bleu': 0.65, 'summary_rouge_1': 0.6, 'summary_rouge_2': 0.5, 'summary_rouge_l': 0.55, 'summary_llm_similarity': 0.78,
                'keywords_jaccard': 0.65, 'processing_time': 1.3
            },
            {
                'id': 'rec3', 'event_type_accuracy': 1.0, 'event_sub_type_accuracy': 0.0, 'severity_accuracy': 1.0,
                'description_jaccard': 0.8, 'description_bleu': 0.75, 'description_rouge_1': 0.7, 'description_rouge_2': 0.6, 'description_rouge_l': 0.65, 'description_llm_similarity': 0.88,
                'summary_jaccard': 0.9, 'summary_bleu': 0.85, 'summary_rouge_1': 0.8, 'summary_rouge_2': 0.7, 'summary_rouge_l': 0.75, 'summary_llm_similarity': 0.92,
                'keywords_jaccard': 0.80, 'processing_time': 1.2
            },
            {
                'id': 'rec4', 'event_type_accuracy': 0.0, 'event_sub_type_accuracy': 0.0, 'severity_accuracy': 0.0,
                'description_jaccard': 'not specified', 'description_bleu': 0.45, 'description_rouge_1': 0.4, 'description_rouge_2': 0.3, 'description_rouge_l': 0.35, 'description_llm_similarity': 0.6,
                'summary_jaccard': 0.6, 'summary_bleu': 0.55, 'summary_rouge_1': 0.5, 'summary_rouge_2': 0.4, 'summary_rouge_l': 0.45, 'summary_llm_similarity': 0.7,
                'keywords_jaccard': 0.55, 'processing_time': 1.5
            },
             {
                'id': 'rec5', 'event_type_accuracy': 1.0, 'event_sub_type_accuracy': 1.0, 'severity_accuracy': 1.0,
                'description_jaccard': 0.9, 'description_bleu': 0.85, 'description_rouge_1': 0.8, 'description_rouge_2': 0.7, 'description_rouge_l': 0.75, 'description_llm_similarity': 0.95,
                'summary_jaccard': 0.95, 'summary_bleu': 0.9, 'summary_rouge_1': 0.85, 'summary_rouge_2': 0.75, 'summary_rouge_l': 0.8, 'summary_llm_similarity': 0.98,
                'keywords_jaccard': 0.90, 'processing_time': 1.0
            },
            {
                'id': 'rec6', 'event_type_accuracy': 1.0, 'event_sub_type_accuracy': 1.0, 'severity_accuracy': 1.0,
                'description_jaccard': 0.85, 'description_bleu': 0.8, 'description_rouge_1': 0.75, 'description_rouge_2': 0.65, 'description_rouge_l': 0.7, 'description_llm_similarity': 0.9,
                'summary_jaccard': 0.88, 'summary_bleu': 0.82, 'summary_rouge_1': 0.78, 'summary_rouge_2': 0.68, 'summary_rouge_l': 0.73, 'summary_llm_similarity': 0.93,
                'keywords_jaccard': 0.85, 'processing_time': 1.6
            },
            {
                'id': 'rec7', 'event_type_accuracy': 0.0, 'event_sub_type_accuracy': 0.0, 'severity_accuracy': 0.0,
                'description_jaccard': 0.5, 'description_bleu': 0.4, 'description_rouge_1': 0.3, 'description_rouge_2': 0.2, 'description_rouge_l': 0.25, 'description_llm_similarity': 0.5,
                'summary_jaccard': 0.55, 'summary_bleu': 0.45, 'summary_rouge_1': 0.4, 'summary_rouge_2': 0.3, 'summary_rouge_l': 0.35, 'summary_llm_similarity': 0.6,
                'keywords_jaccard': 0.45, 'processing_time': 0.9
            }
        ]
    }

    visualizer = Visualizer()

    try:
        print(f"\nAttempting to create visualizations in: {test_output_dir}")
        visualizer.create_visualizations(sample_evaluation_results, test_output_dir)
        print(f"\nVisualizations creation process completed. Checking generated files...")

        # List files in the output directory
        generated_files = list(Path(test_output_dir).iterdir())
        if generated_files:
            print(f"\nSuccessfully generated {len(generated_files)} files in '{test_output_dir}':")
            for f in generated_files:
                print(f" - {f.name}")
            print("\nVerification successful: Plots were generated.")
        else:
            print("\nVerification failed: No plots were generated.")
            
    except Exception as e:
        print(f"\nAn error occurred during visualization creation: {e}")
        print("Verification failed: Plots were NOT generated due to an error.")

    finally:
        # Optional: Keep the directory for manual inspection or remove it
        # shutil.rmtree(test_output_dir)
        # print(f"\nCleaned up test directory: {test_output_dir}")
        pass