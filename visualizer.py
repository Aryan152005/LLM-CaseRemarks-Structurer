import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json
from pathlib import Path
from loguru import logger
import os

class Visualizer:
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        # Use a built-in style instead of seaborn
        plt.style.use('default')
        # Set the color palette using seaborn
        sns.set_palette("husl")
        
    def _save_plot(self, filename: str):
        """Save the current plot"""
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_categorical_metrics(self, metrics: Dict[str, Any]):
        """Plot categorical metrics (accuracy)"""
        categorical_data = metrics.get('categorical_metrics', {})
        if not categorical_data:
            return
            
        fields = list(categorical_data.keys())
        accuracies = [data['accuracy'] for data in categorical_data.values()]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(fields, accuracies)
        plt.title('Accuracy by Field', fontsize=14, pad=20)
        plt.xlabel('Field', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
                    
        self._save_plot('categorical_metrics.png')
        
    def plot_text_similarity_metrics(self, metrics: Dict[str, Any]):
        """Plot text similarity metrics"""
        text_data = metrics.get('text_similarity_metrics', {})
        if not text_data:
            return
            
        fields = list(text_data.keys())
        similarities = [data['mean_cosine_similarity'] for data in text_data.values()]
        stds = [data['std_cosine_similarity'] for data in text_data.values()]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(fields, similarities, yerr=stds, capsize=5)
        plt.title('Text Similarity by Field', fontsize=14, pad=20)
        plt.xlabel('Field', fontsize=12)
        plt.ylabel('Cosine Similarity', fontsize=12)
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
                    
        self._save_plot('text_similarity_metrics.png')
        
    def plot_processing_time(self, metrics: Dict[str, Any]):
        """Plot processing time metrics"""
        processing_data = metrics.get('processing_metrics', {})
        if not processing_data:
            return
            
        plt.figure(figsize=(10, 6))
        times = [processing_data['mean_processing_time']]
        labels = ['Mean Processing Time']
        
        bars = plt.bar(labels, times)
        plt.title('Processing Time', fontsize=14, pad=20)
        plt.ylabel('Time (seconds)', fontsize=12)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s',
                    ha='center', va='bottom')
            
        self._save_plot('processing_time.png')
        
    def plot_rouge_scores(self, metrics: Dict[str, Any]):
        """Plot ROUGE scores"""
        text_data = metrics.get('text_similarity_metrics', {})
        if not text_data:
            return
            
        fields = list(text_data.keys())
        rouge_scores = {
            'ROUGE-1': [],
            'ROUGE-2': [],
            'ROUGE-L': []
        }
        
        for field_data in text_data.values():
            rouge_data = field_data.get('rouge_scores', {})
            rouge_scores['ROUGE-1'].append(rouge_data.get('rouge1', 0))
            rouge_scores['ROUGE-2'].append(rouge_data.get('rouge2', 0))
            rouge_scores['ROUGE-L'].append(rouge_data.get('rougeL', 0))
            
        x = np.arange(len(fields))
        width = 0.25
        
        plt.figure(figsize=(14, 7))
        for i, (metric, scores) in enumerate(rouge_scores.items()):
            plt.bar(x + i*width, scores, width, label=metric)
            
        plt.title('ROUGE Scores by Field', fontsize=14, pad=20)
        plt.xlabel('Field', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.xticks(x + width, fields, rotation=45, ha='right')
        plt.legend(fontsize=10)
        plt.ylim(0, 1)
        
        self._save_plot('rouge_scores.png')
        
    def plot_keyword_similarity(self, metrics: Dict[str, Any]):
        """Plot keyword similarity metrics"""
        keyword_data = metrics.get('set_similarity_metrics', {}).get('keywords', {})
        if not keyword_data:
            return
            
        similarity = keyword_data.get('mean_jaccard_similarity', 0)
        std = keyword_data.get('std_jaccard_similarity', 0)
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(['Keywords'], [similarity], yerr=[std], capsize=5)
        plt.title('Keyword Similarity', fontsize=14, pad=20)
        plt.ylabel('Jaccard Similarity', fontsize=12)
        plt.ylim(0, 1)
        
        # Add value label
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        self._save_plot('keyword_similarity.png')
        
    def create_all_visualizations(self, metrics: Dict[str, Any]):
        """Create all visualizations from the metrics"""
        try:
            self.plot_categorical_metrics(metrics)
            self.plot_text_similarity_metrics(metrics)
            self.plot_processing_time(metrics)
            self.plot_rouge_scores(metrics)
            self.plot_keyword_similarity(metrics)
            
            # Save metrics as JSON for reference
            with open(self.output_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            raise

    def create_visualizations(self, evaluation_results: Dict[str, Any], output_dir: str):
        """Create visualizations for evaluation results"""
        try:
            # Convert results to DataFrame
            df = pd.DataFrame(evaluation_results['detailed_results'])
            metrics = evaluation_results['aggregate_metrics']
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. Event Type and Sub-type Accuracy Comparison
            self._plot_event_accuracy_comparison(metrics, output_dir)
            
            # 2. Text Similarity Metrics Distribution
            self._plot_text_similarity_distribution(df, output_dir)
            
            # 3. LLM Similarity Scores
            self._plot_llm_similarity_scores(df, output_dir)
            
            # 4. Correlation Heatmap
            self._plot_correlation_heatmap(df, output_dir)
            
            logger.info(f"Visualizations created in {output_dir}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
    
    def _plot_event_accuracy_comparison(self, metrics: Dict[str, float], output_dir: str):
        """Plot comparison of strict vs fuzzy accuracy for event types"""
        event_metrics = {
            'Event Type': [metrics['event_type_strict_accuracy'], metrics['event_type_fuzzy_accuracy']],
            'Event Sub-type': [metrics['event_sub_type_strict_accuracy'], metrics['event_sub_type_fuzzy_accuracy']]
        }
        
        df = pd.DataFrame(event_metrics, index=['Strict', 'Fuzzy'])
        ax = df.plot(kind='bar', figsize=(10, 6))
        plt.title('Event Type and Sub-type Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=0)
        
        # Add value labels
        for i in ax.containers:
            ax.bar_label(i, fmt='%.2f')
            
        plt.tight_layout()
        plt.savefig(f"{output_dir}/event_accuracy_comparison.png")
        plt.close()
    
    def _plot_text_similarity_distribution(self, df: pd.DataFrame, output_dir: str):
        possible_fields = ['specified_matter', 'incident_location', 'area']
        metrics = ['jaccard', 'bleu', 'rouge_1', 'rouge_2', 'rouge_l', 'llm_similarity']

        # Filter fields that are present in the dataframe
        text_fields = [
            field for field in possible_fields 
            if any(f"{field}_{metric}" in df.columns for metric in metrics)
        ]

        logger.info(f"Plotting text similarity metrics for: {text_fields}")

        for field in text_fields:
            for metric in metrics:
                col = f"{field}_{metric}"
                if col in df.columns:
                    try:
                        plt.figure(figsize=(10, 5))
                        sns.kdeplot(data=df[col].dropna(), label=metric.upper())
                        plt.title(f'Distribution of {metric.upper()} - {field.replace("_", " ").title()}')
                        plt.xlabel('Similarity Score')
                        plt.ylabel('Density')
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, f"{field}_{metric}_distribution.png"))
                        plt.close()
                    except Exception as e:
                        logger.warning(f"Plot failed for {col}: {str(e)}")
                else:
                    logger.warning(f"Column {col} not found in DataFrame — skipping plot.")

        
    
    def _plot_llm_similarity_scores(self, df: pd.DataFrame, output_dir: str):
        """Plot LLM similarity scores for each text field"""
        text_fields = ['specified_matter', 'incident_location', 'area']
        
        plt.figure(figsize=(10, 6))
        for field in text_fields:
            sns.barplot(x=[field], y=df[f"{field}_llm_similarity"].mean())
        
        plt.title('LLM Similarity Scores by Field')
        plt.ylabel('Average Similarity Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/llm_similarity_scores.png")
        plt.close()
    
    def _plot_correlation_heatmap(self, df: pd.DataFrame, output_dir: str):
        """Plot correlation heatmap of all metrics"""
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap of Evaluation Metrics')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_heatmap.png")
        plt.close()

    def _create_visualizations(self, df: pd.DataFrame) -> None:
        """Create visualizations from evaluation results"""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            
            # 1. Overall Performance Metrics
            metrics = ['type_strict_match', 'sub_type_strict_match', 
                      'specified_matter_jaccard', 'incident_location_jaccard', 'area_jaccard']
            
            plt.figure(figsize=(12, 6))
            df[metrics].mean().plot(kind='bar')
            plt.title('Overall Performance Metrics')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'overall_metrics.png'))
            plt.close()
            
            # 2. Event Type Distribution
            plt.figure(figsize=(10, 6))
            df['event_type'].value_counts().plot(kind='bar')
            plt.title('Event Type Distribution')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'event_type_distribution.png'))
            plt.close()
            
            # 3. Event Sub-type Distribution
            plt.figure(figsize=(12, 6))
            df['event_sub_type'].value_counts().plot(kind='bar')
            plt.title('Event Sub-type Distribution')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'event_sub_type_distribution.png'))
            plt.close()
            
            # 4. Victim State Distribution
            plt.figure(figsize=(10, 6))
            df['state_of_victim'].value_counts().plot(kind='bar')
            plt.title('Victim State Distribution')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'victim_state_distribution.png'))
            plt.close()
            
            # 5. Victim Gender Distribution
            plt.figure(figsize=(8, 6))
            df['victim_gender'].value_counts().plot(kind='pie', autopct='%1.1f%%')
            plt.title('Victim Gender Distribution')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'victim_gender_distribution.png'))
            plt.close()
            
            # 6. Performance by Event Type
            plt.figure(figsize=(12, 6))
            event_type_metrics = df.groupby('event_type')[metrics].mean()
            event_type_metrics.plot(kind='bar')
            plt.title('Performance by Event Type')
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'performance_by_event_type.png'))
            plt.close()
            
            # 7. Performance by Event Sub-type
            plt.figure(figsize=(15, 6))
            event_sub_type_metrics = df.groupby('event_sub_type')[metrics].mean()
            event_sub_type_metrics.plot(kind='bar')
            plt.title('Performance by Event Sub-type')
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'performance_by_event_sub_type.png'))
            plt.close()
            
            # 8. Processing Time Analysis
            plt.figure(figsize=(10, 6))
            plt.scatter(df['processing_time'], df[metrics].mean(axis=1))
            plt.xlabel('Processing Time (seconds)')
            plt.ylabel('Average Performance Score')
            plt.title('Processing Time vs Performance')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'processing_time_analysis.png'))
            plt.close()
            
            self._plot_text_similarity_distribution(df, self.output_dir)

            logger.info(f"Created visualizations in {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            raise 