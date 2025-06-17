from typing import Dict, List, Any, Union, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from loguru import logger
from schema import ProcessedOutput
import json
import pandas as pd
from sklearn.metrics import jaccard_score
from rouge import Rouge
import requests
import re
from typing import Optional


class LLMJudge:
    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        self.ollama_base_url = ollama_base_url
        self.model_name = "llama3.1:8b"
        
    def _call_llm(self, prompt: str) -> str:
        """Call the Ollama LLM API and return the response string"""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            json_resp = response.json()
            return json_resp.get("response", "")
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            raise

    def is_similar(self, text1: str, text2: str) -> Tuple[Optional[float], Optional[str]]:
        prompt = f"""
Compare these two texts and determine if they are semantically similar. Consider:
1. Core meaning and intent
2. Key entities and actions
3. Context and implications

Text 1: "{text1}"
Text 2: "{text2}"

Rate similarity from 0.0 to 1.0 where:
0.0 = Completely different
0.5 = Somewhat related
1.0 = Identical or very similar

Output format:
Score: <number>
Explanation: <brief explanation>

Example:
Score: 0.8
Explanation: Both describe traffic accidents with similar severity and location details
"""
        try:
            response_text = self._call_llm(prompt)
            match = re.search(r"Score:\s*([0-1](?:\.\d+)?)", response_text)
            if match:
                return float(match.group(1)), response_text
            else:
                logger.warning(f"Could not parse LLM response: {response_text}")
                return None, response_text
        except Exception as e:
            logger.error(f"LLM similarity comparison failed: {e}")
            return None, None     
class Evaluator:
    def __init__(self, ground_truth_file: str):
        self.ground_truth = self._load_ground_truth(ground_truth_file)
        self.llm_judge = LLMJudge()
        self.rouge = Rouge()
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.vectorizer = TfidfVectorizer()
        
    def _load_ground_truth(self, file_path: str) -> Dict[str, ProcessedOutput]:
        """Load ground truth data with proper file name handling"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Convert to ProcessedOutput objects and normalize file names
            ground_truth = {}
            for item in data:
                # Ensure file name has .txt extension
                file_name = item.get('file_name', '')
                if not file_name.endswith('.txt'):
                    file_name = f"{file_name}.txt"
                    
                # Create ProcessedOutput object
                try:
                    output = ProcessedOutput(**item)
                    ground_truth[file_name] = output
                except Exception as e:
                    logger.error(f"Error creating ProcessedOutput for {file_name}: {str(e)}")
                    
            return ground_truth
            
        except Exception as e:
            logger.error(f"Error loading ground truth: {str(e)}")
            raise

    def _calculate_text_similarity(self, text1: str, text2: str) -> Dict[str, float]:
        """Calculate various text similarity metrics"""
        if not text1 or not text2:
            return {
                'jaccard': 0.0,
                'bleu': 0.0,
                'rouge_1': 0.0,
                'rouge_2': 0.0,
                'rouge_l': 0.0,
                'llm_similarity': 0.0
            }
            
        # Tokenize texts
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        # Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        jaccard = intersection / union if union > 0 else 0.0
        
        # BLEU score
        bleu = sentence_bleu([text1.split()], text2.split())
        
        # ROUGE scores
        rouge_scores = self.rouge.get_scores(text1, text2)[0]
        
        # LLM similarity
        llm_score, _ = self.llm_judge.is_similar(text1, text2)
        
        return {
            'jaccard': jaccard,
            'bleu': bleu,
            'rouge_1': rouge_scores['rouge-1']['f'],
            'rouge_2': rouge_scores['rouge-2']['f'],
            'rouge_l': rouge_scores['rouge-l']['f'],
            'llm_similarity': llm_score
        }

    def _calculate_event_similarity(self, pred_type: str, pred_sub_type: str, 
                                  true_type: str, true_sub_type: str) -> Dict[str, float]:
        """Calculate similarity metrics for event type and sub-type"""
        # Strict matching
        type_match = 1.0 if pred_type.upper() == true_type.upper() else 0.0
        sub_type_match = 1.0 if pred_sub_type.upper() == true_sub_type.upper() else 0.0
        
        # Fuzzy matching using LLM
        type_similarity, _ = self.llm_judge.is_similar(pred_type, true_type)
        sub_type_similarity, _ = self.llm_judge.is_similar(pred_sub_type, true_sub_type)
        
        return {
            'type_strict_match': type_match,
            'sub_type_strict_match': sub_type_match,
            'type_fuzzy_match': type_similarity,
            'sub_type_fuzzy_match': sub_type_similarity
        }

    def evaluate_predictions(self, predictions: List[ProcessedOutput]) -> Dict[str, Any]:
        """Evaluate predictions against ground truth"""
        results = []
        
        for pred in predictions:
            # Normalize file name
            file_name = pred.file_name
            if not file_name.endswith('.txt'):
                file_name = f"{file_name}.txt"
                
            if file_name not in self.ground_truth:
                logger.warning(f"No ground truth found for {file_name}")
                continue
                
            true = self.ground_truth[file_name]
            
            # Event type and sub-type evaluation
            event_metrics = self._calculate_event_similarity(
                pred.event_type, pred.event_sub_type,
                true.event_type, true.event_sub_type
            )
            
            # Text similarity metrics for all text fields
            text_metrics = {}
            for field in ['specified_matter', 'incident_location', 'area']:
                pred_text = getattr(pred, field, '')
                true_text = getattr(true, field, '')
                text_metrics[field] = self._calculate_text_similarity(pred_text, true_text)
            
            # Combine all metrics
            result = {
                'file_name': file_name,
                **event_metrics,
                **{f"{field}_{metric}": value 
                   for field, metrics in text_metrics.items() 
                   for metric, value in metrics.items()}
            }
            results.append(result)
            
        # Calculate aggregate metrics
        df = pd.DataFrame(results)
        aggregate_metrics = {
            'event_type_strict_accuracy': df['type_strict_match'].mean(),
            'event_type_fuzzy_accuracy': df['type_fuzzy_match'].mean(),
            'event_sub_type_strict_accuracy': df['sub_type_strict_match'].mean(),
            'event_sub_type_fuzzy_accuracy': df['sub_type_fuzzy_match'].mean(),
        }
        
        # Add text similarity metrics
        for field in ['specified_matter', 'incident_location', 'area']:
            for metric in ['jaccard', 'bleu', 'rouge_1', 'rouge_2', 'rouge_l', 'llm_similarity']:
                col = f"{field}_{metric}"
                if col in df.columns:
                    aggregate_metrics[f"{field}_{metric}_mean"] = df[col].mean()
        
        return {
            'detailed_results': results,
            'aggregate_metrics': aggregate_metrics
        }

    def save_evaluation_results(self, results: Dict[str, Any], output_file: str):
        """Save evaluation results to JSON file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Evaluation results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving evaluation results: {str(e)}")

    def _calculate_rouge_scores(self, pred: str, true: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        try:
            scores = self.scorer.score(true, pred)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.warning(f"Error calculating ROUGE scores: {str(e)}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    def _calculate_bleu_score(self, pred: str, true: str) -> float:
        """Calculate BLEU score"""
        try:
            return sentence_bleu([true.split()], pred.split())
        except Exception as e:
            logger.warning(f"Error calculating BLEU score: {str(e)}")
            return 0.0

    def _calculate_jaccard_similarity(self, pred: List[str], true: List[str]) -> float:
        """Calculate Jaccard similarity between two sets"""
        try:
            pred_set = set(pred)
            true_set = set(true)
            intersection = len(pred_set.intersection(true_set))
            union = len(pred_set.union(true_set))
            return intersection / union if union > 0 else 0.0
        except Exception as e:
            logger.warning(f"Error calculating Jaccard similarity: {str(e)}")
            return 0.0

    def evaluate_single(self, prediction: ProcessedOutput, ground_truth: ProcessedOutput) -> Dict[str, Any]:
        """Evaluate a single prediction against ground truth"""
        metrics = {
            'categorical_metrics': {},
            'text_similarity_metrics': {},
            'set_similarity_metrics': {},
            'processing_metrics': {}
        }
        
        # Categorical metrics
        categorical_fields = ['event_type', 'event_sub_type']
        for field in categorical_fields:
            pred_value = getattr(prediction, field)
            true_value = getattr(ground_truth, field)
            
            if pred_value is not None and true_value is not None:
                metrics['categorical_metrics'][field] = {
                    'accuracy': 1.0 if pred_value == true_value else 0.0
                }
        
        # Text similarity metrics are removed as 'location' is not in the schema
        
        # Set similarity metrics are removed as 'keywords' is not in the schema
        
        # Processing metrics
        metrics['processing_metrics'] = {
            'processing_time': prediction.processing_time
        }
        
        return metrics

    def evaluate_batch(self, predictions: List[ProcessedOutput], ground_truths: List[ProcessedOutput]) -> Dict[str, Any]:
        """Evaluate a batch of predictions against ground truths"""
        all_metrics = []
        
        for pred, true in zip(predictions, ground_truths):
            try:
                metrics = self.evaluate_single(pred, true)
                all_metrics.append(metrics)
            except Exception as e:
                logger.error(f"Error evaluating prediction: {str(e)}")
                continue
        
        # Aggregate metrics
        aggregated_metrics = {
            'categorical_metrics': {},
            'text_similarity_metrics': {},
            'set_similarity_metrics': {},
            'processing_metrics': {}
        }
        
        # Aggregate categorical metrics
        for field in ['event_type', 'event_sub_type']:
            accuracies = [m['categorical_metrics'].get(field, {}).get('accuracy', 0.0) 
                         for m in all_metrics if field in m['categorical_metrics']]
            if accuracies:
                aggregated_metrics['categorical_metrics'][field] = {
                    'accuracy': np.mean(accuracies)
                }
        
        # Aggregate text similarity metrics
        # Text similarity aggregation is removed as 'location' is not in the schema
        
        # Aggregate set similarity metrics
        # Set similarity aggregation is removed as 'keywords' is not in the schema
        
        # Aggregate processing metrics
        processing_times = [m['processing_metrics'].get('processing_time', 0.0) for m in all_metrics]
        if processing_times:
            aggregated_metrics['processing_metrics'] = {
                'mean_processing_time': np.mean(processing_times),
                'std_processing_time': np.std(processing_times),
                'total_processing_time': np.sum(processing_times)
            }
        
        return aggregated_metrics 