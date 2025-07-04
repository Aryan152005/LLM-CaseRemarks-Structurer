# evaluator.py

from typing import Dict, List, Any, Union, Tuple, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize # <-- Keep this line (added in previous step)
from rouge_score import rouge_scorer
from loguru import logger
# Import both schemas
from schema import ProcessedOutput, GroundTruthOutput
import json
import pandas as pd
from sklearn.metrics import jaccard_score
# from rouge import Rouge # <-- REMOVE OR COMMENT OUT THIS LINE
import requests
import re
import os # For path manipulation
import sys
# Import for plotting
import matplotlib.pyplot as plt
import seaborn as sns

sys.setrecursionlimit(5000)


COMPLETENESS_CHECK_FIELDS = [
    "event_type", "event_sub_type", "state_of_victim", "victim_gender",
    "specified_matter", "date_reference", "frequency", "repeat_incident",
    "identification", "injury_type", "victim_age", "victim_relation",
    "incident_location", "area", "suspect_description", "object_involved",
    "date_of_birth", "used_weapons", "offender_relation", "mode_of_threat",
    "need_ambulance", "children_involved", "generated_event_sub_type_detail"
]

class LLMJudge:
    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        self.ollama_base_url = ollama_base_url
        self.model_name = "llama3.1:8b"
        
    def _call_llm(self, prompt: str) -> str:
        """Call the Ollama LLM API and return the response string"""
        try:
            cleaned_prompt_for_log = prompt[:200].replace('\n', ' ')
            logger.info(f"LLMJudge calling LLM with prompt (first 200 chars): {cleaned_prompt_for_log}...")
            
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
            return json_resp.get("response", "").strip()
        except requests.exceptions.ConnectionError as ce:
            logger.error(f"Connection Error to Ollama (LLMJudge): {ce}. Is Ollama server running at {self.ollama_base_url}?")
            raise
        except requests.exceptions.RequestException as re:
            logger.error(f"Request Error to Ollama (LLMJudge): {re}")
            raise
        except Exception as e:
            logger.error(f"Error calling LLM (LLMJudge): {str(e)}")
            raise

    def is_similar(self, text1: str, text2: str) -> Tuple[Optional[float], Optional[str]]:
        text1_clean = text1.strip() if text1 else ""
        text2_clean = text2.strip() if text2 else ""

        if not text1_clean and not text2_clean:
            return 1.0, "Both texts are empty, considered identical."
        if not text1_clean or not text2_clean:
            return 0.0, "One text is empty, the other is not."
        if text1_clean.lower() == text2_clean.lower():
            return 1.0, "Texts are identical (case-insensitive)."

        prompt = f"""
SYSTEM: You are an AI assistant designed to compare the semantic similarity of two given texts.
Your task is to analyze the core meaning, intent, key entities, actions, context, and implications of both texts.
Your response must strictly adhere to the following output format. Do not include any other conversational text or pleasantries.

Output Format:
Score: <number from 0.0 to 1.0, typically 2 decimal places>
Explanation: <brief explanation of the similarity or lack thereof>

If you cannot perform the comparison due to sensitive content, policy violations, or if the texts are too vague/unrelated to provide a meaningful semantic score, output 'Score: 0.0' and explain why (e.g., "LLM refused: Sensitive content").

Example 1:
TEXT1: "The cat sat on the mat."
TEXT2: "A feline was resting on the rug."
Score: 0.95
Explanation: Both sentences convey the same meaning using different but semantically similar words.

Example 2:
TEXT1: "I need to buy some groceries for dinner."
TEXT2: "The stock market is crashing."
Score: 0.05
Explanation: The two sentences are completely unrelated in meaning and context.

Example 3 (Sensitive Content Example):
TEXT1: "Discussion about illicit drugs."
TEXT2: "How to manufacture explosives."
Score: 0.0
Explanation: LLM refused: Content violates safety guidelines regarding illegal activities and dangerous materials.

Example 4 (Vague Content Example):
TEXT1: "OTHERS"
TEXT2: "GENERAL NUISANCE"
Score: 0.1
Explanation: The terms are too generic and lack sufficient context for a precise semantic comparison.

USER:
Compare the semantic similarity of the following two texts.

TEXT1:
{text1_clean}

TEXT2:
{text2_clean}

ASSISTANT:
"""
        try:
            response_text = self._call_llm(prompt)
            
            match = re.search(r"Score:\s*([0-1](?:\.\d+)?)\s*(?:\n|$)", response_text, re.IGNORECASE)
            explanation_match = re.search(r"Explanation:\s*(.*)", response_text, re.IGNORECASE)
            explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided."

            if match:
                try:
                    score = float(match.group(1))
                    if 0.0 <= score <= 1.0:
                        return score, f"Score extracted: {score}. Explanation: {explanation}"
                    else:
                        logger.warning(f"LLM returned out-of-range score: {score}. Raw response: '{response_text.strip()}'")
                        return 0.0, f"LLM outputted invalid score: {score}. Raw response: {response_text.strip()}"
                except ValueError:
                    logger.warning(f"Could not convert extracted score '{match.group(1)}' to float. Raw response: '{response_text.strip()}'")
                    return 0.0, f"Failed to convert score: {response_text.strip()}"
            else:
                logger.warning(f"Could not parse LLM response for similarity score (no 'Score:' found or malformed). Raw response: '{response_text.strip()}'")
                
                refusal_phrases = ["cannot compare", "not able to compare", "refused", "sensitive content", "safety guidelines", "inappropriate", "illegal activities", "self-harm", "hate speech"]
                if any(phrase in response_text.lower() for phrase in refusal_phrases):
                    logger.warning(f"LLM explicitly refused comparison. Raw response: '{response_text.strip()}'")
                    return 0.0, f"LLM explicitly refused: {response_text.strip()}"
                else:
                    return 0.0, f"LLM response unparseable: {response_text.strip()}"
            
        except Exception as e:
            logger.error(f"LLM similarity comparison failed for texts (first 50 chars): '{text1_clean[:50]}' vs '{text2_clean[:50]}': {e}")
            return 0.0, f"Exception during LLM call: {str(e)}"
            
class Evaluator:
    def __init__(self, ground_truth_file: str):
        self.ground_truth = self._load_ground_truth(ground_truth_file)
        self.llm_judge = LLMJudge()
        # self.rouge = Rouge()
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.vectorizer = TfidfVectorizer()
        self.smoothing_function = SmoothingFunction()

    def _load_ground_truth(self, file_path: str) -> Dict[str, GroundTruthOutput]:
        """Load ground truth data with proper file name handling, using GroundTruthOutput model."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            ground_truth_map = {}
            for item in data:
                raw_file_name = item.get('file_name', '')
                base_file_name = os.path.basename(raw_file_name)
                if not base_file_name.endswith('.txt'):
                    base_file_name = f"{base_file_name}.txt"
                
                try:
                    cleaned_item = {}
                    for k, v in item.items():
                        if isinstance(v, str) and v.strip().lower() in ["null", "none", ""]:
                            cleaned_item[k] = "Not specified" if k in ["state_of_victim", "victim_gender"] else None
                        elif v is None:
                            cleaned_item[k] = "Not specified" if k in ["state_of_victim", "victim_gender"] else None
                        else:
                            cleaned_item[k] = v

                    gt_data_for_model = {k: cleaned_item[k] for k in GroundTruthOutput.model_fields.keys() if k in cleaned_item}
                    # IMPORTANT: Add 'generated_event_sub_type_detail' to GroundTruthOutput if it's expected
                    # For now, if not present in GT, it will default to None/'' in comparison.
                    # When you generate GT for this, ensure it's included.
                    
                    gt_output = GroundTruthOutput(**gt_data_for_model)
                    ground_truth_map[base_file_name] = gt_output
                except Exception as e:
                    logger.error(f"Error creating GroundTruthOutput for {base_file_name} from data {item}: {e}")
                    continue
                    
            return ground_truth_map
                
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from ground truth file {file_path}: {e}")
            raise
        except FileNotFoundError:
            logger.error(f"Ground truth file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading ground truth: {e}")
            raise

    

    def _calculate_text_similarity(self, text1: str, text2: str) -> Dict[str, float]:
        """
        Calculate various text similarity metrics using rouge-score for ROUGE and
        adding explicit string conversion for robustness.
        """
        # Ensure texts are always strings, converting None to empty string
        text1_safe = str(text1) if text1 is not None else ""
        text2_safe = str(text2) if text2 is not None else ""

        # Normalize texts by stripping whitespace. This handles strings that are
        # just spaces or contain only whitespace characters.
        text1_stripped = text1_safe.strip()
        text2_stripped = text2_safe.strip()

        # Handle cases where both texts are empty after stripping
        if not text1_stripped and not text2_stripped:
            return {
                'jaccard': 1.0,
                'bleu': 1.0,
                'rouge_1': 1.0,
                'rouge_2': 1.0,
                'rouge_l': 1.0,
                'llm_similarity': 1.0
            }
        # Handle cases where one text is empty and the other is not
        elif not text1_stripped or not text2_stripped:
            return {
                'jaccard': 0.0,
                'bleu': 0.0,
                'rouge_1': 0.0,
                'rouge_2': 0.0,
                'rouge_l': 0.0,
                'llm_similarity': 0.0
            }

        # --- Jaccard Similarity ---
        # Convert to lowercase and tokenize using NLTK for better results
        tokens1_list = word_tokenize(text1_stripped.lower()) # Changed from .split()
        tokens2_list = word_tokenize(text2_stripped.lower()) # Changed from .split()

        # Convert to set for Jaccard calculation
        tokens1_set = set(tokens1_list)
        tokens2_set = set(tokens2_list)

        intersection = len(tokens1_set.intersection(tokens2_set))
        union = len(tokens1_set.union(tokens2_set))
        jaccard = intersection / union if union > 0 else 0.0

        # --- BLEU Score ---
        # NLTK's sentence_bleu expects a list of reference token lists.
        # Use the already tokenized lists for BLEU
        bleu_ref_tokens = tokens1_list # Using word_tokenize output
        bleu_hyp_tokens = tokens2_list # Using word_tokenize output

        # Check if either token list is empty before calculating BLEU
        if bleu_ref_tokens and bleu_hyp_tokens:
            bleu = sentence_bleu([bleu_ref_tokens], bleu_hyp_tokens, smoothing_function=self.smoothing_function.method1)
        else:
            bleu = 0.0 # Cannot calculate BLEU if one or both token lists are empty

        # --- ROUGE Scores (using rouge-score library) ---
        rouge_scores_output = {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}
        try:
            # Use self.scorer (from rouge_scorer.RougeScorer)
            scores = self.scorer.score(text1_stripped, text2_stripped)

            # Map the rouge-score output format to your desired format
            rouge_scores_output['rouge-1']['f'] = scores['rouge1'].fmeasure
            rouge_scores_output['rouge-2']['f'] = scores['rouge2'].fmeasure
            rouge_scores_output['rouge-l']['f'] = scores['rougeL'].fmeasure # 'rougeL' is the key for rouge-score

        except Exception as e: # Catch broader exceptions for robustness
            logger.warning(f"ROUGE calculation error with rouge-score: {e} for '{text1_stripped[:50]}' vs '{text2_stripped[:50]}'")

        # --- LLM Similarity ---
        llm_score, _ = self.llm_judge.is_similar(text1_stripped, text2_stripped)
        llm_score = llm_score if llm_score is not None else 0.0 # Ensure it's a float

        return {
            'jaccard': jaccard,
            'bleu': bleu,
            'rouge_1': rouge_scores_output['rouge-1']['f'],
            'rouge_2': rouge_scores_output['rouge-2']['f'],
            'rouge_l': rouge_scores_output['rouge-l']['f'],
            'llm_similarity': llm_score
        }
    
    def _is_field_complete(self, value: Any) -> bool:
        """Determines if a field value is considered 'complete' based on user's definition."""
        if value is None:
            return False
        if isinstance(value, str):
            stripped_value = value.strip().lower()
            return stripped_value not in ["", "not specified", "not applicable"]
        return True
    
    def _calculate_completeness_score(self, processed_output: ProcessedOutput, ground_truth: GroundTruthOutput) -> float:
        """
        Calculates a completeness score for a single record.
        The score is based on the percentage of non-empty comparable fields in the processed output
        that are also non-empty in the ground truth.
        """
        populated_fields_in_gt = 0
        populated_and_matched_in_processed = 0

        for field in COMPLETENESS_CHECK_FIELDS:
            gt_value = getattr(ground_truth, field, None)
            processed_value = getattr(processed_output, field, None)

            # Consider a field 'populated' if its value is not None, not an empty string/list, and not "not specified"
            is_gt_populated = gt_value is not None and \
                              (str(gt_value).strip().lower() != 'not specified' if isinstance(gt_value, str) else True) and \
                              (gt_value != "" if isinstance(gt_value, str) else True)
                              
            is_processed_populated = processed_value is not None and \
                                     (str(processed_value).strip().lower() != 'not specified' if isinstance(processed_value, str) else True) and \
                                     (processed_value != "" if isinstance(processed_value, str) else True)

            if is_gt_populated:
                populated_fields_in_gt += 1
                if is_processed_populated:
                    populated_and_matched_in_processed += 1
        
        if populated_fields_in_gt == 0:
            return 100.0 # If ground truth has no populated fields to check, consider it 100% complete
        
        # Score based on how many ground truth populated fields were also populated in processed output
        completeness = (populated_and_matched_in_processed / populated_fields_in_gt) * 100
        return completeness

    def _calculate_event_similarity(self, pred_type: str, pred_sub_type: str, 
                                     true_type: str, true_sub_type: str) -> Dict[str, float]:
        """Calculate similarity metrics for event type and sub-type"""
        pred_type_norm = pred_type.strip().upper() if pred_type else ""
        pred_sub_type_norm = pred_sub_type.strip().upper() if pred_sub_type else ""
        true_type_norm = true_type.strip().upper() if true_type else ""
        true_sub_type_norm = true_sub_type.strip().upper() if true_sub_type else ""

        type_match = 1.0 if pred_type_norm == true_type_norm and pred_type_norm not in ["", "NOT SPECIFIED"] else 0.0
        sub_type_match = 1.0 if pred_sub_type_norm == true_sub_type_norm and pred_sub_type_norm not in ["", "NOT SPECIFIED"] else 0.0
        
        llm_type_pred = pred_type if pred_type_norm not in ["", "NOT SPECIFIED"] else ""
        llm_type_true = true_type if true_type_norm not in ["", "NOT SPECIFIED"] else ""
        llm_sub_type_pred = pred_sub_type if pred_sub_type_norm not in ["", "NOT SPECIFIED"] else ""
        llm_sub_type_true = true_sub_type if true_sub_type_norm not in ["", "NOT SPECIFIED"] else ""

        type_similarity, _ = self.llm_judge.is_similar(llm_type_pred, llm_type_true)
        sub_type_similarity, _ = self.llm_judge.is_similar(llm_sub_type_pred, llm_sub_type_true)
        
        type_similarity = type_similarity if type_similarity is not None else 0.0
        sub_type_similarity = sub_type_similarity if sub_type_similarity is not None else 0.0
        
        return {
            'type_strict_match': type_match,
            'sub_type_strict_match': sub_type_match,
            'type_fuzzy_match': type_similarity,
            'sub_type_fuzzy_match': sub_type_similarity
        }

    def plot_event_metrics(self, results_df: pd.DataFrame, output_dir: str = 'plots'):
        """
        Generates bar plots for event type and sub-type accuracy (strict and fuzzy).
        """
        if results_df.empty:
            logger.warning("No data in results_df to plot event metrics.")
            return

        os.makedirs(output_dir, exist_ok=True)
        
        metrics_to_plot = {
            'Event Type Strict Accuracy': 'type_strict_match',
            'Event Type Fuzzy Accuracy (LLM)': 'type_fuzzy_match',
            'Event Sub-Type Strict Accuracy': 'sub_type_strict_match',
            'Event Sub-Type Fuzzy Accuracy (LLM)': 'sub_type_fuzzy_match',
        }

        # Calculate mean for each metric across all samples
        plot_data = {
            'Metric': [],
            'Mean Score': []
        }

        for display_name, col_name in metrics_to_plot.items():
            if col_name in results_df.columns:
                mean_score = results_df[col_name].mean()
                plot_data['Metric'].append(display_name)
                plot_data['Mean Score'].append(mean_score)
            else:
                logger.warning(f"Column '{col_name}' not found in DataFrame for plotting. Skipping.")

        if not plot_data['Metric']:
            logger.warning("No event metrics found to plot. Ensure correct columns are present.")
            return

        plot_df = pd.DataFrame(plot_data)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Metric', y='Mean Score', data=plot_df, palette='viridis')
        plt.title('Mean Accuracy for Event Type and Sub-Type', fontsize=16)
        plt.ylabel('Mean Score (0-1.0)', fontsize=12)
        plt.xlabel('') # No label needed for x-axis as labels are descriptive
        plt.ylim(0, 1) # Scores are between 0 and 1
        plt.xticks(rotation=15, ha='right', fontsize=10) # Rotate labels for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add value labels on top of bars
        for index, row in plot_df.iterrows():
            plt.text(index, row['Mean Score'] + 0.02, f"{row['Mean Score']:.2f}", color='black', ha="center", fontsize=10)

        plot_path = os.path.join(output_dir, 'event_type_subtype_accuracy.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Event type and sub-type accuracy plot saved to {plot_path}")


    def evaluate_predictions(self, predictions: List[ProcessedOutput], output_dir: str = 'plots') -> Dict[str, Any]:
        """Evaluate predictions against ground truth and generate plots."""
        results = []
        
        # Define all comparable text-based fields present in both ProcessedOutput and GroundTruthOutput
        # IMPORTANT: Ensure this list accurately reflects fields in BOTH schemas that you want to compare.
        # Added 'generated_event_sub_type_detail' here.
        comparable_text_fields = [
            "specified_matter", "date_reference", "frequency", "repeat_incident",
            "identification", "injury_type", "victim_age", "victim_relation",
            "incident_location", "area", "suspect_description", "object_involved",
            "used_weapons", "offender_relation", "mode_of_threat", "need_ambulance",
            "children_involved", "date_of_birth", "generated_event_sub_type_detail" 
        ]

        # Define categorical fields that will be evaluated for strict accuracy
        # event_type and event_sub_type are INCLUDED here now to ensure their strict match is part of the dataframe and aggregate
        categorical_fields_for_accuracy = [
            'event_type', 'event_sub_type', 'state_of_victim', 'victim_gender'
        ]

        for pred in predictions:
            file_name_key = os.path.basename(pred.file_name if pred.file_name else "")
            if not file_name_key.endswith('.txt'):
                file_name_key = f"{file_name_key}.txt"
                
            if file_name_key not in self.ground_truth:
                logger.warning(f"No ground truth found for {file_name_key}. Skipping evaluation for this prediction.")
                # Initialize result dict for the current prediction if we are skipping for lack of ground truth
                current_result = {'file_name': file_name_key}
                current_result['processing_metrics'] = {'processing_time': pred.processing_time}
                current_result['completeness_score'] = 0.0 # Assign 0 completeness if no GT to compare against
                # Set all other metrics to 0 or appropriate default if no GT
                current_result.update({
                    'type_strict_match': 0.0, 'sub_type_strict_match': 0.0,
                    'type_fuzzy_match': 0.0, 'sub_type_fuzzy_match': 0.0
                })
                for field in categorical_fields_for_accuracy:
                    current_result[f'{field}_strict_match'] = 0.0
                for field in comparable_text_fields:
                    current_result[f"{field}_jaccard"] = 0.0
                    current_result[f"{field}_bleu"] = 0.0
                    current_result[f"{field}_rouge_1"] = 0.0
                    current_result[f"{field}_rouge_2"] = 0.0
                    current_result[f"{field}_rouge_l"] = 0.0
                    current_result[f"{field}_llm_similarity"] = 0.0
                results.append(current_result) # Add this skipped result to maintain structure
                continue # Now continue with the next prediction

            true = self.ground_truth[file_name_key]
            
            # Initialize result dict for the current prediction
            current_result = {'file_name': file_name_key}
            current_result['processing_metrics'] = {'processing_time': pred.processing_time}

            # Evaluate Event Type and Sub-type (categorical + fuzzy from LLM)
            # These values (type_strict_match, sub_type_strict_match, type_fuzzy_match, sub_type_fuzzy_match)
            # are directly added to current_result.
            event_metrics = self._calculate_event_similarity(
                pred.event_type, pred.event_sub_type,
                true.event_type, true.event_sub_type
            )
            current_result.update(event_metrics)
            
            # Calculate and add completeness score
            completeness_score = self._calculate_completeness_score(pred, true)
            current_result['completeness_score'] = completeness_score

            # Add strict categorical matches for ALL fields in categorical_fields_for_accuracy,
            # INCLUDING event_type and event_sub_type.
            # The values from event_metrics for type_strict_match and sub_type_strict_match
            # will be used if they exist, or re-calculated here.
            # It's generally better to use the specific calculation if available, as in event_metrics.
            # So, we ensure that if 'event_type' or 'event_sub_type' are in categorical_fields_for_accuracy,
            # their 'strict_match' metric is correctly taken from event_metrics.
            for field in categorical_fields_for_accuracy:
                if field == 'event_type':
                    current_result[f'{field}_strict_match'] = event_metrics.get('type_strict_match', 0.0)
                elif field == 'event_sub_type':
                    current_result[f'{field}_strict_match'] = event_metrics.get('sub_type_strict_match', 0.0)
                else:
                    pred_value = getattr(pred, field, None)
                    true_value = getattr(true, field, None)
                    
                    pred_val_comp = pred_value.strip().upper() if isinstance(pred_value, str) else ''
                    true_val_comp = true_value.strip().upper() if isinstance(true_value, str) else ''

                    current_result[f'{field}_strict_match'] = 1.0 if (pred_val_comp and true_val_comp and pred_val_comp == true_val_comp) else 0.0

            # Text similarity metrics for all relevant text fields
            # Including 'generated_event_sub_type_detail' here
            for field in comparable_text_fields:
                pred_text = getattr(pred, field, None)
                true_text = getattr(true, field, None)
                
                pred_text_clean = pred_text if pred_text is not None and pred_text.lower() != 'not specified' else ''
                true_text_clean = true_text if true_text is not None and true_text.lower() != 'not specified' else ''

                field_metrics = self._calculate_text_similarity(pred_text_clean, true_text_clean)
                for metric, value in field_metrics.items():
                    current_result[f"{field}_{metric}"] = value
                
            results.append(current_result)
            
        if not results:
            logger.warning("No evaluation results to aggregate. Check if ground truth files match predictions.")
            return {'detailed_results': [], 'aggregate_metrics': {}}
        else: # This 'else' block ensures df is always defined before use
            df = pd.DataFrame(results)
            
            # Initialize aggregate_metrics
            aggregate_metrics = {}

            # Aggregate event type and sub-type (now correctly picked from event_metrics via current_result)
            # These columns will exist in the DataFrame because of current_result.update(event_metrics)
            aggregate_metrics['event_type_strict_accuracy'] = df['type_strict_match'].mean() if 'type_strict_match' in df.columns else 0.0
            aggregate_metrics['event_type_fuzzy_accuracy'] = df['type_fuzzy_match'].mean() if 'type_fuzzy_match' in df.columns else 0.0
            aggregate_metrics['event_sub_type_strict_accuracy'] = df['sub_type_strict_match'].mean() if 'sub_type_strict_match' in df.columns else 0.0
            aggregate_metrics['event_sub_type_fuzzy_accuracy'] = df['sub_type_fuzzy_match'].mean() if 'sub_type_fuzzy_match' in df.columns else 0.0

            # Aggregate other categorical fields (e.g., state_of_victim, victim_gender)
            # This loop now correctly handles all fields in categorical_fields_for_accuracy,
            # including event_type and event_sub_type whose strict matches were ensured above.
            for field in categorical_fields_for_accuracy:
                col_name = f"{field}_strict_match"
                if col_name in df.columns:
                    aggregate_metrics[f"{field}_strict_accuracy"] = df[col_name].mean()
                else:
                    aggregate_metrics[f"{field}_strict_accuracy"] = 0.0

            # Aggregate text similarity metrics
            for field in comparable_text_fields:
                for metric in ['jaccard', 'bleu', 'rouge_1', 'rouge_2', 'rouge_l', 'llm_similarity']:
                    col = f"{field}_{metric}"
                    if col in df.columns:
                        aggregate_metrics[f"{field}_{metric}_mean"] = df[col].mean()
            
            # Aggregate completeness score
            if 'completeness_score' in df.columns:
                aggregate_metrics['completeness_metrics'] = {
                    'mean_completeness_score': df['completeness_score'].mean(),
                    'std_completeness_score': df['completeness_score'].std() if len(df['completeness_score']) > 1 else 0.0
                }
            else:
                aggregate_metrics['completeness_metrics'] = {
                    'mean_completeness_score': 0.0,
                    'std_completeness_score': 0.0
                }

            # Call the plotting function here
            self.plot_event_metrics(df, output_dir) # Pass the DataFrame and output directory

            return {
                'detailed_results': results,
                'aggregate_metrics': aggregate_metrics
            }
    
    def save_evaluation_results(self, results: Dict[str, Any], output_file: str):
        """Save evaluation results to JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Evaluation results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving evaluation results: {str(e)}")

    def _calculate_rouge_and_bleu(self, text1: str, text2: str) -> Dict[str, float]:
        # This method is not called anywhere in the current Evaluator logic.
        # It seems redundant with _calculate_text_similarity. Keeping it if it has future use.
        scores = {}
        
        ref_tokens = text1.lower().split() if text1 else []
        hyp_tokens = text2.lower().split() if text2 else []

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(text1, text2) 

        scores['rouge1_fmeasure'] = rouge_scores['rouge1'].fmeasure
        scores['rouge2_fmeasure'] = rouge_scores['rouge2'].fmeasure
        scores['rougeL_fmeasure'] = rouge_scores['rougeL'].fmeasure

        if ref_tokens and hyp_tokens:
            scores['bleu_score'] = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=self.smoothing_function.method1)
        else:
            scores['bleu_score'] = 0.0

        return scores

    def _calculate_bleu_score(self, pred: str, true: str) -> float:
        # This method is not called anywhere in the current Evaluator logic.
        # It seems redundant with _calculate_text_similarity. Keeping it if it has future use.
        """Calculate BLEU score"""
        if not pred or not true:
            return 0.0
        try:
            pred_tokens = pred.lower().split()
            true_tokens = true.lower().split()
            if pred_tokens and true_tokens:
                return sentence_bleu([true_tokens], pred_tokens, smoothing_function=self.smoothing_function.method1)
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"Error calculating BLEU score: {str(e)}. Pred: '{pred[:50]}', True: '{true[:50]}'")
            return 0.0

    def _calculate_jaccard_similarity(self, pred: List[str], true: List[str]) -> float:
        # This method is not called anywhere in the current Evaluator logic.
        # It seems redundant with _calculate_text_similarity. Keeping it if it has future use.
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

    def evaluate_single(self, prediction: ProcessedOutput, ground_truth: GroundTruthOutput) -> Dict[str, Any]:
        """
        Evaluate a single prediction against ground truth.
        """
        metrics = {
            'categorical_metrics': {},
            'text_similarity_metrics': {},
            'processing_metrics': {}    
        }
        
        # Categorical metrics
        # state_of_victim and victim_gender strict matches
        categorical_fields_strict = ['state_of_victim', 'victim_gender']    
        for field in categorical_fields_strict:
            pred_value = getattr(prediction, field, None)
            true_value = getattr(ground_truth, field, None)
            
            pred_val_comp = pred_value.upper() if isinstance(pred_value, str) else ''
            true_val_comp = true_value.upper() if isinstance(true_value, str) else ''

            if pred_val_comp and true_val_comp:
                metrics['categorical_metrics'][field] = {
                    'accuracy': 1.0 if pred_val_comp == true_val_comp else 0.0    
                }
            else:
                    metrics['categorical_metrics'][field] = {'accuracy': 0.0}

        # Event type and sub-type evaluation (combines strict and fuzzy)
        event_metrics = self._calculate_event_similarity(
            prediction.event_type, prediction.event_sub_type,
            ground_truth.event_type, ground_truth.event_sub_type
        )
        # Add event_type/sub_type directly to categorical_metrics for consistency in single evaluation output
        metrics['categorical_metrics']['event_type'] = {
            'strict_accuracy': event_metrics['type_strict_match'],
            'fuzzy_accuracy': event_metrics['type_fuzzy_match']
        }
        metrics['categorical_metrics']['event_sub_type'] = {
            'strict_accuracy': event_metrics['sub_type_strict_match'],
            'fuzzy_accuracy': event_metrics['sub_type_fuzzy_match']
        }

        # Text similarity for relevant fields
        # IMPORTANT: Added 'generated_event_sub_type_detail' here
        comparable_text_fields = [
            "specified_matter", "date_reference", "frequency", "repeat_incident",
            "identification", "injury_type", "victim_age", "victim_relation",
            "incident_location", "area", "suspect_description", "object_involved",
            "used_weapons", "offender_relation", "mode_of_threat", "need_ambulance",
            "children_involved", "date_of_birth", "generated_event_sub_type_detail"
        ]

        for field in comparable_text_fields:
            pred_text = getattr(prediction, field, None)
            true_text = getattr(ground_truth, field, None)
            
            pred_text_clean = pred_text if pred_text is not None and pred_text.lower() != 'not specified' else ''
            true_text_clean = true_text if true_text is not None and true_text.lower() != 'not specified' else ''

            metrics['text_similarity_metrics'][field] = self._calculate_text_similarity(pred_text_clean, true_text_clean)


        # Processing metrics (only from prediction)
        metrics['processing_metrics'] = {
            'processing_time': prediction.processing_time if hasattr(prediction, 'processing_time') else None
        }
        
        return metrics

    def evaluate_batch(self, predictions: List[ProcessedOutput], ground_truths: List[GroundTruthOutput]) -> Dict[str, Any]:
        """
        Evaluate a batch of predictions against ground truths, assuming a 1:1 correspondence
        between the order of predictions and ground_truths lists.
        """
        if len(predictions) != len(ground_truths):
            logger.error("Mismatch in number of predictions and ground truths for batch evaluation.")
            raise ValueError("Number of predictions must match number of ground truths.")

        all_metrics = []
        
        for pred, true in zip(predictions, ground_truths):
            try:
                metrics = self.evaluate_single(pred, true)
                all_metrics.append(metrics)
            except Exception as e:
                logger.error(f"Error evaluating prediction for {pred.file_name if hasattr(pred, 'file_name') else 'unknown file'}: {str(e)}")
                continue
        
        # Aggregate metrics
        aggregated_metrics = {
            'categorical_metrics': {},
            'text_similarity_metrics': {},    
            'processing_metrics': {}
        }
        
        # Aggregate categorical metrics (including event_type and sub_type)
        # Use the structure from evaluate_single for aggregation clarity
        categorical_fields_for_aggregation = {
            'event_type': ['strict_accuracy', 'fuzzy_accuracy'],
            'event_sub_type': ['strict_accuracy', 'fuzzy_accuracy'],
            'state_of_victim': ['accuracy'],
            'victim_gender': ['accuracy']
        }

        for field, metric_types in categorical_fields_for_aggregation.items():
            for metric_type in metric_types:
                # Access deeply nested metrics
                scores = [m['categorical_metrics'].get(field, {}).get(metric_type, 0.0)
                          for m in all_metrics if field in m['categorical_metrics']]
                if scores:
                    # Initialize nested dict if it doesn't exist
                    if field not in aggregated_metrics['categorical_metrics']:
                        aggregated_metrics['categorical_metrics'][field] = {}
                    aggregated_metrics['categorical_metrics'][field][f'mean_{metric_type}'] = np.mean(scores)
                else:
                    if field not in aggregated_metrics['categorical_metrics']:
                        aggregated_metrics['categorical_metrics'][field] = {}
                    aggregated_metrics['categorical_metrics'][field][f'mean_{metric_type}'] = 0.0

        # Aggregate text similarity metrics for each comparable field
        # IMPORTANT: Included 'generated_event_sub_type_detail' here
        comparable_text_fields = [
            "specified_matter", "date_reference", "frequency", "repeat_incident",
            "identification", "injury_type", "victim_age", "victim_relation",
            "incident_location", "area", "suspect_description", "object_involved",
            "used_weapons", "offender_relation", "mode_of_threat", "need_ambulance",
            "children_involved", "date_of_birth", "generated_event_sub_type_detail"
        ]
        
        for field in comparable_text_fields:
            aggregated_metrics['text_similarity_metrics'][field] = {}
            for metric in ['jaccard', 'bleu', 'rouge_1', 'rouge_2', 'rouge_l', 'llm_similarity']:
                scores = [m['text_similarity_metrics'].get(field, {}).get(metric, 0.0)
                          for m in all_metrics if field in m['text_similarity_metrics']]
                if scores:
                    aggregated_metrics['text_similarity_metrics'][field][f'mean_{metric}'] = np.mean(scores)
                else:
                    aggregated_metrics['text_similarity_metrics'][field][f'mean_{metric}'] = 0.0

        # Aggregate processing metrics
        processing_times = [m['processing_metrics'].get('processing_time', 0.0) for m in all_metrics if m['processing_metrics'].get('processing_time') is not None]
        if processing_times:
            aggregated_metrics['processing_metrics'] = {
                'mean_processing_time': np.mean(processing_times),
                'std_processing_time': np.std(processing_times),
                'total_processing_time': np.sum(processing_times)
            }
        else:
            aggregated_metrics['processing_metrics'] = {    
                'mean_processing_time': 0.0,
                'std_processing_time': 0.0,
                'total_processing_time': 0.0
            }
                                        
        return aggregated_metrics