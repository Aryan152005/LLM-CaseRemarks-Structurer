import json
import os
from pathlib import Path
from typing import List
from datetime import datetime
import pandas as pd
from loguru import logger
from schema import ProcessedOutput
from text_processor import TextProcessor
from evaluator import Evaluator
from visualizer import Visualizer
from field_performance_visualizer import FieldPerformanceVisualizer

# =================== CONFIGURABLE PATHS ===================
DATA_FOLDER = "test_data/hindi/hindi_42_txt"
GROUND_TRUTH_PATH = "test_data/english/cleaned_new_40_eng.json"
OUTPUT_BASE_PATH = "results_hindi/mistral_results/predictions"
CSV_OUTPUT_PATH = "results_hindi/mistral_results/predictions.csv"
EVAL_RESULTS_BASE = "results_hindi/mistral_results/evaluation_results"
VISUALIZATION_DIR = "results_hindi/mistral_results/visualizations"
LOG_FILE = "file.log"

BATCH_SIZE = 10  # Save after every 10 files
MAX_FILE_SIZE_MB = 10  # Create new JSON file if this size is exceeded
# ===========================================================

logger.add(LOG_FILE, rotation="500 MB")

def load_ground_truth(file_path: str) -> list:
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading ground truth: {str(e)}")
        return []

def append_to_json_file(new_data: List[dict], file_base_path: str, max_file_size_mb: int = 10) -> str:
    os.makedirs(os.path.dirname(file_base_path), exist_ok=True)
    index = 0
    current_path = f"{file_base_path}.json"

    # Find a file that either doesn't exist or is not too large
    while os.path.exists(current_path) and os.path.getsize(current_path) > max_file_size_mb * 1024 * 1024:
        index += 1
        current_path = f"{file_base_path}_{index}.json"

    # Load existing content if any
    if os.path.exists(current_path):
        try:
            with open(current_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            existing_data = []
    else:
        existing_data = []

    # Append new data
    existing_data.extend(new_data)

    # Save updated file
    with open(current_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Appended {len(new_data)} predictions to {current_path}")
    return current_path

def save_csv(predictions: List[ProcessedOutput], path: str):
    try:
        df = pd.DataFrame([p.model_dump() for p in predictions])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
    except Exception as e:
        logger.error(f"Error saving CSV: {str(e)}")

def main():
    processor = TextProcessor()
    logger.info("TextProcessor initialized.")

    evaluator = Evaluator(ground_truth_file=GROUND_TRUTH_PATH)
    visualizer = Visualizer()
    # NEW: Initialize the FieldPerformanceVisualizer
    field_performance_visualizer = FieldPerformanceVisualizer(output_dir=os.path.join(VISUALIZATION_DIR, "field_performance_matrix"))
    
    predictions: List[ProcessedOutput] = []
    batch: List[ProcessedOutput] = []

    if not os.path.exists(DATA_FOLDER):
        logger.error(f"Data folder '{DATA_FOLDER}' not found.")
        return

    all_files = sorted([f for f in os.listdir(DATA_FOLDER) if f.endswith(".txt")])
    if not all_files:
        logger.warning("No .txt files found for processing.")
        return

    for idx, filename in enumerate(all_files, 1):
        file_path = os.path.join(DATA_FOLDER, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            result = processor.process_text(text)
            result.file_name = filename
            predictions.append(result)
            batch.append(result)
            logger.info(f"[{idx}/{len(all_files)}] Processed {filename}")

            if len(batch) >= BATCH_SIZE:
                serialized_batch = [b.model_dump() for b in batch]
                append_to_json_file(serialized_batch, OUTPUT_BASE_PATH, max_file_size_mb=MAX_FILE_SIZE_MB)
                batch.clear()

        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")

    # Save any remaining batch
    if batch:
        serialized_batch = [b.model_dump() for b in batch]
        final_json_path = append_to_json_file(serialized_batch, OUTPUT_BASE_PATH, max_file_size_mb=MAX_FILE_SIZE_MB)
    else:
        final_json_path = f"{OUTPUT_BASE_PATH}.json"

    # Save full CSV
    save_csv(predictions, CSV_OUTPUT_PATH)

    # Evaluate
    evaluation_results = evaluator.evaluate_predictions(predictions)

    # Save evaluation results
    eval_path = final_json_path.replace("predictions", "evaluation_results")
    evaluator.save_evaluation_results(evaluation_results, eval_path)

    # Save visualizations
    visualizer.create_visualizations(evaluation_results, VISUALIZATION_DIR)

    # NEW: Generate the field performance matrix plot
    field_performance_visualizer.generate_field_performance_matrix(evaluation_results, output_filename="field_performance_matrix_heatmap.png")

if __name__ == "__main__":
    main()
