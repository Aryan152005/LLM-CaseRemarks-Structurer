import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from loguru import logger
from schema import ProcessedOutput, FIELD_VALUE_SCHEMA
from text_processor import TextProcessor
from evaluator import Evaluator
from visualizer import Visualizer
import os
from datetime import datetime

# Configure logger
logger.add("file.log", rotation="500 MB")

def load_ground_truth(file_path: str) -> list:
    """Load ground truth data from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading ground truth: {str(e)}")
        return []

def save_predictions(predictions: list, output_dir: str):
    """Save predictions to JSON and CSV files"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert ProcessedOutput objects to dictionaries and handle datetime serialization
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return str(obj)
            
        json_data = [pred.model_dump() for pred in predictions]
        
        # Save as JSON
        json_path = os.path.join(output_dir, "predictions.json")
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=serialize_datetime)
        logger.info(f"Predictions saved to {json_path}")
        
        # Save as CSV
        csv_path = os.path.join(output_dir, "predictions.csv")
        df = pd.DataFrame(json_data)
        df.to_csv(csv_path, index=False)
        logger.info(f"Predictions saved to {csv_path}")
        
    except Exception as e:
        logger.error(f"Error saving predictions: {str(e)}")

def main():
    # Initialize components
    # TextProcessor is initialized without arguments, using its default Ollama URL
    processor = TextProcessor()
    logger.info("TextProcessor initialized.")

    evaluator = Evaluator(ground_truth_file="data/ground_truth_annotations_fixed.json")
    visualizer = Visualizer()
    
    # Process all text files in the data directory
    data_folder = "data/ground_truth_hin" # This path should exist and contain your .txt files
    predictions = []
    
    if not os.path.exists(data_folder):
        logger.error(f"Data folder '{data_folder}' not found. Please ensure it exists and contains your input text files.")
        return

    # Check if data_folder is empty or contains no .txt files
    txt_files_found = False
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            txt_files_found = True
            file_path = os.path.join(data_folder, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f: # Added encoding for robustness
                    text = f.read()
                
                # Process the text
                result = processor.process_text(text)
                result.file_name = filename # Assign filename after processing for output consistency
                predictions.append(result)
                logger.info(f"Processed {filename}")
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
    
    if not txt_files_found:
        logger.warning(f"No .txt files found in '{data_folder}'. No predictions will be generated.")
        return

    # Save predictions
    save_predictions(predictions, "hindi_output_10_hermes")
    
    # Evaluate predictions
    evaluation_results = evaluator.evaluate_predictions(predictions)
    evaluator.save_evaluation_results(evaluation_results, "hindi_output_10_hermes/evaluation_results.json")
    
    # Create visualizations
    visualizer.create_visualizations(evaluation_results, "hindi_output_10_hermes/visualizations")

if __name__ == "__main__":
    main()