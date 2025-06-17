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
    processor = TextProcessor()
    evaluator = Evaluator(ground_truth_file="data/ground_truth_annotations_fixed.json")
    visualizer = Visualizer()
    
    # Process all text files in the data directory
    data_folder = "data/input_eng"
    predictions = []
    
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_folder, filename)
            try:
                with open(file_path, 'r') as f:
                    text = f.read()
                
                # Process the text
                result = processor.process_text(text)
                result.file_name = filename
                predictions.append(result)
                logger.info(f"Processed {filename}")
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
    
    # Save predictions
    save_predictions(predictions, "output")
    
    # Evaluate predictions
    evaluation_results = evaluator.evaluate_predictions(predictions)
    evaluator.save_evaluation_results(evaluation_results, "output/evaluation_results.json")
    
    # Create visualizations
    visualizer.create_visualizations(evaluation_results, "output/visualizations")

if __name__ == "__main__":
    main() 