# Emergency Call Text Processing System

A Python-based system for extracting structured information from emergency call texts using LLM-based processing and comprehensive evaluation metrics.

## Features

- Structured information extraction from emergency call texts
- Support for both English and Hinglish text inputs
- Comprehensive evaluation metrics including:
  - Categorical accuracy
  - Text similarity (Cosine, ROUGE, BLEU)
  - Keyword similarity (Jaccard)
  - Processing time analysis
- Detailed visualizations of all metrics
- Robust error handling and logging
- Output in both JSON and CSV formats

## Prerequisites

- Python 3.8+
- Ollama server running locally with llama2:8b model
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure Ollama server is running with llama2:8b model:
```bash
ollama run llama2:8b
```

## Project Structure

```
.
├── schema.py              # Schema definitions and data models
├── text_processor.py      # Text processing and LLM integration
├── evaluator.py          # Evaluation metrics calculation
├── visualizer.py         # Metrics visualization
├── main.py              # Main orchestration script
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## Usage

1. Prepare your schema file (schema.json) with event type mappings:
```json
{
    "event_type_mappings": {
        "fire": ["building_fire", "vehicle_fire", "wildfire"],
        "medical": ["cardiac", "trauma", "respiratory"],
        "crime": ["robbery", "assault", "burglary"]
    }
}
```

2. Prepare your ground truth data in ground_truth/annotations/annotations.json:
```json
[
    {
        "file_name": "call_001.txt",
        "file_text": "Emergency call reporting fire in building",
        "event_type": "fire",
        "event_sub_type": "building_fire",
        "location": "123 Main St",
        "severity": "high",
        "description": "Multiple floors affected",
        "keywords": ["fire", "building", "emergency"]
    }
]
```

3. Run the system:
```bash
python main.py
```

## Output

The system generates the following outputs:

1. Predictions:
   - output/predictions.json
   - output/predictions.csv

2. Visualizations (in visualizations/):
   - categorical_metrics.png
   - text_similarity_metrics.png
   - processing_time.png
   - rouge_scores.png
   - keyword_similarity.png
   - metrics.json

3. Logs:
   - processing.log

## Evaluation Metrics

The system calculates and visualizes:

1. Categorical Metrics:
   - Accuracy for event_type, event_sub_type, and severity

2. Text Similarity Metrics:
   - Cosine similarity for description and location
   - ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
   - BLEU score

3. Set Similarity Metrics:
   - Jaccard similarity for keywords

4. Processing Metrics:
   - Mean processing time
   - Standard deviation
   - Total processing time

## Error Handling

The system includes comprehensive error handling for:
- File I/O operations
- LLM API calls
- Data parsing and validation
- Edge cases in text processing

All errors are logged with detailed information in processing.log.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 