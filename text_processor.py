import json
import time
from typing import Dict, Any, Optional
import requests
from loguru import logger
from schema import ProcessedOutput, FIELD_VALUE_SCHEMA, ALL_EVENT_SUB_TYPES, derive_event_type
import datetime
from difflib import get_close_matches
import re
from typing import List

# Keyword mapping for post-processing recovery
KEYWORD_EVENT_MAP = {
    # event_type: [keywords]
    'VIOLENT CRIME': ['assault', 'attack', 'beaten', 'violence', 'threat', 'robbery', 'murder', 'kidnapping', 'abuse', 'suicide'],
    'THEFT & BURGLARY': ['theft', 'stolen', 'burglary', 'snatching', 'rob', 'break-in', 'vehicle theft'],
    'TRAFFIC INCIDENTS': ['accident', 'hit and run', 'rash driving', 'traffic', 'vehicle', 'bike', 'car', 'run over'],
    'SOCIAL ISSUES': ['salary', 'wages', 'labour', 'family issue', 'dispute', 'neighbour', 'senior citizen', 'migrant'],
    'PUBLIC NUISANCE': ['nuisance', 'pollution', 'illegal', 'trespass', 'dumping', 'noise'],
    'FIRE & HAZARDS': ['fire', 'hazard', 'gas leak', 'electrical', 'building fire', 'landscape fire'],
    'MISSING PERSONS': ['missing', 'lost', 'child line', 'found person'],
    'NATURAL INCIDENTS': ['flood', 'earthquake', 'landslide', 'disaster', 'rainy'],
    'PUBLIC DISTURBANCE': ['scuffle', 'drunken', 'gambling', 'strike', 'nudity'],
    'RESCUE OPERATIONS': ['rescue', 'search', 'well rescue', 'water rescue', 'road crash rescue'],
    'MEDICAL EMERGENCIES': ['ambulance', 'heart attack', 'bleeding', 'collapsed', 'breathing', 'fire injury'],
}

# Few-shot examples for the prompt
FEW_SHOT_EXAMPLES = [
    {
        "input": "Hello Police control room, there has been a terrible accident at Dharampur Chowk, two children are injured by a car.",
        "output": {
            "event_type": "TRAFFIC INCIDENTS",
            "event_sub_type": "ACCIDENT",
            "state_of_victim": "Injured",
            "victim_gender": "not specified",
            "specified_matter": "accident at Dharampur Chowk, two children injured",
            "incident_location": "Dharampur Chowk",
            "area": "near Him Palace"
        }
    },
    {
        "input": "My bike has been stolen from Devbhoomi Bandkhedi, Roorkee, please help!",
        "output": {
            "event_type": "THEFT & BURGLARY",
            "event_sub_type": "VEHICLE THEFT",
            "state_of_victim": "not specified",
            "victim_gender": "not specified",
            "specified_matter": "bike stolen from Devbhoomi Bandkhedi, Roorkee",
            "incident_location": "Devbhoomi Bandkhedi, Roorkee, Uttarakhand",
            "area": "Roorkee"
        }
    },
    {
        "input": "There is a fire in the building at Main Street, people are trapped!",
        "output": {
            "event_type": "FIRE & HAZARDS",
            "event_sub_type": "BUILDING FIRE",
            "state_of_victim": "not specified",
            "victim_gender": "not specified",
            "specified_matter": "fire in the building at Main Street, people trapped",
            "incident_location": "Main Street",
            "area": None
        }
    }
]


def normalize_text(text: str) -> str:
    return ' '.join(text.lower().strip().split())

def get_few_shot_examples_str():
    lines = []
    for ex in FEW_SHOT_EXAMPLES:
        lines.append('---')
        lines.append('Input Transcript:')
        lines.append(ex['input'])
        lines.append('Output:')
        for k in ProcessedOutput.model_fields.keys():
            v = ex['output'].get(k, 'not specified')
            lines.append(f"{k}: {v if v is not None else 'not specified'}")
    return '\n'.join(lines)

class TextProcessor:
    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        self.ollama_base_url = ollama_base_url
        self.model_name = "llama3.1:8b"
        self.allowed_event_types = FIELD_VALUE_SCHEMA["event_type"]
        self.allowed_event_sub_types = ALL_EVENT_SUB_TYPES
        
    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call the Ollama LLM API"""
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
            return response.json()
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            raise

    def _create_extraction_prompt(self, text: str) -> str:
        safe_text = text.replace('"""', '\"\"\"')

        return f"""
    SYSTEM ROLE:
    You are an AI system assisting the Emergency Response Support System (ERSS) project under C-DAC, analyzing 112 emergency call transcripts. Your task is to classify and extract accurate structured metadata from unstructured call conversations between the caller and the emergency call taker.

    GENERAL OBJECTIVE:
    Your primary focus is to determine the correct `event_sub_type` from the transcript, and then derive the `event_type` from it. All other fields must be filled strictly based on what is explicitly mentioned in the transcript.

    YOUR RULES (Follow These STRICTLY):

    1. **event_sub_type (Most Important Field):**
    - Choose EXACTLY ONE sub-type from the pre-defined list.
    - DO NOT use `OTHERS` unless the incident is genuinely new and doesn't match any schema category, even loosely.
    - If forced to use `OTHERS`, follow this format: `OTHERS: <short new subtype label>` (1–3 words only).
    - Make every effort to match an existing sub-type before choosing `OTHERS`.

    2. **event_type:**
    - Do NOT generate this directly.
    - Instead, infer it automatically from the selected `event_sub_type` using the internal mapping.

    3. **Categorical Fields** (like `state_of_victim`, `victim_gender`, `need_ambulance`):
    - Only select from the exact allowed options.
    - If unknown or not mentioned, set as "not specified".

    4. **Text/Freeform Fields** (like `incident_location`, `specified_matter`, `suspect_description`):
    - If clearly present, extract the most accurate and specific text as stated in the transcript.
    - If unclear or absent, write "not specified".

    5. **Field-by-field logic:**
    - `specified_matter`: Write a detailed 1–2 line summary of the incident in natural language.
    - `injury_type`, `used_weapons`, `offender_relation`, etc. — extract only if clearly mentioned.
    - DO NOT hallucinate or assume facts not stated by the caller.

    FORMAT STRICTNESS:
    - OUTPUT MUST follow this format exactly: `field_name: value`
    - One field per line, in the order given in the schema.
    - Do NOT include extra commentary, explanations, or markdown.

    ---

    SCHEMA DEFINITIONS:

    event_sub_type: One from the following predefined list (use `OTHERS: <label>` ONLY if none match even loosely):
    {', '.join(ALL_EVENT_SUB_TYPES)}

    event_type: Automatically derived internally based on event_sub_type

    state_of_victim: One of {FIELD_VALUE_SCHEMA['state_of_victim']}

    victim_gender: One of {FIELD_VALUE_SCHEMA['victim_gender']}

    (Other fields and options remain as described in the schema. Use "not specified" when not clear.)

    ---

    FEW-SHOT EXAMPLES:
    {get_few_shot_examples_str()}

    ---

    INPUT TRANSCRIPT (verbatim):
    \"\"\"{safe_text}\"\"\"

    ---
    YOUR RESPONSE (STRICTLY in field: value format):
    """


    def _parse_llm_field_value_output(self, llm_output: str) -> dict:
        result = {}
        allowed_fields = list(ProcessedOutput.__annotations__.keys())

        for line in llm_output.splitlines():
            if ':' not in line:
                continue

            field, value = line.split(':', 1)
            field = field.strip()
            value = value.strip()

            if value.lower() in ["null", "none", "", "not_defined"]:
                value = "not specified"

            if field not in allowed_fields:
                logger.warning(f"Unknown field returned by LLM: {field}")
                continue

            result[field] = value

        return result




    def process_text(self, text: str, file_name: Optional[str] = None) -> ProcessedOutput:
        """Process text and extract structured information"""
        start_time = time.time()
        
        try:
            # Create prompt
            prompt = self._create_extraction_prompt(text)
            
            # Call LLM
            response = self._call_llm(prompt)
            response_text = response.get('response', '')
            
            # Parse field: value output
            extracted_data = self._parse_llm_field_value_output(response_text)
            
            # event_type is derived from event_sub_type
            sub_type = extracted_data.get("event_sub_type", "OTHERS")
            # Handle OTHERS: <label>
            if sub_type.startswith("OTHERS:"):
                extracted_data["event_type"] = "OTHERS"
            else:
                extracted_data["event_type"] = derive_event_type(sub_type)
            
            # Fill all required fields
            for field in extracted_data:
                if field not in ProcessedOutput.model_fields:
                    logger.warning(f"Unknown field returned by LLM: {field}")

            
            # Add processing metadata
            extracted_data.update({
                "processing_time": time.time() - start_time,
                "file_name": file_name,
                "file_text": text,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            # Create ProcessedOutput object
            output = ProcessedOutput(**extracted_data)
            
            return output
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            raise

    
    def process_batch(self, texts: List[str], file_names: Optional[List[str]] = None) -> List[ProcessedOutput]:

        """Process a batch of texts"""
        if file_names is None:
            file_names = [None] * len(texts)
            
        results = []
        for text, file_name in zip(texts, file_names):
            try:
                result = self.process_text(text, file_name)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process text: {str(e)}")
                continue
                
        return results 