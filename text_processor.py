import json
import time
from typing import Dict, Any, Optional, List
import requests
from loguru import logger
# Assuming schema.py is correct and contains these imports:
from schema import ProcessedOutput, FIELD_VALUE_SCHEMA, ALL_EVENT_SUB_TYPES, derive_event_type
import datetime
from difflib import get_close_matches
import re

# Keyword mapping for post-processing recovery (kept for context, not directly used in the fix)
KEYWORD_EVENT_MAP = {
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
# IMPORTANT: Updated examples to reflect the new `generated_event_sub_type_detail` field
FEW_SHOT_EXAMPLES = [
    {
    "file_name": "audio6_truth",
    "event_info_text": "Hello Hello Haji Namaskar Madam do you need help, what is the police number? Yes sir Madam, what about me, my bike might get stolen, Madam, where are you calling from, Roorkee Madam, Uttarakhand, where in Roorkee, Devbhoomi Bandkhedi, yes, Devbhoomi Bandkhedi, which police station would be there, I am telling you, it is Ganganehar, which bike is it, the bike is Honda Shine, which colour, red colour, what is your name? My name is Luv Kumar, tell me the vehicle number, UP twenty UP twenty B C B C, yes sir, B for ball, C for cat, yes sir, eighty one ninety eight eight one nine eight Haji, yes sir, it's okay, I'm telling you.",
    "event_type": "THEFT & BURGLARY",
    "event_sub_type": "VEHICLE THEFT",
    "state_of_victim": "not specified",
    "victim_gender": "not specified",
    "specified_matter": "my bike might get stolen",
    "date_reference": "not specified",
    "frequency": "not specified",
    "repeat_incident": "not specified",
    "identification": "Luv Kumar",
    "injury_type": "not specified",
    "victim_age": "not specified",
    "victim_relation": "not specified",
    "incident_location": "Devbhoomi Bandkhedi, Roorkee, Uttarakhand",
    "area": "Roorkee",
    "suspect_description": "not specified",
    "object_involved": "Honda Shine bike (red colour)",
    "used_weapons": "not specified",
    "offender_relation": "not specified",
    "mode_of_threat": "not specified",
    "need_ambulance": "not specified",
    "children_involved": "not specified",
    "generated_event_sub_type_detail": "not specified"
    },
    {
    "file_name": "audio5_truth",
    "event_info_text": "Hello, do you need any help, Police ji, Madam, I am saying that we have a cousin sister, she has been divorced, so her husband has beaten her and thrown her out, he is at my house, he keeps coming to my house again and again, she is our cousin sister, she has been divorced, her husband drinks and eats, so he keeps coming here, even after leaving me, she came to my house, my cousin sister keeps coming here again and again to beat her and there is a lot of blood on her hands too, tell me the address, tell me your address, this is Metro Hospital, Metro Hospital, Akash Vikas Colony, Akash Vikas Colony, Awas Vikas, ji, your name, my name, Shazia Parveen, what would be the police station, that place, Jaspur, you are talking from Udham Udham Singh Nagar, district Udham Singh Nagar, ji, yes, Uttarakhand, Jaspur, okay, I am telling you okay.",
    "event_type": "VIOLENT CRIME",
    "event_sub_type": "DOMESTIC VIOLENCE",
    "state_of_victim": "Distressed",
    "victim_gender": "not specified",
    "specified_matter": "her husband has beaten her and thrown her out, he keeps coming to my house again and again",
    "date_reference": "not specified",
    "frequency": "not specified",
    "repeat_incident": "yes (implied from \"he keeps coming here\")",
    "identification": "Shazia Parveen",
    "injury_type": "bleeding",
    "victim_age": "not specified",
    "victim_relation": "cousin sister",
    "incident_location": "Akash Vikas Colony, Awas Vikas",
    "area": "Jaspur, Udham Singh Nagar",
    "suspect_description": "not specified",
    "object_involved": "not specified",
    "used_weapons": "hands (implied from \"there is a lot of blood on her hands too\")",
    "offender_relation": "husband of the victim (implied)",
    "mode_of_threat": "physical violence",
    "need_ambulance": "yes (implied from \"a lot of blood on her hands too\")",
    "children_involved": "not specified",
    "generated_event_sub_type_detail": "not specified"
    },
    {
    "file_name": "audio10_truth",
    "event_info_text": "Hello Police Control Room Hello Sir, please tell me Yes Ma'am, I have faced a problem what has happened here in Lal Thappad area what happened is that I had bought a car just fifteen minutes ago so what happened is that I had bought a car, okay just fifteen minutes ago so he said that everything is okay with the car, I had bought a car, made the payment etc. and took the car fifteen minutes ago and now the car developed a fault on the way and when I went to give it back to him, he is not picking up the phone or taking it out, which car have you bought, BMW Seven series, second hand, first second, second hand, so here is that doctor, he has a clinic, he is not opening the clinic here right now and he is not picking up the phone either, so you guys can please come and help me because we are not local here, where in Lal Thappad, in Lal Thappad, Surprise Hotel, opposite to Surprise Hotel, Sunrise Sunrise Sunrise Hotel, what is your name, my name is Abhilash Kumar, so you guys will send someone from the police because you are telling me, telling me about this I am ok",
    "event_type": "OTHERS",
    "event_sub_type": "OTHERS",
    "state_of_victim": "not specified",
    "victim_gender": "not specified",
    "specified_matter": "I had bought a car just fifteen minutes ago so what happened is that I had bought a car, okay just fifteen minutes ago so he said that everything is okay with the car, I had bought a car, made the payment etc. and took the car fifteen minutes ago and now the car developed a fault on the way and when I went to give it back to him, he is not picking up the phone or taking it out...",
    "date_reference": "not specified",
    "frequency": "not specified",
    "repeat_incident": "not specified",
    "identification": "Abhilash Kumar",
    "injury_type": "not specified",
    "victim_age": "not specified",
    "victim_relation": "not specified",
    "incident_location": "Lal Thappad area, opposite to Surprise Hotel, Sunrise Hotel",
    "area": "Lal Thappad",
    "suspect_description": "not specified",
    "object_involved": "car (BMW Seven series)",
    "used_weapons": "not specified",
    "offender_relation": "not specified",
    "mode_of_threat": "not specified",
    "need_ambulance": "not specified",
    "children_involved": "not specified",
    "generated_event_sub_type_detail": "VEHICLE FRAUD"
    },
]

def normalize_text(text: str) -> str:
    return ' '.join(text.lower().strip().split())

def get_few_shot_examples_str():
    lines = []
    for ex in FEW_SHOT_EXAMPLES:
        lines.append('---')
        lines.append('Input Transcript:')
        # Changed 'input' to 'event_info_text'
        lines.append(f'"""{ex["event_info_text"]}"""')
        lines.append('Output:')
        # Iterate over ProcessedOutput's fields to ensure order and completeness
        for k in ProcessedOutput.model_fields.keys():
            # Changed ex['output'].get(k) to ex.get(k)
            v = ex.get(k)
            # Ensure 'None' maps to "not specified" for LLM output, as per schema's default expectations
            if v is None or (isinstance(v, str) and v.lower() in ["null", "none"]):
                # Apply specific casing for "not specified" based on field
                if k == 'state_of_victim':
                    lines.append(f"{k}: Not specified")
                elif k in ['victim_gender', 'repeat_incident', 'need_ambulance', 'children_involved', 'generated_event_sub_type_detail', 'area', 'date_of_birth', 'contact_number']: # Added date_of_birth, contact_number for explicit handling
                    lines.append(f"{k}: not specified")
                else:
                    lines.append(f"{k}: not specified") # Default for other fields
            else:
                lines.append(f"{k}: {v}")
    return '\n'.join(lines)


class TextProcessor:
    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        self.ollama_base_url = ollama_base_url
        self.model_name = "spooknik/hermes-2-pro-mistral-7b:latest" # Or whatever model you are using
        self.allowed_event_types = FIELD_VALUE_SCHEMA["event_type"]
        self.allowed_event_sub_types = ALL_EVENT_SUB_TYPES
        
        # Create a mapping for common casing issues for literal fields
        self.literal_field_corrections = {
            "state_of_victim": {val.lower(): val for val in FIELD_VALUE_SCHEMA["state_of_victim"]},
            "victim_gender": {val.lower(): val for val in FIELD_VALUE_SCHEMA["victim_gender"]},
            "repeat_incident": {val.lower(): val for val in ["yes", "no", "not specified"]},
            "need_ambulance": {val.lower(): val for val in ["yes", "no", "not specified"]},
            "children_involved": {val.lower(): val for val in ["yes", "no", "not specified"]},
            # Add other literal fields from FIELD_VALUE_SCHEMA if they exist and need similar correction
        }

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call the Ollama LLM API"""
        try:
            newline_char = '\n'
            logger.info(f"TextProcessor calling LLM with prompt (first 200 chars): {prompt[:200].replace(newline_char, ' ')}...")
            
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1, # Keep temperature low for structured extraction
                        "num_ctx": 4096 # Adjust context window if your prompts are very long
                    }
                }
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as ce:
            logger.error(f"Connection Error to Ollama: {ce}. Is Ollama server running at {self.ollama_base_url}?")
            raise # Re-raise the exception to propagate it
        except requests.exceptions.RequestException as re:
            logger.error(f"Request Error to Ollama: {re}")
            raise
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            raise

    def _create_extraction_prompt(self, text: str) -> str:
        safe_text = text.replace('"""', '\"\"\"')

        return f"""
SYSTEM ROLE:
You are an AI system assisting the Emergency Response Support System (ERSS) project under C-DAC, analyzing 112 emergency call transcripts. Your task is to classify and extract accurate structured metadata from unstructured call conversations between the caller and the emergency call taker.

GENERAL OBJECTIVE:
Your primary focus is to determine the correct `event_sub_type` from the transcript. Then, ensure all other fields are filled strictly based on what is explicitly mentioned.

YOUR RULES (Follow These STRICTLY):

1.  **PRIORITY FIELD: event_sub_type (CRITICAL)**
    * **MUST** choose EXACTLY ONE sub-type from the predefined list provided below.
    * **NEVER** generate a sub-type that is not in the list.
    * **ONLY** if the incident is genuinely and uniquely unclassifiable into ANY existing specific category (even loosely), then set `event_sub_type` to `OTHERS`. This should be a rare exception.
    * If you set `event_sub_type` to `OTHERS`, then you **MUST** provide a brief, specific, and descriptive label (1-3 words, e.g., "VEHICLE SNATCHING", "CHEMICAL SPILL") for this new type in the `generated_event_sub_type_detail` field.
    * If you successfully match an existing `event_sub_type`, then `generated_event_sub_type_detail` **MUST** be "not specified".
    * Be flexible with phrasing: "bike stolen" should map to "VEHICLE THEFT"; "fire in house" to "BUILDING FIRE". Consider synonyms and related concepts.

2.  **event_type:**
    * Do NOT generate this field.
    * It will be inferred automatically by the system based on the `event_sub_type` you provide.

3.  **Categorical Fields** (like `state_of_victim`, `victim_gender`, `need_ambulance`, `repeat_incident`, `children_involved`):
    * Only select from the **exact** allowed options.
    * If unknown or not mentioned, set as "not specified".
    * Be strict about casing: "Not specified" for `state_of_victim`, "not specified" for `victim_gender`, `repeat_incident`, `need_ambulance`, `children_involved`, and `generated_event_sub_type_detail`.

4.  **Text/Freeform Fields** (like `incident_location`, `specified_matter`, `suspect_description`, `area`, `contact_number`):
    * If clearly present, extract the most accurate and specific text **as stated in the transcript**.
    * If unclear or absent, write "not specified".

5.  **Field-by-field logic:**
    * `specified_matter`: Write a detailed 1–2 line summary of the incident in natural language from the transcript.
    * `incident_location`, `area`: Extract location details if mentioned. `area` should be a broader geographical region.
    * `contact_number`: Extract if a phone number is provided.
    * `injury_type`, `used_weapons`, `offender_relation`, etc. — extract only if clearly mentioned.
    * DO NOT hallucinate or assume facts not stated by the caller.

FORMAT STRICTNESS:
- OUTPUT MUST follow this format exactly: `field_name: value`
- One field per line, in the order given in the schema.
- Do NOT include any introductory or concluding remarks, explanations, or markdown fences (like ```json). Just the field: value pairs.

---

SCHEMA DEFINITIONS:

event_sub_type: One from the following predefined list (Choose one of these, or 'OTHERS' only if absolutely necessary):
{', '.join(ALL_EVENT_SUB_TYPES)}

event_type: Automatically derived internally based on event_sub_type (DO NOT GENERATE)

state_of_victim: One of {FIELD_VALUE_SCHEMA['state_of_victim']}

victim_gender: One of {FIELD_VALUE_SCHEMA['victim_gender']}

generated_event_sub_type_detail: (Optional) A specific label for 'OTHERS' event_sub_type (e.g., VEHICLE SNATCHING, CHEMICAL SPILL). Set to "not specified" if event_sub_type is NOT 'OTHERS'.

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
        model_fields = ProcessedOutput.model_fields.keys()
        
        default_not_specified_mapping = {
            "state_of_victim": "Not specified",
            "victim_gender": "not specified",
            "repeat_incident": "not specified",
            "need_ambulance": "not specified",
            "children_involved": "not specified",
            "generated_event_sub_type_detail": "not specified", # Default for new field
            "area": "not specified", # Default for area
            "date_of_birth": "not specified",
            "contact_number": "not specified"
        }

        for line in llm_output.splitlines():
            if not line.strip() or ':' not in line:
                continue
            
            field, value = line.split(':', 1)
            field = field.strip()
            value = value.strip()

            # Normalize common "None", empty string, or invalid outputs to "not specified"
            if value.lower() in ["null", "none", "", "not_defined", "n/a"]:
                value = "not specified" 

            if field not in model_fields:
                logger.warning(f"Unknown field returned by LLM: {field}. Skipping.")
                continue

            # --- Special Handling for event_sub_type and generated_event_sub_type_detail ---
            # Process these so they are in 'result' before the final correction step
            if field == "event_sub_type":
                normalized_llm_value = value.upper()
                # Store it as is for now, will be corrected in post-processing if needed
                result[field] = normalized_llm_value
                # If LLM explicitly outputted OTHERS with a detail, handle that early
                match = re.match(r"OTHERS:\s*(.+)", normalized_llm_value)
                if match:
                    result["event_sub_type"] = "OTHERS"
                    detail = match.group(1).strip()
                    if detail:
                        result["generated_event_sub_type_detail"] = detail
                    else:
                        result["generated_event_sub_type_detail"] = "not specified"
                # else: a direct match or fuzzy match will be handled in post-processing
            
            elif field == "generated_event_sub_type_detail":
                result[field] = value if value.lower() not in ["null", "none", ""] else "not specified"
            
            # --- General Literal Field Correction (case sensitivity & fuzzy matching) ---
            elif field in self.literal_field_corrections:
                corrected_value = self.literal_field_corrections[field].get(value.lower())
                if corrected_value: # If a direct lowercase match found, use its correct casing
                    result[field] = corrected_value
                else: # If not found, try get_close_matches for more flexibility
                    allowed_values_for_field = list(self.literal_field_corrections[field].keys())
                    matches = get_close_matches(value.lower(), allowed_values_for_field, n=1, cutoff=0.8) # Adjust cutoff
                    if matches:
                        result[field] = self.literal_field_corrections[field][matches[0]]
                        logger.info(f"Corrected '{field}' value '{value.lower()}' to closest match '{result[field]}'.")
                    else:
                        # If still no match, and it's a "not specified" type, use the default from mapping
                        result[field] = default_not_specified_mapping.get(field, "not specified") 
                        logger.warning(f"LLM returned invalid literal for '{field}': '{value}'. Defaulting to '{result[field]}'.")
            
            else: # For other fields (freeform text)
                result[field] = value

        # --- NEW POST-PROCESSING STEP FOR event_sub_type AND generated_event_sub_type_detail ---
        # This runs after all fields from LLM output have been initially parsed.
        current_sub_type = result.get("event_sub_type", "").upper()
        current_generated_detail = result.get("generated_event_sub_type_detail", "").upper()

        # Scenario 1: LLM outputted OTHERS for event_sub_type but provided a specific detail that IS a known sub-type
        if current_sub_type == "OTHERS" and current_generated_detail and current_generated_detail != "NOT SPECIFIED":
            # Try to directly match the generated detail to ALL_EVENT_SUB_TYPES
            if current_generated_detail in ALL_EVENT_SUB_TYPES:
                result["event_sub_type"] = current_generated_detail # Promote the detail to the main sub_type
                result["generated_event_sub_type_detail"] = "not specified" # Clear the detail field
                logger.info(f"Promoted event_sub_type from 'OTHERS' to '{current_generated_detail}' based on 'generated_event_sub_type_detail'.")
            else:
                # If not a direct match, try fuzzy matching the generated detail
                close_matches_from_detail = get_close_matches(current_generated_detail, ALL_EVENT_SUB_TYPES, n=1, cutoff=0.65)
                if close_matches_from_detail:
                    corrected_sub_type = close_matches_from_detail[0]
                    result["event_sub_type"] = corrected_sub_type
                    result["generated_event_sub_type_detail"] = "not specified"
                    logger.info(f"Promoted event_sub_type from 'OTHERS' to '{corrected_sub_type}' (fuzzy match) based on 'generated_event_sub_type_detail'.")
                # Else: it was OTHERS and the detail is genuinely not a known sub-type, so keep it as is.
        
        # Scenario 2: LLM outputted a non-OTHERS event_sub_type, but it's not a valid one
        elif current_sub_type not in ALL_EVENT_SUB_TYPES:
            # Re-run fuzzy match for the primary event_sub_type field if it's currently invalid
            close_matches = get_close_matches(current_sub_type, ALL_EVENT_SUB_TYPES, n=1, cutoff=0.65)
            if close_matches:
                result["event_sub_type"] = close_matches[0]
                logger.info(f"Corrected invalid event_sub_type '{current_sub_type}' to closest match '{close_matches[0]}'.")
                if "generated_event_sub_type_detail" in result and result["generated_event_sub_type_detail"].lower() != "not specified":
                    logger.warning(f"Cleared 'generated_event_sub_type_detail' as event_sub_type is now specific.")
                    result["generated_event_sub_type_detail"] = "not specified"
            else:
                # If still no match after fuzzy, force to "OTHERS"
                original_invalid_sub_type = result.get("event_sub_type", "not specified")
                result["event_sub_type"] = "OTHERS"
                # If generated_event_sub_type_detail is not already set and it's not a generic "not specified", use original_invalid_sub_type
                if result.get("generated_event_sub_type_detail", "").lower() == "not specified" and original_invalid_sub_type.lower() != "not specified":
                    result["generated_event_sub_type_detail"] = original_invalid_sub_type
                logger.warning(f"LLM returned persistently invalid event_sub_type '{original_invalid_sub_type}'. Forcing to 'OTHERS' and storing original in 'generated_event_sub_type_detail'.")
        
        # Scenario 3: If event_sub_type is now a specific type, ensure generated_event_sub_type_detail is 'not specified'
        if result.get("event_sub_type", "").upper() != "OTHERS":
            result["generated_event_sub_type_detail"] = "not specified"

        # --- Final check for missing required fields and default filling ---
        for field_name, field_info in ProcessedOutput.model_fields.items():
            if field_name not in result:
                if field_name == "file_name":
                    continue # Handled during ProcessedOutput instantiation
                elif field_name == "event_type": 
                    # This will be derived AFTER event_sub_type is finalized, so temporary placeholder is fine
                    result[field_name] = "OTHERS" 
                else:
                    # For other required/optional fields not returned by LLM, default
                    result[field_name] = default_not_specified_mapping.get(field_name, "not specified")

        return result


    def process_text(self, text: str, file_name: Optional[str] = None) -> ProcessedOutput:
        """Process text and extract structured information"""
        start_time = time.time()
        
        try:
            # Create prompt
            prompt = self._create_extraction_prompt(text)
            
            # Call LLM
            # Ensure _call_llm returns {"response": "..."} as expected
            response = self._call_llm(prompt)
            response_text = response.get('response', '')
            
            # Parse field: value output
            extracted_data = self._parse_llm_field_value_output(response_text)
            
            # --- Assign file_name before Pydantic validation ---
            extracted_data["file_name"] = file_name if file_name is not None else "unspecified_file"

            # --- Derive event_type AFTER event_sub_type has been processed/corrected ---
            # This is crucial for correct categorization
            sub_type = extracted_data.get("event_sub_type", "OTHERS")
            extracted_data["event_type"] = derive_event_type(sub_type)
            
            # Add processing metadata
            extracted_data.update({
                "processing_time": time.time() - start_time,
                "file_text": text,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            # Create ProcessedOutput object
            output = ProcessedOutput(**extracted_data)
            
            return output
            
        except Exception as e:
            logger.error(f"Error processing text for file '{file_name or 'unknown'}': {e}")
            raise # Re-raise to let main.py handle individual file failures

    def process_batch(self, texts: List[str], file_names: Optional[List[str]] = None) -> List[ProcessedOutput]:
        """Process a batch of texts"""
        if file_names is None:
            file_names = [f"unspecified_file_{i}.txt" for i in range(len(texts))]
            
        results = []
        for text, file_name in zip(texts, file_names):
            try:
                result = self.process_text(text, file_name)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process text for file '{file_name}': {str(e)}")
                continue # Continue to next file if an error occurs.
                
        return results