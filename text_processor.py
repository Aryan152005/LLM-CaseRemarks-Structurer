import json
import time
from typing import Dict, Any, Optional, List
import requests
from loguru import logger
from schema import ProcessedOutput, FIELD_VALUE_SCHEMA, ALL_EVENT_SUB_TYPES, derive_event_type
import datetime
from difflib import get_close_matches
import re
import os
from dotenv import load_dotenv
load_dotenv()

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
    "repeat_incident": "yes",
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
    "need_ambulance": "yes",
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
        lines.append(f'"""{ex["event_info_text"]}"""')
        lines.append('Output:')
        # Iterate over ProcessedOutput's fields to ensure order and completeness
        for k in ProcessedOutput.model_fields.keys():
            v = ex.get(k)
            # Ensure 'None' maps to "not specified" for LLM output, as per schema's default expectations
            if v is None or (isinstance(v, str) and v.lower() in ["null", "none"]):
                # Apply specific casing for "not specified" based on field
                if k == 'state_of_victim':
                    lines.append(f"{k}: not specified")
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

        # --- Logic for seamless model selection starts here ---
        # Get LLM provider preference from environment variable, default to 'none' to force explicit selection
        self.llm_provider = os.getenv("LLM_PROVIDER", "none").lower() 

        # Initialize model-specific parameters based on the chosen provider
        if self.llm_provider == "gemini":
            self.model_name = "gemini-2.0-flash"
            self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                logger.warning(
                    "LLM_PROVIDER is set to 'gemini', but GEMINI_API_KEY is not found in environment variables. "
                    "Gemini API calls will likely fail. Please set GEMINI_API_KEY."
                )
            self.hf_api_url = None # Not applicable for Gemini
            print(f"TextProcessor initialized: Using LLM Provider: Gemini ({self.model_name})") # Print statement
        elif self.llm_provider == "gemma-hf":
            self.model_name = "google/gemma-3-12b-it"
            self.api_key = os.getenv("HUGGINGFACE_API_KEY") 
            if not self.api_key:
                logger.warning(
                    "LLM_PROVIDER is set to 'gemma-hf', but HUGGINGFACE_API_KEY is not found in environment variables. "
                    "Hugging Face Inference API calls will likely fail. Please set HUGGINGFACE_API_KEY."
                )
            self.hf_api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
            print(f"TextProcessor initialized: Using LLM Provider: Hugging Face ({self.model_name})") # Print statement
        elif self.llm_provider == "ollama": 
            self.model_name = "mistral:7b" # Default Ollama model
            self.api_key = None # Ollama typically doesn't use an API key in this manner
            self.hf_api_url = None # Not applicable for Ollama
            print(f"TextProcessor initialized: Using LLM Provider: Ollama ({self.model_name})") # Print statement
        else: 
            self.model_name = None
            self.api_key = None
            self.hf_api_url = None
            error_message = (
                f"Invalid LLM_PROVIDER specified: '{self.llm_provider}'. "
                "Supported providers are 'gemini', 'gemma-hf', or 'ollama'. "
                "Please set the LLM_PROVIDER environment variable correctly."
            )
            logger.error(error_message)
            print(f"TextProcessor initialization failed: {error_message}") # Print statement for error
            raise ValueError("Error: Invalid LLM_PROVIDER specified. Cannot initialize TextProcessor.")
        # --- Logic for seamless model selection ends here ---

        self.allowed_event_types = FIELD_VALUE_SCHEMA["event_type"]
        self.allowed_event_sub_types = ALL_EVENT_SUB_TYPES

        # Create a mapping for common casing issues for literal fields
        self.literal_field_corrections = {
            "state_of_victim": {val.lower(): val for val in FIELD_VALUE_SCHEMA["state_of_victim"]},
            "victim_gender": {val.lower(): val for val in FIELD_VALUE_SCHEMA["victim_gender"]},
            "repeat_incident": {v.lower(): v for v in ["yes", "no", "not specified", "not applicable"]},
            "need_ambulance": {v.lower(): v for v in ["yes", "no", "not specified", "not applicable"]},
            "children_involved": {v.lower(): v for v in ["yes", "no", "not specified", "not applicable"]},
        }

    # This is now the *single* _call_llm method that handles all providers
    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """
        Calls the appropriate LLM API (Ollama, Gemini, or Hugging Face Gemma) based on the
        'LLM_PROVIDER' environment variable set during initialization.
        """
        newline_char = '\n'
        
        if self.model_name is None:
            logger.error("LLM model not initialized due to an invalid LLM_PROVIDER. Cannot call LLM.")
            return {"response": "Error: LLM model not configured properly."}

        if self.llm_provider == "gemini":
            # --- Gemini 2.0 Flash LLM logic ---
            if not self.api_key:
                logger.error("Cannot call Gemini LLM: GEMINI_API_KEY is not set.")
                raise ValueError("GEMINI_API_KEY is required for Gemini LLM calls.")

            max_retries = 5
            retry_delay_seconds = 60

            for attempt in range(max_retries):
                try:
                    logger.info(f"TextProcessor calling Gemini LLM with prompt (first 200 chars): {prompt[:200].replace(newline_char, ' ')}... (Attempt {attempt + 1}/{max_retries})")

                    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"

                    payload = {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [{"text": prompt}]
                            }
                        ],
                        "generationConfig": {
                            "temperature": 0.1,
                            "maxOutputTokens": 2048
                        }
                    }

                    response = requests.post(
                        api_url,
                        headers={'Content-Type': 'application/json'},
                        json=payload
                    )
                    response.raise_for_status()

                    result = response.json()

                    if result.get("candidates") and result["candidates"][0].get("content") and \
                            result["candidates"][0]["content"].get("parts") and result["candidates"][0]["content"]["parts"][0].get("text"):
                        generated_text = result["candidates"][0]["content"]["parts"][0]["text"]
                        logger.info("Successfully received response from Gemini API.")
                        return {"response": generated_text}
                    else:
                        logger.warning(f"Unexpected response structure from Gemini API: {result}")
                        return {"response": "Error: Could not parse LLM response."}

                except requests.exceptions.HTTPError as http_err:
                    if http_err.response.status_code == 429:
                        logger.warning(f"Rate limit hit (HTTP 429). Retrying in {retry_delay_seconds} seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay_seconds)
                    else:
                        logger.error(f"HTTP Error calling Gemini API: {http_err}. Status Code: {http_err.response.status_code}. Response: {http_err.response.text}")
                        raise
                except requests.exceptions.ConnectionError as ce:
                    logger.error(f"Connection Error to Gemini API: {ce}. Please check your network connection. (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay_seconds)
                except requests.exceptions.RequestException as re:
                    logger.error(f"General Request Error to Gemini API: {re}. (Attempt {attempt + 1}/{max_retries})")
                    raise
                except Exception as e:
                    logger.error(f"An unexpected error occurred while calling Gemini LLM: {str(e)}. (Attempt {attempt + 1}/{max_retries})")
                    raise

            logger.error(f"Failed to get response from Gemini API after {max_retries} attempts due to persistent rate limiting or other errors.")
            return {"response": "Error: Failed to get LLM response after multiple retries."}

        elif self.llm_provider == "gemma-hf":
            # --- Hugging Face Gemma 3 12B LLM logic ---
            if not self.api_key:
                logger.error("Cannot call Hugging Face Inference API: HUGGINGFACE_API_KEY is not set.")
                raise ValueError("HUGGINGFACE_API_KEY is required for Hugging Face Inference API calls.")

            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": 0.1,
                    "max_new_tokens": 2048,
                    "return_full_text": False 
                }
            }
            
            max_retries = 5
            retry_delay_seconds = 10 

            for attempt in range(max_retries):
                try:
                    logger.info(f"TextProcessor calling Hugging Face Gemma LLM with prompt (first 200 chars): {prompt[:200].replace(newline_char, ' ')}... (Attempt {attempt + 1}/{max_retries})")

                    response = requests.post(self.hf_api_url, headers=headers, json=payload)
                    response.raise_for_status() 

                    result = response.json()
                    
                    if isinstance(result, list) and result and "generated_text" in result[0]:
                        generated_text = result[0]["generated_text"]
                        logger.info("Successfully received response from Hugging Face Gemma API.")
                        return {"response": generated_text}
                    else:
                        logger.warning(f"Unexpected response structure from Hugging Face Gemma API: {result}")
                        return {"response": "Error: Could not parse LLM response from Hugging Face."}

                except requests.exceptions.HTTPError as http_err:
                    if http_err.response.status_code == 429:
                        logger.warning(f"Rate limit hit (HTTP 429). Retrying in {retry_delay_seconds} seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay_seconds)
                    else:
                        logger.error(f"HTTP Error calling Hugging Face Gemma API: {http_err}. Status Code: {http_err.response.status_code}. Response: {http_err.response.text}")
                        raise
                except requests.exceptions.ConnectionError as ce:
                    logger.error(f"Connection Error to Hugging Face Gemma API: {ce}. Please check your network connection. (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay_seconds)
                except requests.exceptions.RequestException as re:
                    logger.error(f"General Request Error to Hugging Face Gemma API: {re}. (Attempt {attempt + 1}/{max_retries})")
                    raise
                except Exception as e:
                    logger.error(f"An unexpected error occurred while calling Hugging Face Gemma LLM: {str(e)}. (Attempt {attempt + 1}/{max_retries})")
                    raise
            
            logger.error(f"Failed to get response from Hugging Face Gemma API after {max_retries} attempts due to persistent rate limiting or other errors.")
            return {"response": "Error: Failed to get LLM response from Hugging Face after multiple retries."}

        elif self.llm_provider == "ollama": 
            # --- Ollama LLM logic ---
            try:
                logger.info(f"TextProcessor calling Ollama LLM with prompt (first 200 chars): {prompt[:200].replace(newline_char, ' ')}...")

                response = requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1, 
                            "num_ctx": 4096 
                        }
                    }
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.ConnectionError as ce:
                logger.error(f"Connection Error to Ollama: {ce}. Is Ollama server running at {self.ollama_base_url}?")
                raise
            except requests.exceptions.RequestException as re:
                logger.error(f"Request Error to Ollama: {re}")
                raise
            except Exception as e:
                logger.error(f"Error calling Ollama LLM: {str(e)}")
                raise
        else:
            logger.error(f"No valid LLM provider '{self.llm_provider}' recognized. Cannot call LLM.")
            return {"response": "Error: LLM provider not correctly configured or recognized."}

    # def __init__(self, ollama_base_url: str = "http://localhost:11434"):
    #     self.ollama_base_url = ollama_base_url
    #     self.model_name = "llama3.1:8b" # Chang model with preference
    #     self.allowed_event_types = FIELD_VALUE_SCHEMA["event_type"]
    #     self.allowed_event_sub_types = ALL_EVENT_SUB_TYPES
        
    #     # Create a mapping for common casing issues for literal fields
    #     self.literal_field_corrections = {
    #         "state_of_victim": {val.lower(): val for val in FIELD_VALUE_SCHEMA["state_of_victim"]},
    #         "victim_gender": {val.lower(): val for val in FIELD_VALUE_SCHEMA["victim_gender"]},
    #         "repeat_incident": {v.lower(): v for v in ["yes", "no", "not specified", "not applicable"]},
    #         "need_ambulance": {v.lower(): v for v in ["yes", "no", "not specified", "not applicable"]},
    #         "children_involved": {v.lower(): v for v in ["yes", "no", "not specified", "not applicable"]},
    #     }
    #     # Uncomment below lines to use Gemini 2.0 Flash model
    #     # self.api_key = os.getenv("GEMINI_API_KEY")
    #     # self.model_name = "gemini-2.0-flash" 

    # def _call_llm(self, prompt: str) -> Dict[str, Any]:
    #     """Call the Ollama LLM API"""
    #     try:
    #         newline_char = '\n'
    #         logger.info(f"TextProcessor calling LLM with prompt (first 200 chars): {prompt[:200].replace(newline_char, ' ')}...")
            
    #         response = requests.post(
    #             f"{self.ollama_base_url}/api/generate",
    #             json={
    #                 "model": self.model_name,
    #                 "prompt": prompt,
    #                 "stream": False,
    #                 "options": {
    #                     "temperature": 0.1, # Keep temperature low for structured extraction
    #                     "num_ctx": 4096 # Adjust context window if prompt is long
    #                 }
    #             }
    #         )
    #         response.raise_for_status()
    #         return response.json()
    #     except requests.exceptions.ConnectionError as ce:
    #         logger.error(f"Connection Error to Ollama: {ce}. Is Ollama server running at {self.ollama_base_url}?")
    #         raise # Re-raise the exception to propagate it
    #     except requests.exceptions.RequestException as re:
    #         logger.error(f"Request Error to Ollama: {re}")
    #         raise
    #     except Exception as e:
    #         logger.error(f"Error calling LLM: {str(e)}")
    #         raise

    # Uncomment below method to use Gemini 2.0 Flash model
    # def _call_llm(self, prompt: str) -> Dict[str, Any]:
    #     """
    #     Call the Gemini 2.0 Flash LLM API with retry logic for rate limits.
    #     If a 429 (Too Many Requests) error is encountered, it will wait and retry.
    #     """
    #     max_retries = 5  # Maximum number of retries
    #     retry_delay_seconds = 60 # Initial delay in seconds (e.g., 1 minute)

    #     for attempt in range(max_retries):
    #         try:
    #             newline_char = '\n'
    #             logger.info(f"TextProcessor calling LLM with prompt (first 200 chars): {prompt[:200].replace(newline_char, ' ')}... (Attempt {attempt + 1}/{max_retries})")
                
    #             # Gemini API endpoint for text generation
    #             api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"

    #             # Construct the payload for Gemini API
    #             payload = {
    #                 "contents": [
    #                     {
    #                         "role": "user",
    #                         "parts": [{"text": prompt}]
    #                     }
    #                 ],
    #                 "generationConfig": {
    #                     "temperature": 0.1, # Keep temperature low for structured extraction
    #                     "maxOutputTokens": 2048 
    #                 }
    #             }

    #             response = requests.post(
    #                 api_url,
    #                 headers={'Content-Type': 'application/json'},
    #                 json=payload
    #             )
    #             response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
                
    #             result = response.json()

    #             # Extract the text from the Gemini API response
    #             if result.get("candidates") and result["candidates"][0].get("content") and \
    #                result["candidates"][0]["content"].get("parts") and result["candidates"][0]["content"]["parts"][0].get("text"):
    #                 generated_text = result["candidates"][0]["content"]["parts"][0]["text"]
    #                 logger.info("Successfully received response from Gemini API.")
    #                 return {"response": generated_text} # Return in a similar structure to original
    #             else:
    #                 logger.warning(f"Unexpected response structure from Gemini API: {result}")
    #                 return {"response": "Error: Could not parse LLM response."}

    #         except requests.exceptions.HTTPError as http_err:
    #             if http_err.response.status_code == 429: # Rate limit error
    #                 logger.warning(f"Rate limit hit (HTTP 429). Retrying in {retry_delay_seconds} seconds... (Attempt {attempt + 1}/{max_retries})")
    #                 time.sleep(retry_delay_seconds)
    #                 # You might want to increase the delay for subsequent retries (e.g., exponential backoff)
    #                 # retry_delay_seconds *= 2 
    #             else:
    #                 logger.error(f"HTTP Error calling Gemini API: {http_err}. Status Code: {http_err.response.status_code}. Response: {http_err.response.text}")
    #                 raise # Re-raise other HTTP errors immediately
    #         except requests.exceptions.ConnectionError as ce:
    #             logger.error(f"Connection Error to Gemini API: {ce}. Please check your network connection. (Attempt {attempt + 1}/{max_retries})")
    #             # For connection errors, you might also want to retry, but with a different strategy or fewer retries
    #             time.sleep(retry_delay_seconds) # Wait before retrying connection issues
    #         except requests.exceptions.RequestException as re:
    #             logger.error(f"General Request Error to Gemini API: {re}. (Attempt {attempt + 1}/{max_retries})")
    #             raise # Re-raise other request exceptions
    #         except Exception as e:
    #             logger.error(f"An unexpected error occurred while calling LLM: {str(e)}. (Attempt {attempt + 1}/{max_retries})")
    #             raise # Re-raise any other unexpected errors

    #     # If all retries are exhausted without success
    #     logger.error(f"Failed to get response from Gemini API after {max_retries} attempts due to persistent rate limiting or other errors.")
    #     return {"response": "Error: Failed to get LLM response after multiple retries."}


    def _create_extraction_prompt(self, text: str) -> str:
        safe_text = text.replace('"""', '\"\"\"')

        return f"""
SYSTEM ROLE:
You are an AI system assisting the Emergency Response Support System (ERSS) project, analyzing 112 emergency call transcripts. Your task is to classify and extract accurate structured metadata from unstructured call conversations between the caller and the emergency call taker.

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
    * Be strict about casing: "not specified" for `state_of_victim`, "not specified" for `victim_gender`, `repeat_incident`, `need_ambulance`, `children_involved`, and `generated_event_sub_type_detail`.

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

    def _fuzzy_match_field_name(self, llm_field_name: str) -> Optional[str]:
        """
        Attempts to fuzzy match an LLM-returned field name to a known ProcessedOutput field name.
        Returns the matched schema f  ield name or None if no good match is found.
        """
        llm_field_name_lower = llm_field_name.lower().replace('_', '')
        
        # Pre-process schema field names for matching (remove underscores for more robust fuzzy matching)
        processed_model_fields = {f.lower().replace('_', ''): f for f in ProcessedOutput.model_fields.keys()}

        # Try direct match after normalization
        if llm_field_name_lower in processed_model_fields:
            return processed_model_fields[llm_field_name_lower]

        # Try fuzzy matching
        # Consider a list of all normalized schema field names for get_close_matches
        normalized_schema_fields = list(processed_model_fields.keys())

        # Lower the cutoff if you want more aggressive matching, but be careful with false positives
        matches = get_close_matches(llm_field_name_lower, normalized_schema_fields, n=1, cutoff=0.75) # Adjusted cutoff

        if matches:
            matched_normalized_field = matches[0]
            original_schema_field = processed_model_fields[matched_normalized_field]
            logger.info(f"Fuzzy matched LLM field '{llm_field_name}' to '{original_schema_field}'.")
            return original_schema_field
        
        return None


    def _parse_llm_field_value_output(self, llm_output: str) -> dict:
        result = {}
        model_fields = ProcessedOutput.model_fields.keys() # Original schema field names
        
        default_not_specified_mapping = {
            "state_of_victim": "not specified",
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
            
            llm_field, value = line.split(':', 1)
            llm_field = llm_field.strip()
            raw_value_from_llm = value.strip() # Keep the raw value for specific checks

            # --- Fuzzy match the LLM's field name to the schema's field name ---
            matched_field = self._fuzzy_match_field_name(llm_field)

            if not matched_field:
                logger.warning(f"Unknown field returned by LLM: '{llm_field}'. Skipping.")
                continue
            
            # Use the matched_field for all subsequent logic
            field = matched_field
            
            # Initialize value to the raw LLM output, will be refined below
            processed_value = raw_value_from_llm

            # --- START OF NEW LOGIC FOR YES/NO EXTRACTION ---
            if field in ["need_ambulance", "children_involved", "repeat_incident"]: # Apply to all relevant yes/no fields
                lower_raw_value = raw_value_from_llm.lower()
                yes_match = re.search(r'\byes\b', lower_raw_value)
                no_match = re.search(r'\bno\b', lower_raw_value)

                if yes_match and not no_match: # Found 'yes' but not 'no'
                    processed_value = "yes"
                    logger.info(f"Extracted 'yes' for '{field}' from '{raw_value_from_llm}'.")
                elif no_match and not yes_match: # Found 'no' but not 'yes'
                    processed_value = "no"
                    logger.info(f"Extracted 'no' for '{field}' from '{raw_value_from_llm}'.")
                elif yes_match and no_match: # Both 'yes' and 'no' found (ambiguous)
                    processed_value = "not specified"
                    logger.warning(f"Ambiguous 'yes' and 'no' found for '{field}': '{raw_value_from_llm}'. Defaulting to 'not specified'.")
                # If neither 'yes' nor 'no' is found, processed_value remains raw_value_from_llm
                # and will be handled by the general literal field correction below.
            # --- END OF NEW LOGIC FOR YES/NO EXTRACTION ---
            
            # Normalize common "None", empty string, or invalid outputs to "not specified"
            if processed_value.lower() in ["null", "none", "", "not_defined", "n/a"]:
                processed_value = "not specified" 

            # --- Special Handling for event_sub_type and generated_event_sub_type_detail ---
            # Process these so they are in 'result' before the final correction step
            if field == "event_sub_type":
                normalized_llm_value = processed_value.upper()
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
                result[field] = processed_value if processed_value.lower() not in ["null", "none", ""] else "not specified"
            
            # --- General Literal Field Correction (case sensitivity & fuzzy matching) ---
            elif field in self.literal_field_corrections:
                # Use processed_value for lookup
                corrected_value = self.literal_field_corrections[field].get(processed_value.lower())
                if corrected_value: # If a direct lowercase match found, use its correct casing
                    result[field] = corrected_value
                else: # If not found, try get_close_matches for more flexibility
                    allowed_values_for_field = list(self.literal_field_corrections[field].keys())
                    matches = get_close_matches(processed_value.lower(), allowed_values_for_field, n=1, cutoff=0.8) # Adjust cutoff
                    if matches:
                        result[field] = self.literal_field_corrections[field][matches[0]]
                        logger.info(f"Corrected '{field}' value '{processed_value.lower()}' to closest match '{result[field]}'.")
                    else:
                        # If still no match, and it's a "not specified" type, use the default from mapping
                        result[field] = default_not_specified_mapping.get(field, "not specified") 
                        logger.warning(f"LLM returned invalid literal for '{field}': '{processed_value}'. Defaulting to '{result[field]}'.")
            
            else: # For other fields (freeform text)
                result[field] = processed_value

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