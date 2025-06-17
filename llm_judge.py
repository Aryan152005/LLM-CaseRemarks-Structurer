import json
import requests
from typing import Dict, Any, Tuple
from loguru import logger

class LLMJudge:
    def __init__(self, ollama_base_url: str = "http://localhost:11434", model_name: str = "llama3.1:8b"):
        self.ollama_base_url = ollama_base_url
        self.model_name = model_name
        
    def _create_similarity_prompt(self, text1: str, text2: str) -> str:
        """Create a prompt for text similarity evaluation"""
        return f"""
You are a text similarity judge. Your task is to determine if two texts are similar in meaning or intent, despite potential differences in phrasing, spelling, or grammatical errors.

Examples:
Text A: "Car stolen from parking lot"
Text B: "Vehicle theft reported in parking area"
Response: 1 (similar)

Text A: "Fire in building"
Text B: "Water leak in basement"
Response: 0 (not similar)

Text A: "Man threatening with knife"
Text B: "Person brandishing weapon"
Response: 1 (similar)

Now, evaluate these texts:
Text A: "{text1}"
Text B: "{text2}"

Respond with ONLY '1' if the texts are similar in meaning/intent, or '0' if they are not similar.
Do not include any explanation or additional text.
"""

    def is_similar(self, text1: str, text2: str) -> Tuple[int, float]:
        """Determine if two texts are similar using LLM judgment
        
        Returns:
            Tuple[int, float]: (similarity_score (0 or 1), confidence_score (0-1))
        """
        try:
            prompt = self._create_similarity_prompt(text1, text2)
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract the response and clean it
            response_text = result.get("response", "").strip()
            similarity_score = int(response_text) if response_text in ["0", "1"] else 0
            
            # Log the comparison for debugging
            logger.debug(f"Comparing texts:\nA: {text1}\nB: {text2}\nScore: {similarity_score}")
            
            return similarity_score, 1.0  # For now, using 1.0 as confidence
            
        except Exception as e:
            logger.error(f"Error in LLM similarity judgment: {str(e)}")
            return 0, 0.0

    def batch_similarity(self, text_pairs: list[Tuple[str, str]]) -> list[Tuple[int, float]]:
        """Evaluate similarity for multiple text pairs"""
        return [self.is_similar(text1, text2) for text1, text2 in text_pairs] 