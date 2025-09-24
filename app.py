"""
Sentiment Analysis Toolbox - Entresent
A comprehensive tool for analyzing text sentiment using multiple models.
Supports both valence analysis and Ekman emotions.
Run with:  streamlit run app.py
"""

import json
import re
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
from openai import OpenAI
import nltk

# --- One-time downloads (quiet) ------------------------------------------------
nltk.download("vader_lexicon", quiet=True)

# --- Streamlit page config -----------------------------------------------------
st.set_page_config(page_title="Entresent - Sentiment Analysis", page_icon="🎭", layout="wide")

# --- Session state -------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = None
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "explain_mode" not in st.session_state:
    st.session_state.explain_mode = "None"
if "explanations" not in st.session_state:
    st.session_state.explanations = {}
if "measurement_type" not in st.session_state:
    st.session_state.measurement_type = "Intensity"
if "measurement_scale" not in st.session_state:
    st.session_state.measurement_scale = "Continuous (0-1)"

# --- Shared instructions -------------------------------------------------------
def playground_developer_instruction(measurement_type: str = "intensity") -> str:
    """
    Developer instruction for valence analysis.
    Supports intensity, likelihood, or both measurements.
    """
    if measurement_type == "intensity":
        return (
            "Analyze the sentiment of the provided text and score its positive, negative, and neutral "
            "sentiment independently.\n\n"
            "- For a given input text, output only a JSON object with the following structure "
            '(do not include any explanations, notes, or code fences):\n'
            '    {"positive": x, "negative": y, "neutral": z}\n'
            "- Each of x, y, and z should be a floating-point value between 0 and 1, representing the independent "
            "intensity of the respective sentiment.\n"
            "- Ensure that each sentiment dimension is scored independently and may sum to more or less than 1.\n\n"
            "# Output Format\n\n"
            "Return only a single-line JSON object in the format:\n"
            '{"positive": [float between 0 and 1], "negative": [float between 0 and 1], "neutral": [float between 0 and 1]}\n\n'
            "# Example\n\n"
            'Input:\nText: "I love sunny days, but today has been a bit overwhelming."\n\n'
            'Output:\n{"positive": 0.6, "negative": 0.4, "neutral": 0.3}\n\n'
            "# Notes\n\n"
            "- Do not include any explanation, commentary, or code formatting—only the JSON object as output.\n"
            "- Each sentiment score should be evaluated independently for the input text.\n\n"
            "Remember: Output only the JSON object as described, with all values in the [0, 1] range."
        )
    elif measurement_type == "likelihood":
        return (
            "Analyze the sentiment of the provided text and determine the likelihood (probability) "
            "of each sentiment being the primary sentiment.\n\n"
            "- For a given input text, output only a JSON object with the following structure "
            '(do not include any explanations, notes, or code fences):\n'
            '    {"positive": x, "negative": y, "neutral": z}\n'
            "- Each of x, y, and z should be a floating-point value between 0 and 1, representing the probability "
            "that this is the dominant sentiment.\n"
            "- IMPORTANT: The values MUST sum to exactly 1.0 as they are probabilities.\n\n"
            "# Output Format\n\n"
            "Return only a single-line JSON object in the format:\n"
            '{"positive": [probability 0-1], "negative": [probability 0-1], "neutral": [probability 0-1]}\n\n'
            "# Example\n\n"
            'Input:\nText: "I love sunny days, but today has been a bit overwhelming."\n\n'
            'Output:\n{"positive": 0.5, "negative": 0.3, "neutral": 0.2}\n\n'
            "# Notes\n\n"
            "- The three values must sum to 1.0\n"
            "- Think of this as: what's the probability this text is primarily positive vs negative vs neutral?\n\n"
            "Remember: Output only the JSON object with probabilities summing to 1.0"
        )
    else:  # both
        return (
            "Analyze the sentiment of the provided text and provide BOTH intensity and likelihood measures.\n\n"
            "- For a given input text, output only a JSON object with the following structure:\n"
            '    {\n'
            '        "intensity": {"positive": a, "negative": b, "neutral": c},\n'
            '        "likelihood": {"positive": x, "negative": y, "neutral": z}\n'
            '    }\n\n'
            "- Intensity values (a, b, c): Independent strength of each sentiment (0-1), may sum to more than 1\n"
            "- Likelihood values (x, y, z): Probability of each being primary sentiment, MUST sum to 1.0\n\n"
            "# Output Format\n\n"
            "Return only this JSON structure:\n"
            '{"intensity": {"positive": [0-1], "negative": [0-1], "neutral": [0-1]}, '
            '"likelihood": {"positive": [0-1], "negative": [0-1], "neutral": [0-1]}}\n\n'
            "# Example\n\n"
            'Input:\nText: "I love sunny days, but today has been a bit overwhelming."\n\n'
            'Output:\n{"intensity": {"positive": 0.6, "negative": 0.4, "neutral": 0.3}, '
            '"likelihood": {"positive": 0.5, "negative": 0.3, "neutral": 0.2}}\n\n'
            "Remember: Output only the JSON object as described."
        )

def ekman_developer_instruction(measurement_type: str = "intensity") -> str:
    """
    Instruction for Ekman emotions analysis.
    Supports intensity, likelihood, or both measurements.
    """
    if measurement_type == "intensity":
        return (
            "Analyze the emotions in the provided text and score each Ekman emotion "
            "independently.\n\n"
            "- For a given input text, output only a JSON object with the following structure "
            '(do not include any explanations, notes, or code fences):\n'
            '    {"happiness": a, "sadness": b, "fear": c, "anger": d, "disgust": e, "contempt": f, "surprise": g}\n'
            "- Each value (a through g) should be a floating-point value between 0 and 1, representing the independent "
            "intensity of the respective emotion.\n"
            "- Ensure that each emotion dimension is scored independently and may sum to more or less than 1.\n\n"
            "# Output Format\n\n"
            "Return only a single-line JSON object in the format:\n"
            '{"happiness": [float between 0 and 1], "sadness": [float between 0 and 1], "fear": [float between 0 and 1], '
            '"anger": [float between 0 and 1], "disgust": [float between 0 and 1], "contempt": [float between 0 and 1], '
            '"surprise": [float between 0 and 1]}\n\n'
            "# Example\n\n"
            'Input:\nText: "I was shocked to find out my friend betrayed me. I feel so angry and hurt."\n\n'
            'Output:\n{"happiness": 0.0, "sadness": 0.7, "fear": 0.2, "anger": 0.8, "disgust": 0.4, "contempt": 0.5, "surprise": 0.6}\n\n'
            "# Notes\n\n"
            "- Do not include any explanation, commentary, or code formatting—only the JSON object as output.\n"
            "- Each emotion score should be evaluated independently for the input text.\n"
            "- All seven emotions must be present in the output, even if their value is 0.\n\n"
            "Remember: Output only the JSON object as described, with all values in the [0, 1] range."
        )
    elif measurement_type == "likelihood":
        return (
            "Analyze the emotions in the provided text and determine the likelihood (probability) "
            "of each Ekman emotion being the primary emotion.\n\n"
            "- For a given input text, output only a JSON object with all 7 Ekman emotions\n"
            "- Each value should be a probability (0-1) that this is the dominant emotion\n"
            "- IMPORTANT: All seven values MUST sum to exactly 1.0 as they are probabilities\n\n"
            "# Output Format\n\n"
            '{"happiness": [prob], "sadness": [prob], "fear": [prob], "anger": [prob], '
            '"disgust": [prob], "contempt": [prob], "surprise": [prob]}\n\n'
            "# Example\n\n"
            'Input:\nText: "I was shocked to find out my friend betrayed me. I feel so angry and hurt."\n\n'
            'Output:\n{"happiness": 0.0, "sadness": 0.25, "fear": 0.05, "anger": 0.35, "disgust": 0.1, "contempt": 0.15, "surprise": 0.1}\n\n'
            "Remember: All seven values must sum to 1.0"
        )
    else:  # both
        return (
            "Analyze the emotions in the provided text and provide BOTH intensity and likelihood measures "
            "for all Ekman emotions.\n\n"
            "- Output a JSON object with this structure:\n"
            '    {\n'
            '        "intensity": {all 7 emotions with independent intensity scores 0-1},\n'
            '        "likelihood": {all 7 emotions with probabilities summing to 1.0}\n'
            '    }\n\n'
            "# Output Format\n\n"
            '{"intensity": {"happiness": [0-1], "sadness": [0-1], "fear": [0-1], "anger": [0-1], '
            '"disgust": [0-1], "contempt": [0-1], "surprise": [0-1]}, '
            '"likelihood": {"happiness": [0-1], "sadness": [0-1], "fear": [0-1], "anger": [0-1], '
            '"disgust": [0-1], "contempt": [0-1], "surprise": [0-1]}}\n\n'
            "# Example\n\n"
            'Input:\nText: "I was shocked to find out my friend betrayed me."\n\n'
            'Output:\n{"intensity": {"happiness": 0.0, "sadness": 0.7, "fear": 0.2, "anger": 0.8, '
            '"disgust": 0.4, "contempt": 0.5, "surprise": 0.6}, '
            '"likelihood": {"happiness": 0.0, "sadness": 0.25, "fear": 0.05, "anger": 0.35, '
            '"disgust": 0.1, "contempt": 0.15, "surprise": 0.1}}\n\n'
            "Remember: Output only the JSON object. Likelihood values must sum to 1.0"
        )

def get_explanation_prompt(mode: str, explain_mode: str) -> str:
    """
    Generate explanation prompt based on mode and explanation level.
    """
    if explain_mode == "None":
        return ""
    
    if explain_mode == "Short Explanation":
        if mode == "valence":
            return "\n\nAfter the JSON, provide a brief one-sentence explanation of why you assigned these sentiment scores."
        else:
            return "\n\nAfter the JSON, provide a brief one-sentence explanation of why you assigned these emotion scores."
    else:  # Long Explanation
        if mode == "valence":
            return "\n\nAfter the JSON, provide a detailed explanation (2-3 sentences) of your reasoning for each sentiment score, including specific words or phrases that influenced your assessment."
        else:
            return "\n\nAfter the JSON, provide a detailed explanation (2-3 sentences) of your reasoning for the emotion scores, including specific words or phrases that influenced your assessment of each emotion."

# --- Utilities -----------------------------------------------------------------
def _clip01(x: float) -> float:
    try:
        return float(max(0.0, min(1.0, x)))
    except Exception:
        return 0.0


LIKERT_LABELS = {
    1: "Not at all",
    2: "Slightly",
    3: "Moderately",
    4: "Strongly",
    5: "Extremely",
}


def _to_likert(value: float) -> int:
    """Convert a 0-1 score into a 1-5 Likert scale value."""
    try:
        numeric = float(value)
    except Exception:
        return 1

    scaled = np.floor(numeric * 5.0) + 1
    return int(np.clip(scaled, 1, 5))


def _apply_measurement_scale(scores: Dict[str, float], scale: str) -> Dict[str, float]:
    """Apply the configured measurement scale to DeepSeek/OpenAI scores."""

    if not isinstance(scores, dict) or scale != "Likert (1-5)":
        return scores

    scaled_scores: Dict[str, float] = {}
    for key, value in scores.items():
        if isinstance(value, dict):
            scaled_scores[key] = _apply_measurement_scale(value, scale)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            scaled_scores[key] = _to_likert(float(value))
        else:
            scaled_scores[key] = value

    return scaled_scores

def _safe_json_loads(s: str, mode: str = "valence", measurement_type: str = "intensity") -> Tuple[Dict[str, float], str]:
    """
    Parse JSON object safely, returning a dict with expected keys and any explanation.
    Returns (scores_dict, explanation)
    """
    explanation = ""
    
    # Try to extract JSON and explanation separately
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", s, flags=re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        # Get everything after the JSON as explanation
        json_end = json_match.end()
        if json_end < len(s):
            explanation = s[json_end:].strip()
    else:
        json_str = s
    
    try:
        data = json.loads(json_str)
    except Exception:
        data = {}
    
    # Handle "both" measurement type
    if measurement_type == "both":
        if "intensity" in data and "likelihood" in data:
            # Return intensity scores as primary, store likelihood separately if needed
            intensity_data = data.get("intensity", {})
            # For now, we'll return intensity scores (can extend to handle both)
            data = intensity_data
    
    if mode == "valence":
        scores = {
            "positive": _clip01(data.get("positive", 0.0)),
            "negative": _clip01(data.get("negative", 0.0)),
            "neutral": _clip01(data.get("neutral", 0.0)),
        }
    else:  # ekman
        scores = {
            "happiness": _clip01(data.get("happiness", 0.0)),
            "sadness": _clip01(data.get("sadness", 0.0)),
            "fear": _clip01(data.get("fear", 0.0)),
            "anger": _clip01(data.get("anger", 0.0)),
            "disgust": _clip01(data.get("disgust", 0.0)),
            "contempt": _clip01(data.get("contempt", 0.0)),
            "surprise": _clip01(data.get("surprise", 0.0)),
        }
    
    return scores, explanation

def _responses_text(resp) -> str:
    """
    Robustly extract text from a Responses API result.
    Prefers resp.output_text, falls back to walking resp.output.
    """
    try:
        if getattr(resp, "output_text", None):
            return resp.output_text
    except Exception:
        pass
    try:
        chunks: List[str] = []
        for item in (getattr(resp, "output", []) or []):
            if getattr(item, "type", "") == "message":
                for c in (getattr(item, "content", []) or []):
                    ctype = getattr(c, "type", "")
                    if ctype in ("output_text", "text"):
                        chunks.append(getattr(c, "text", "") or "")
        if chunks:
            return "".join(chunks)
    except Exception:
        pass
    return ""

# --- Core analyzer -------------------------------------------------------------
class SentimentAnalyzer:
    """Main class for handling different sentiment analysis models."""

    def __init__(self) -> None:
        self.models_initialized: Dict[str, bool] = {}
        self.vader = None
        self.setup_models()

    # ---------------- Local (no API) ----------------
    def setup_models(self) -> None:
        """Initialize models that don't require API keys."""
        try:
            self.vader = SentimentIntensityAnalyzer()
            self.models_initialized["VADER"] = True
        except Exception as e:
            st.warning(f"VADER initialization failed: {e}")
            self.models_initialized["VADER"] = False

    def analyze_vader(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER (valence only)."""
        try:
            assert self.vader is not None, "VADER not initialized"
            scores = self.vader.polarity_scores(text)
            return {
                "positive": float(scores.get("pos", 0.0)),
                "negative": float(scores.get("neg", 0.0)),
                "neutral": float(scores.get("neu", 0.0)),
                "compound": float(scores.get("compound", 0.0)),
            }
        except Exception as e:
            st.error(f"VADER analysis failed: {e}")
            return {"positive": 0.0, "negative": 0.0, "neutral": 0.0, "compound": 0.0}

    # ---------------- Hugging Face (requests) ----------------
    def analyze_siebert(self, text: str, api_key: str) -> Dict[str, float]:
        """Analyze sentiment using SiEBERT via Hugging Face Inference API (valence only)."""
        try:
            if not api_key:
                st.error("Hugging Face API key required for SiEBERT")
                return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

            api_url = "https://api-inference.huggingface.co/models/siebert/sentiment-roberta-large-english"
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.post(api_url, headers=headers, json={"inputs": text[:512]}, timeout=60)

            if response.status_code == 503:
                st.warning("SiEBERT model is loading on HF. Try again in a few seconds.")
                return {"positive": 0.5, "negative": 0.5, "neutral": 0.0}
            response.raise_for_status()

            result = response.json()
            if st.session_state.debug_mode:
                st.code(f"SiEBERT raw response:\n{json.dumps(result, indent=2)}")

            # HF output formats: [[{label,score}]] or [{label,score}]
            result_item = {}
            if isinstance(result, list) and result:
                result_item = result[0][0] if isinstance(result[0], list) and result[0] else result[0]
            elif isinstance(result, dict):
                result_item = result

            label = str(result_item.get("label", "")).lower()
            score = _clip01(float(result_item.get("score", 0.5)))

            if "positive" in label:
                return {"positive": score, "negative": _clip01(1 - score), "neutral": 0.0}
            if "negative" in label:
                return {"negative": score, "positive": _clip01(1 - score), "neutral": 0.0}
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}

        except Exception as e:
            st.error(f"SiEBERT analysis failed: {e}")
            if st.session_state.debug_mode:
                st.exception(e)
            return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    def analyze_bart(self, text: str, api_key: str, mode: str = "valence") -> Dict[str, float]:
        """
        Analyze sentiment using BART (MNLI) via Hugging Face Inference API.
        Supports both valence and Ekman emotions modes.
        """
        try:
            if not api_key:
                st.error("Hugging Face API key required for BART")
                if mode == "valence":
                    return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
                else:
                    return {"happiness": 0.0, "sadness": 0.0, "fear": 0.0, "anger": 0.0, 
                           "disgust": 0.0, "contempt": 0.0, "surprise": 0.0}

            api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
            headers = {"Authorization": f"Bearer {api_key}"}
            
            if mode == "valence":
                labels = ["positive", "negative", "neutral"]
            else:  # ekman
                labels = ["happiness", "sadness", "fear", "anger", "disgust", "contempt", "surprise"]
            
            payload = {
                "inputs": text[:512],
                "parameters": {"candidate_labels": labels, "multi_label": True},
            }
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)

            if response.status_code == 503:
                st.warning("BART model is loading on HF. Try again shortly.")
                if mode == "valence":
                    return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
                else:
                    return {label: 0.14 for label in labels}  # Equal distribution
                    
            response.raise_for_status()

            result = response.json()
            if st.session_state.debug_mode:
                st.code(f"BART raw response:\n{json.dumps(result, indent=2)}")

            scores_map: Dict[str, float] = {label: 0.0 for label in labels}
            blob = result[0] if isinstance(result, list) and result else result
            if isinstance(blob, dict) and "labels" in blob and "scores" in blob:
                for label, score in zip(blob["labels"], blob["scores"]):
                    if label in scores_map:
                        scores_map[label] = _clip01(float(score))
            
            return scores_map

        except Exception as e:
            st.error(f"BART analysis failed: {e}")
            if st.session_state.debug_mode:
                st.exception(e)
            if mode == "valence":
                return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            else:
                return {"happiness": 0.0, "sadness": 0.0, "fear": 0.0, "anger": 0.0, 
                       "disgust": 0.0, "contempt": 0.0, "surprise": 0.0}

    # ---------------- DeepSeek (OpenAI-compatible Chat Completions) ------------
    def analyze_deepseek(self, text: str, api_key: str, mode: str = "valence", text_idx: int = 0) -> Dict[str, float]:
        """
        Analyze sentiment using DeepSeek.
        Supports both valence and Ekman emotions modes with explanations and measurement types.
        """
        try:
            if not api_key:
                st.error("DeepSeek API key required for DeepSeek")
                if mode == "valence":
                    missing_key_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
                else:
                    missing_key_scores = {
                        "happiness": 0.0,
                        "sadness": 0.0,
                        "fear": 0.0,
                        "anger": 0.0,
                        "disgust": 0.0,
                        "contempt": 0.0,
                        "surprise": 0.0,
                    }
                return _apply_measurement_scale(missing_key_scores, st.session_state.measurement_scale)

            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

            measurement = st.session_state.measurement_type.lower()
            instruction = (
                playground_developer_instruction(measurement) if mode == "valence" 
                else ekman_developer_instruction(measurement)
            )
            
            # Add explanation request if enabled
            if st.session_state.explain_mode != "None":
                instruction += get_explanation_prompt(mode, st.session_state.explain_mode)
            
            user_text = f"Text: {text[:1000]}"

            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": user_text},
                ],
                temperature=0.0,
                max_tokens=400 if measurement == "both" else (300 if st.session_state.explain_mode == "Long Explanation" else 150),
            )

            result_text = (resp.choices[0].message.content or "").strip()
            scores, explanation = _safe_json_loads(result_text, mode, measurement)
            scores = _apply_measurement_scale(scores, st.session_state.measurement_scale)
            
            # Store explanation if present
            if explanation and st.session_state.explain_mode != "None":
                if "DeepSeek" not in st.session_state.explanations:
                    st.session_state.explanations["DeepSeek"] = {}
                st.session_state.explanations["DeepSeek"][text_idx] = explanation
            
            return scores

        except Exception as e:
            st.error(f"DeepSeek analysis failed: {e}")
            if st.session_state.debug_mode:
                st.exception(e)
            if mode == "valence":
                fallback_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            else:
                fallback_scores = {
                    "happiness": 0.0,
                    "sadness": 0.0,
                    "fear": 0.0,
                    "anger": 0.0,
                    "disgust": 0.0,
                    "contempt": 0.0,
                    "surprise": 0.0,
                }
            return _apply_measurement_scale(fallback_scores, st.session_state.measurement_scale)

    # ---------------- OpenAI (GPT-5 nano via Responses API with GPT-4o fallback) --------------------
    def analyze_gpt5nano(self, text: str, api_key: str, mode: str = "valence", text_idx: int = 0) -> Dict[str, float]:
        """
        Analyze sentiment using OpenAI GPT-5 nano via Responses API with GPT-4o fallback.
        Supports both valence and Ekman emotions modes with explanations and measurement types.
        """
        try:
            if not api_key:
                st.error("OpenAI API key required for GPT-5 nano")
                if mode == "valence":
                    missing_key_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
                else:
                    missing_key_scores = {
                        "happiness": 0.0,
                        "sadness": 0.0,
                        "fear": 0.0,
                        "anger": 0.0,
                        "disgust": 0.0,
                        "contempt": 0.0,
                        "surprise": 0.0,
                    }
                return _apply_measurement_scale(missing_key_scores, st.session_state.measurement_scale)

            client = OpenAI(api_key=api_key)
            measurement = st.session_state.measurement_type.lower()
            instruction = (
                playground_developer_instruction(measurement) if mode == "valence" 
                else ekman_developer_instruction(measurement)
            )
            
            # Add explanation request if enabled
            if st.session_state.explain_mode != "None":
                instruction += get_explanation_prompt(mode, st.session_state.explain_mode)

            # Try GPT-5 nano first
            try:
                # For GPT-5 nano, we can't use strict JSON schema with "both" or with explanations
                # So only use schema for simple cases
                use_schema = (st.session_state.explain_mode == "None" and measurement != "both")
                
                if use_schema:
                    # Build schema based on mode
                    if mode == "valence":
                        schema_obj = {
                            "type": "object",
                            "properties": {
                                "positive": {"type": "number", "minimum": 0, "maximum": 1},
                                "negative": {"type": "number", "minimum": 0, "maximum": 1},
                                "neutral":  {"type": "number", "minimum": 0, "maximum": 1},
                            },
                            "required": ["positive", "negative", "neutral"],
                            "additionalProperties": False,
                        }
                        schema_name = "SentimentScores"
                    else:  # ekman
                        schema_obj = {
                            "type": "object",
                            "properties": {
                                "happiness": {"type": "number", "minimum": 0, "maximum": 1},
                                "sadness": {"type": "number", "minimum": 0, "maximum": 1},
                                "fear": {"type": "number", "minimum": 0, "maximum": 1},
                                "anger": {"type": "number", "minimum": 0, "maximum": 1},
                                "disgust": {"type": "number", "minimum": 0, "maximum": 1},
                                "contempt": {"type": "number", "minimum": 0, "maximum": 1},
                                "surprise": {"type": "number", "minimum": 0, "maximum": 1},
                            },
                            "required": ["happiness", "sadness", "fear", "anger", "disgust", "contempt", "surprise"],
                            "additionalProperties": False,
                        }
                        schema_name = "EkmanEmotions"
                    
                    text_format = {
                        "format": {
                            "type": "json_schema",
                            "name": schema_name,
                            "schema": schema_obj,
                        },
                        "verbosity": "low",
                    }
                else:
                    text_format = {"verbosity": "medium"}

                include_list = ["reasoning.encrypted_content"] if st.session_state.debug_mode else []

                resp = client.responses.create(
                    model="gpt-5-nano",
                    input=[
                        {"role": "developer", "content": [{"type": "input_text", "text": instruction}]},
                        {"role": "user", "content": [{"type": "input_text", "text": f"Text: {text[:1000]}"}]},
                    ],
                    text=text_format,
                    reasoning={"effort": "low"},
                    tools=[],
                    store=False,
                    include=include_list,
                    max_output_tokens=400 if measurement == "both" else (300 if st.session_state.explain_mode == "Long Explanation" else 150),
                    temperature=0.0,
                )

                if st.session_state.debug_mode:
                    st.code(resp.model_dump_json(indent=2) if hasattr(resp, "model_dump_json") else str(resp))

                out = _responses_text(resp).strip()
                if st.session_state.debug_mode:
                    st.code(f"GPT-5 nano output_text:\n{out}")

                scores, explanation = _safe_json_loads(out, mode, measurement)
                scores = _apply_measurement_scale(scores, st.session_state.measurement_scale)
                
                # Store explanation if present
                if explanation and st.session_state.explain_mode != "None":
                    if "GPT-5 nano" not in st.session_state.explanations:
                        st.session_state.explanations["GPT-5 nano"] = {}
                    st.session_state.explanations["GPT-5 nano"][text_idx] = explanation
                
                return scores

            except Exception as nano_error:
                # Fallback to GPT-4o
                st.info("📋 GPT-5 nano unavailable, using GPT-4o fallback...")
                
                messages = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": f"Text: {text[:1000]}"}
                ]
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.0,
                    max_tokens=400 if measurement == "both" else (300 if st.session_state.explain_mode == "Long Explanation" else 150),
                )
                
                result_text = response.choices[0].message.content
                scores, explanation = _safe_json_loads(result_text, mode, measurement)
                scores = _apply_measurement_scale(scores, st.session_state.measurement_scale)
                
                # Store explanation if present
                if explanation and st.session_state.explain_mode != "None":
                    if "GPT-4o (fallback)" not in st.session_state.explanations:
                        st.session_state.explanations["GPT-4o (fallback)"] = {}
                    st.session_state.explanations["GPT-4o (fallback)"][text_idx] = explanation
                
                return scores

        except Exception as e:
            st.error(f"OpenAI analysis failed: {e}")
            if st.session_state.debug_mode:
                st.exception(e)
            if mode == "valence":
                fallback_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            else:
                fallback_scores = {
                    "happiness": 0.0,
                    "sadness": 0.0,
                    "fear": 0.0,
                    "anger": 0.0,
                    "disgust": 0.0,
                    "contempt": 0.0,
                    "surprise": 0.0,
                }
            return _apply_measurement_scale(fallback_scores, st.session_state.measurement_scale)

    # ---------------- Dispatcher ----------------
    def analyze_text(self, text: str, model: str, api_keys: Dict[str, str], mode: str = "valence", text_idx: int = 0) -> Dict[str, float]:
        """Dispatch to the selected model."""
        if model == "VADER":
            return self.analyze_vader(text)
        if model == "SiEBERT":
            return self.analyze_siebert(text, api_keys.get("Entresent_HF_API", ""))
        if model == "BART":
            return self.analyze_bart(text, api_keys.get("Entresent_HF_API", ""), mode)
        if model == "DeepSeek":
            return self.analyze_deepseek(text, api_keys.get("Entresent_DS_API", ""), mode, text_idx)
        if model == "GPT-5 nano":
            return self.analyze_gpt5nano(text, api_keys.get("Entresent_OAI_API", ""), mode, text_idx)
        
        # Default return based on mode
        if mode == "valence":
            return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        else:
            return {"happiness": 0.0, "sadness": 0.0, "fear": 0.0, "anger": 0.0, 
                   "disgust": 0.0, "contempt": 0.0, "surprise": 0.0}

# --- File helpers --------------------------------------------------------------
def load_texts_from_file(uploaded_file) -> List[str]:
    """Load texts from uploaded CSV or Excel file (first column or 'text'/'Text')."""
    texts: List[str] = []
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format")
            return texts

        if "text" in df.columns:
            texts = df["text"].dropna().astype(str).tolist()
        elif "Text" in df.columns:
            texts = df["Text"].dropna().astype(str).tolist()
        else:
            texts = df.iloc[:, 0].dropna().astype(str).tolist()

        return texts
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return []

def create_results_dataframe(texts: List[str], results: Dict, truncate: bool = True, include_explanations: bool = False) -> pd.DataFrame:
    """Create a formatted DataFrame from analysis results including explanations if available."""
    df_data: List[Dict] = []

    for i, text in enumerate(texts):
        display_text = text[:100] + "..." if (truncate and len(text) > 100) else text
        row: Dict = {"Text": display_text}
        
        # Add scores
        for model, scores_list in results.items():
            if i < len(scores_list):
                for sentiment, value in scores_list[i].items():
                    if sentiment != "compound":  # Skip compound for cleaner display
                        if isinstance(value, (int, float, np.integer, np.floating)):
                            if (
                                st.session_state.measurement_scale == "Likert (1-5)"
                                and isinstance(value, (int, np.integer))
                                and int(value) in LIKERT_LABELS
                            ):
                                row[f"{model}_{sentiment}"] = int(value)
                            else:
                                row[f"{model}_{sentiment}"] = round(float(value), 3)
                        else:
                            row[f"{model}_{sentiment}"] = value
        
        # Add explanations if available
        if include_explanations and st.session_state.explanations:
            for model_name in ["DeepSeek", "GPT-5 nano", "GPT-4o (fallback)"]:
                if model_name in st.session_state.explanations:
                    if i in st.session_state.explanations[model_name]:
                        explanation = st.session_state.explanations[model_name][i]
                        # Truncate long explanations for table display
                        if len(explanation) > 150:
                            explanation = explanation[:147] + "..."
                        row[f"{model_name} Explanation"] = explanation
        
        df_data.append(row)

    return pd.DataFrame(df_data)

# --- App -----------------------------------------------------------------------
def main() -> None:
    # Header
    st.title("🎭 Entresent - Sentiment Analysis Toolbox")
    st.markdown("*Advanced sentiment and emotion analysis using state-of-the-art AI models. Analyze text to understand emotional valence (positive/negative/neutral) or detect specific emotions using Ekman's framework.*")
    
    # ---------------- Sidebar: Config (reordered) ----------------
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # 1. Analysis Mode (first)
        st.subheader("1. Analysis Mode")
        analysis_mode = st.radio(
            "Select analysis type:",
            ["Valence", "Ekman Emotions"],
            help="Valence: positive/negative/neutral | Ekman: 7 basic emotions"
        )
        
        st.divider()
        
        # 2. Model Selection (second)
        st.subheader("2. Model Selection")
        
        if analysis_mode == "Valence":
            available_models = ["VADER", "SiEBERT", "BART", "DeepSeek", "GPT-5 nano"]
        else:  # Ekman Emotions
            available_models = ["BART", "DeepSeek", "GPT-5 nano"]
        
        benchmark_mode = st.checkbox("🏁 Benchmark Mode (Run all available models)", value=False)
        
        if not benchmark_mode:
            selected_model = st.selectbox("Select Model", available_models)
            models_to_run = [selected_model]
        else:
            models_to_run = available_models
            st.info(f"Running all {len(models_to_run)} models")
        
        st.divider()
        
        # 3. API Keys (third)
        st.subheader("3. API Keys")
        
        # Retrieve API keys (simplified without the info messages here)
        use_secrets = st.checkbox("Use Streamlit Secrets", value=True)
        
        api_keys: Dict[str, str] = {}
        if use_secrets:
            try:
                api_keys["Entresent_HF_API"] = st.secrets.get("Entresent_HF_API", "")
                api_keys["Entresent_DS_API"] = st.secrets.get("Entresent_DS_API", "")
                api_keys["Entresent_OAI_API"] = st.secrets.get("Entresent_OAI_API", "")
                st.success("✅ Using Streamlit secrets")
            except Exception:
                st.warning("Secrets not configured")
                api_keys["Entresent_HF_API"] = st.text_input("Hugging Face API Key", type="password")
                api_keys["Entresent_DS_API"] = st.text_input("DeepSeek API Key", type="password")
                api_keys["Entresent_OAI_API"] = st.text_input("OpenAI API Key", type="password")
        else:
            api_keys["Entresent_HF_API"] = st.text_input("Hugging Face API Key", type="password")
            api_keys["Entresent_DS_API"] = st.text_input("DeepSeek API Key", type="password")
            api_keys["Entresent_OAI_API"] = st.text_input("OpenAI API Key", type="password")
        
        # Model availability check
        if analysis_mode == "Valence":
            models_status = [
                "✅ VADER" if "VADER" in available_models else "",
                "✅ SiEBERT" if api_keys.get("Entresent_HF_API") and "SiEBERT" in available_models else "❌ SiEBERT",
                "✅ BART" if api_keys.get("Entresent_HF_API") and "BART" in available_models else "❌ BART",
                "✅ DeepSeek" if api_keys.get("Entresent_DS_API") else "⚠️ DeepSeek",
                "✅ GPT-5 nano" if api_keys.get("Entresent_OAI_API") else "⚠️ GPT-5 nano",
            ]
        else:
            models_status = [
                "✅ BART" if api_keys.get("Entresent_HF_API") else "❌ BART",
                "✅ DeepSeek" if api_keys.get("Entresent_DS_API") else "⚠️ DeepSeek",
                "✅ GPT-5 nano" if api_keys.get("Entresent_OAI_API") else "⚠️ GPT-5 nano",
            ]
        
        with st.expander("Model Availability", expanded=False):
            for status in models_status:
                if status:
                    st.write(status)
        
        with st.expander("📚 API Key Help", expanded=False):
            st.markdown(
                """
- **Hugging Face**: https://huggingface.co/settings/tokens  
- **DeepSeek**: https://platform.deepseek.com/  
- **OpenAI**: https://platform.openai.com/api-keys
"""
            )
        
        st.divider()
        
        # 4. Advanced Settings (last)
        st.subheader("4. Advanced Settings")
        
        with st.expander("🔧 Advanced Options", expanded=False):
            debug_mode = st.checkbox(
                "Enable Debug Mode", 
                value=False, 
                help="Show raw outputs and tracebacks"
            )
            st.session_state.debug_mode = debug_mode
            
            st.divider()
            
            # Measurement Type selection
            st.markdown("**📊 Measurement Type**")
            measurement_type = st.selectbox(
                "DeepSeek & OpenAI measurement",
                ["Intensity", "Likelihood", "Both"],
                help="Intensity: Independent strength (can sum >1) | Likelihood: Probability (sums to 1) | Both: Get both metrics"
            )
            st.session_state.measurement_type = measurement_type
            
            if measurement_type == "Intensity":
                st.caption("📈 Measures emotional strength independently")
            elif measurement_type == "Likelihood":
                st.caption("🎲 Measures probability distribution")
            else:
                st.caption("📊 Returns both intensity and likelihood")

            st.divider()

            # Measurement scaling selection
            st.markdown("**📐 Output Scaling**")
            measurement_scale = st.selectbox(
                "Scaling for DeepSeek & OpenAI outputs",
                ["Continuous (0-1)", "Likert (1-5)"],
                help="Choose how DeepSeek and OpenAI scores should be displayed",
            )
            st.session_state.measurement_scale = measurement_scale

            if measurement_scale == "Continuous (0-1)":
                st.caption("🔁 Raw scores between 0 and 1")
            else:
                likert_description = " | ".join(f"{value} - {label}" for value, label in LIKERT_LABELS.items())
                st.caption(f"🪄 Likert scale: {likert_description}")

            st.divider()

            # Explainable AI settings
            st.markdown("**🤖 Explainable AI**")
            explain_mode = st.selectbox(
                "Explanation Level (OpenAI & DeepSeek)",
                ["None", "Short Explanation", "Long Explanation"],
                help="Get AI reasoning for sentiment/emotion scores"
            )
            st.session_state.explain_mode = explain_mode
            
            if explain_mode != "None":
                st.info(f"💡 {explain_mode} enabled")

    # ---------------- Main content area (full width) ----------------
    # Input section
    st.subheader("📝 Input Text")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        input_method = st.radio("Input Method", ["Text Area", "File Upload"], horizontal=True)
    
    texts: List[str] = []
    if input_method == "Text Area":
        text_input = st.text_area(
            "Enter texts (one per line)",
            height=200,
            placeholder="Enter your first text here\nEnter your second text here\n...",
        )
        if text_input:
            texts = [t.strip() for t in text_input.split("\n") if t.strip()]
    else:
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
        if uploaded_file:
            texts = load_texts_from_file(uploaded_file)
            st.success(f"✅ Loaded {len(texts)} texts from file")
    
    # Analysis settings section
    st.subheader("🎯 Analysis Settings")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Analysis Mode:** {analysis_mode}")
    with col2:
        st.markdown(f"**Texts to Analyze:** {len(texts)}")
    with col3:
        st.markdown(f"**Models to Run:** {len(models_to_run)}")
    
    # Check for missing keys
    missing_keys = []
    if not benchmark_mode:
        if selected_model in ["SiEBERT", "BART"] and not api_keys.get("Entresent_HF_API"):
            missing_keys.append(f"Hugging Face API key (for {selected_model})")
        if selected_model == "DeepSeek" and not api_keys.get("Entresent_DS_API"):
            missing_keys.append("DeepSeek API key")
        if selected_model == "GPT-5 nano" and not api_keys.get("Entresent_OAI_API"):
            missing_keys.append("OpenAI API key")
    else:
        if "BART" in models_to_run and not api_keys.get("Entresent_HF_API"):
            missing_keys.append("Hugging Face API key")
        if "DeepSeek" in models_to_run and not api_keys.get("Entresent_DS_API"):
            missing_keys.append("DeepSeek API key")
        if "GPT-5 nano" in models_to_run and not api_keys.get("Entresent_OAI_API"):
            missing_keys.append("OpenAI API key")
    
    if missing_keys:
        st.warning(f"⚠️ Missing: {', '.join(missing_keys)}")
    
    # Single centered analyze button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        analyze_button = st.button(
            "**🚀 ANALYZE**", 
            type="primary", 
            disabled=len(texts) == 0,
            use_container_width=True
        )
    
    # ---------------- Execution ----------------
    if analyze_button and texts:
        st.session_state.explanations = {}  # Clear previous explanations
        analyzer = SentimentAnalyzer()
        results: Dict[str, List[Dict[str, float]]] = {}
        
        total_ops = max(1, len(models_to_run) * len(texts))
        progress = st.progress(0)
        status = st.empty()
        op = 0
        
        mode = analysis_mode.lower().replace(" ", "_")
        if mode == "ekman_emotions":
            mode = "ekman"
        
        for model in models_to_run:
            status.text(f"Running {model}...")
            model_results: List[Dict[str, float]] = []
            for idx, t in enumerate(texts):
                try:
                    scores = analyzer.analyze_text(t, model, api_keys, mode, idx)
                except Exception as e:
                    st.error(f"{model} failed on text {idx+1}: {e}")
                    if st.session_state.debug_mode:
                        st.exception(e)
                    if mode == "valence":
                        scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
                    else:
                        scores = {"happiness": 0.0, "sadness": 0.0, "fear": 0.0, "anger": 0.0, 
                                "disgust": 0.0, "contempt": 0.0, "surprise": 0.0}
                model_results.append(scores)
                op += 1
                progress.progress(op / total_ops)
            results[model] = model_results
        
        progress.empty()
        status.empty()
        st.session_state.results = (texts, results, analysis_mode)
        st.success("✅ Analysis complete!")
        st.rerun()

    # ---------------- Results ----------------
    if st.session_state.results:
        texts, results, result_mode = st.session_state.results
        st.divider()
        st.subheader(f"📊 Results - {result_mode}")

        # Options row
        col1, col2, col3 = st.columns(3)
        with col1:
            show_full_text = st.checkbox("Show full text", value=False)
        with col2:
            highlight_max = st.checkbox("Highlight max values", value=True)
        with col3:
            download_format = st.selectbox("Download format", ["CSV", "Excel"])

        # Results DataFrame with explanations
        include_explanations = st.session_state.explain_mode != "None" and bool(st.session_state.explanations)
        df_results = create_results_dataframe(texts, results, truncate=not show_full_text, include_explanations=include_explanations)

        if highlight_max:
            # Only highlight numeric columns (not explanation columns)
            score_cols = [c for c in df_results.columns if c != "Text" and "Explanation" not in c]
            styled = df_results.style.highlight_max(subset=score_cols, color="lightgreen", axis=1)
            st.dataframe(styled, width="stretch")
        else:
            st.dataframe(df_results, width="stretch")

        if st.session_state.measurement_scale == "Likert (1-5)":
            legend = ", ".join(f"{value} - {label}" for value, label in LIKERT_LABELS.items())
            st.caption(f"Likert legend: {legend}")

        # Downloads
        if download_format == "CSV":
            csv_data = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name=f"{result_mode.lower().replace(' ', '_')}_analysis_results.csv",
                mime="text/csv",
            )
        else:
            try:
                output = BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    df_results.to_excel(writer, index=False, sheet_name="Results")
                st.download_button(
                    label="📥 Download Results as Excel",
                    data=output.getvalue(),
                    file_name=f"{result_mode.lower().replace(' ', '_')}_analysis_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as e:
                st.error(f"Excel export failed. Falling back to CSV. Error: {e}")
                csv_data = df_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name=f"{result_mode.lower().replace(' ', '_')}_analysis_results.csv",
                    mime="text/csv",
                )

        # Summary statistics (only for multiple models)
        if len(results) > 1:
            st.divider()
            st.subheader("📈 Summary Statistics")
            summary_rows = []
            
            # Get emotion/sentiment keys from first result
            first_model_results = list(results.values())[0]
            if first_model_results:
                emotion_keys = [k for k in first_model_results[0].keys() if k != "compound"]
            else:
                emotion_keys = []
            
            for model, scores in results.items():
                row = {"Model": model}
                for emotion in emotion_keys:
                    avg_val = float(np.mean([s.get(emotion, 0.0) for s in scores])) if scores else 0.0
                    row[f"Avg {emotion.capitalize()}"] = round(avg_val, 3)
                summary_rows.append(row)
            
            df_summary = pd.DataFrame(summary_rows)
            st.dataframe(df_summary, width="stretch")
            
            # Create bar chart
            chart_data = df_summary.set_index("Model")
            st.bar_chart(chart_data)

if __name__ == "__main__":
    main()
