"""
Sentiment Analysis Toolbox
A comprehensive tool for analyzing text sentiment using multiple models.
Supports both valence analysis and Ekman emotions.
Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import re
from io import BytesIO
from typing import Dict, List

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
st.set_page_config(page_title="Sentiment Analysis Toolbox", page_icon="🎭", layout="wide")

# --- Session state -------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = None
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

# --- Shared instructions -------------------------------------------------------
def playground_developer_instruction() -> str:
    """
    Mirrors your working Responses API developer text from the Playground.
    Reused for GPT-5 (Responses) and for DeepSeek (as a system message).
    """
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

def ekman_developer_instruction() -> str:
    """
    Instruction for Ekman emotions analysis.
    Similar to valence instruction but for 7 basic emotions.
    """
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

# --- Utilities -----------------------------------------------------------------
def _clip01(x: float) -> float:
    try:
        return float(max(0.0, min(1.0, x)))
    except Exception:
        return 0.0

def _safe_json_loads(s: str, mode: str = "valence") -> Dict[str, float]:
    """
    Parse JSON object safely, returning a dict with expected keys.
    For valence: positive/negative/neutral
    For ekman: happiness/sadness/fear/anger/disgust/contempt/surprise
    """
    try:
        data = json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
    
    if mode == "valence":
        return {
            "positive": _clip01(data.get("positive", 0.0)),
            "negative": _clip01(data.get("negative", 0.0)),
            "neutral": _clip01(data.get("neutral", 0.0)),
        }
    else:  # ekman
        return {
            "happiness": _clip01(data.get("happiness", 0.0)),
            "sadness": _clip01(data.get("sadness", 0.0)),
            "fear": _clip01(data.get("fear", 0.0)),
            "anger": _clip01(data.get("anger", 0.0)),
            "disgust": _clip01(data.get("disgust", 0.0)),
            "contempt": _clip01(data.get("contempt", 0.0)),
            "surprise": _clip01(data.get("surprise", 0.0)),
        }

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
        self.vader: SentimentIntensityAnalyzer | None = None
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
    def analyze_deepseek(self, text: str, api_key: str, mode: str = "valence") -> Dict[str, float]:
        """
        Analyze sentiment using DeepSeek.
        Supports both valence and Ekman emotions modes.
        """
        try:
            if not api_key:
                st.error("DeepSeek API key required for DeepSeek")
                if mode == "valence":
                    return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
                else:
                    return {"happiness": 0.0, "sadness": 0.0, "fear": 0.0, "anger": 0.0, 
                           "disgust": 0.0, "contempt": 0.0, "surprise": 0.0}

            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

            instruction = playground_developer_instruction() if mode == "valence" else ekman_developer_instruction()
            user_text = f"Text: {text[:1000]}"

            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": user_text},
                ],
                temperature=0.0,
                max_tokens=120 if mode == "ekman" else 80,
            )

            result_text = (resp.choices[0].message.content or "").strip()
            return _safe_json_loads(result_text, mode)

        except Exception as e:
            st.error(f"DeepSeek analysis failed: {e}")
            if st.session_state.debug_mode:
                st.exception(e)
            if mode == "valence":
                return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            else:
                return {"happiness": 0.0, "sadness": 0.0, "fear": 0.0, "anger": 0.0, 
                       "disgust": 0.0, "contempt": 0.0, "surprise": 0.0}

    # ---------------- OpenAI (GPT-5 nano via Responses API) --------------------
    def analyze_gpt5nano(self, text: str, api_key: str, mode: str = "valence") -> Dict[str, float]:
        """
        Analyze sentiment using OpenAI GPT-5 nano via Responses API.
        Supports both valence and Ekman emotions modes.
        """
        try:
            if not api_key:
                st.error("OpenAI API key required for GPT-5 nano")
                if mode == "valence":
                    return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
                else:
                    return {"happiness": 0.0, "sadness": 0.0, "fear": 0.0, "anger": 0.0, 
                           "disgust": 0.0, "contempt": 0.0, "surprise": 0.0}

            client = OpenAI(api_key=api_key)
            instruction = playground_developer_instruction() if mode == "valence" else ekman_developer_instruction()

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

            include_list = ["reasoning.encrypted_content"] if st.session_state.debug_mode else []

            resp = client.responses.create(
                model="gpt-5-nano",
                input=[
                    {"role": "developer", "content": [{"type": "input_text", "text": instruction}]},
                    {"role": "user", "content": [{"type": "input_text", "text": f"Text: {text[:1000]}"}]},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": schema_obj,
                    },
                    "verbosity": "low",
                },
                reasoning={"effort": "low"},
                tools=[],
                store=False,
                include=include_list,
                max_output_tokens=150 if mode == "ekman" else 120,
                temperature=0.0,
            )

            if st.session_state.debug_mode:
                st.code(resp.model_dump_json(indent=2) if hasattr(resp, "model_dump_json") else str(resp))

            out = _responses_text(resp).strip()
            if st.session_state.debug_mode:
                st.code(f"GPT-5 nano output_text:\n{out}")

            return _safe_json_loads(out, mode)

        except Exception as e:
            st.error(f"GPT-5 nano analysis failed: {e}")
            if st.session_state.debug_mode:
                st.exception(e)
            if mode == "valence":
                return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            else:
                return {"happiness": 0.0, "sadness": 0.0, "fear": 0.0, "anger": 0.0, 
                       "disgust": 0.0, "contempt": 0.0, "surprise": 0.0}

    # ---------------- Dispatcher ----------------
    def analyze_text(self, text: str, model: str, api_keys: Dict[str, str], mode: str = "valence") -> Dict[str, float]:
        """Dispatch to the selected model."""
        if model == "VADER":
            return self.analyze_vader(text)
        if model == "SiEBERT":
            return self.analyze_siebert(text, api_keys.get("Entresent_HF_API", ""))
        if model == "BART":
            return self.analyze_bart(text, api_keys.get("Entresent_HF_API", ""), mode)
        if model == "DeepSeek":
            return self.analyze_deepseek(text, api_keys.get("Entresent_DS_API", ""), mode)
        if model == "GPT-5 nano":
            return self.analyze_gpt5nano(text, api_keys.get("Entresent_OAI_API", ""), mode)
        
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

def create_results_dataframe(texts: List[str], results: Dict, truncate: bool = True) -> pd.DataFrame:
    """Create a formatted DataFrame from analysis results."""
    df_data: List[Dict[str, float]] = []

    for i, text in enumerate(texts):
        display_text = text[:100] + "..." if (truncate and len(text) > 100) else text
        row: Dict[str, float] = {"Text": display_text}
        for model, scores_list in results.items():
            if i < len(scores_list):
                for sentiment, value in scores_list[i].items():
                    if sentiment != "compound":  # Skip compound for cleaner display
                        row[f"{model}_{sentiment}"] = round(float(value), 3)
        df_data.append(row)

    return pd.DataFrame(df_data)

# --- App -----------------------------------------------------------------------
def main() -> None:
    st.title("🎭 Sentiment Analysis Toolbox")
    st.markdown("Analyze text sentiment using multiple state-of-the-art models.")

    # ---------------- Sidebar: Config ----------------
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Mode Selection
        st.subheader("Analysis Mode")
        analysis_mode = st.radio(
            "Select analysis type:",
            ["Valence", "Ekman Emotions"],
            help="Valence: positive/negative/neutral | Ekman: 7 basic emotions"
        )
        
        st.divider()
        
        # API Keys
        st.subheader("API Keys")
        
        if analysis_mode == "Valence":
            st.info(
                "ℹ️ VADER runs locally. SiEBERT & BART use Hugging Face API. "
                "DeepSeek and GPT-5 nano require their respective API keys."
            )
        else:  # Ekman Emotions
            st.info(
                "ℹ️ Ekman Emotions mode uses BART, DeepSeek, and GPT-5 nano only. "
                "All require API keys."
            )

        with st.expander("📚 How to get API keys"):
            st.markdown(
                """
- **Hugging Face API Key**: https://huggingface.co/settings/tokens  
- **DeepSeek API Key**: https://platform.deepseek.com/  
- **OpenAI API Key**: https://platform.openai.com/api-keys
"""
            )

        use_secrets = st.checkbox("Use Streamlit Secrets", value=True)

        api_keys: Dict[str, str] = {}
        if use_secrets:
            try:
                api_keys["Entresent_HF_API"] = st.secrets.get("Entresent_HF_API", "")
                api_keys["Entresent_DS_API"] = st.secrets.get("Entresent_DS_API", "")
                api_keys["Entresent_OAI_API"] = st.secrets.get("Entresent_OAI_API", "")
                st.success("Using API keys from Streamlit secrets")
            except Exception:
                st.warning("Secrets not configured. Please add API keys manually.")
                api_keys["Entresent_HF_API"] = st.text_input(
                    "Hugging Face API Key (required for SiEBERT/BART)", type="password"
                )
                api_keys["Entresent_DS_API"] = st.text_input("DeepSeek API Key", type="password")
                api_keys["Entresent_OAI_API"] = st.text_input("OpenAI API Key (for GPT-5 nano)", type="password")
        else:
            api_keys["Entresent_HF_API"] = st.text_input(
                "Hugging Face API Key (required for SiEBERT/BART)", type="password"
            )
            api_keys["Entresent_DS_API"] = st.text_input("DeepSeek API Key", type="password")
            api_keys["Entresent_OAI_API"] = st.text_input("OpenAI API Key (for GPT-5 nano)", type="password")

        st.divider()

        # Model Selection
        st.subheader("Model Selection")
        
        if analysis_mode == "Valence":
            available_models = ["VADER", "SiEBERT", "BART", "DeepSeek", "GPT-5 nano"]
            models_status = [
                "✅ VADER (ready)",
                "✅ SiEBERT" if api_keys.get("Entresent_HF_API") else "❌ SiEBERT (HF API key required)",
                "✅ BART" if api_keys.get("Entresent_HF_API") else "❌ BART (HF API key required)",
                "✅ DeepSeek" if api_keys.get("Entresent_DS_API") else "⚠️ DeepSeek (API key missing)",
                "✅ GPT-5 nano" if api_keys.get("Entresent_OAI_API") else "⚠️ GPT-5 nano (API key missing)",
            ]
        else:  # Ekman Emotions
            available_models = ["BART", "DeepSeek", "GPT-5 nano"]
            models_status = [
                "✅ BART" if api_keys.get("Entresent_HF_API") else "❌ BART (HF API key required)",
                "✅ DeepSeek" if api_keys.get("Entresent_DS_API") else "⚠️ DeepSeek (API key missing)",
                "✅ GPT-5 nano" if api_keys.get("Entresent_OAI_API") else "⚠️ GPT-5 nano (API key missing)",
            ]
        
        with st.expander("Model Availability", expanded=False):
            for status in models_status:
                st.write(status)

        benchmark_mode = st.checkbox("🏁 Benchmark Mode (Run all available models)", value=False)

        with st.expander("🔧 Advanced Settings", expanded=False):
            debug_mode = st.checkbox(
                "Enable Debug Mode", value=False, help="Show raw outputs and tracebacks for troubleshooting"
            )
            st.session_state.debug_mode = debug_mode

        if not benchmark_mode:
            selected_model = st.selectbox("Select Model", available_models)
            models_to_run = [selected_model]
        else:
            models_to_run = available_models
            st.info(f"Running all {len(models_to_run)} models")

    # ---------------- Main: Input & Settings ----------------
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📝 Input Text")
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
                st.success(f"Loaded {len(texts)} texts from file")

    with col2:
        st.subheader("🎯 Analysis Settings")
        st.info(f"**Mode:** {analysis_mode}")
        st.info(f"**Texts to analyze:** {len(texts)}")
        st.info(f"**Models to run:** {', '.join(models_to_run)}")

        missing_keys = []
        if not benchmark_mode:
            if selected_model in ["SiEBERT", "BART"] and not api_keys.get("Entresent_HF_API"):
                missing_keys.append(f"Hugging Face API key (required for {selected_model})")
            if selected_model == "DeepSeek" and not api_keys.get("Entresent_DS_API"):
                missing_keys.append("DeepSeek API key")
            if selected_model == "GPT-5 nano" and not api_keys.get("Entresent_OAI_API"):
                missing_keys.append("OpenAI API key")
        else:
            if analysis_mode == "Valence" and not api_keys.get("Entresent_HF_API"):
                missing_keys.append("Hugging Face API key (for SiEBERT & BART)")
            elif analysis_mode == "Ekman Emotions" and not api_keys.get("Entresent_HF_API"):
                missing_keys.append("Hugging Face API key (for BART)")
            if not api_keys.get("Entresent_DS_API"):
                missing_keys.append("DeepSeek API key")
            if not api_keys.get("Entresent_OAI_API"):
                missing_keys.append("OpenAI API key")

        if missing_keys:
            st.warning(f"⚠️ Missing: {', '.join(missing_keys)}")

        run_button = st.button("🚀 Analyze", type="primary", disabled=len(texts) == 0)

    # ---------------- Execution ----------------
    if run_button and texts:
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
            for t in texts:
                try:
                    scores = analyzer.analyze_text(t, model, api_keys, mode)
                except Exception as e:
                    st.error(f"{model} failed on a text: {e}")
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

    # ---------------- Results ----------------
    if st.session_state.results:
        texts, results, result_mode = st.session_state.results
        st.divider()
        st.subheader(f"📊 Results - {result_mode}")

        colA, colB, colC = st.columns([1, 1, 1])
        with colA:
            show_full_text = st.checkbox("Show full text", value=False)
        with colB:
            highlight_max = st.checkbox("Highlight max values", value=True)
        with colC:
            download_format = st.selectbox("Download format", ["CSV", "Excel"])

        df_results = create_results_dataframe(texts, results, truncate=not show_full_text)

        if highlight_max:
            score_cols = [c for c in df_results.columns if c != "Text"]
            styled = df_results.style.highlight_max(subset=score_cols, color="lightgreen", axis=1)
            st.dataframe(styled, use_container_width=True)
        else:
            st.dataframe(df_results, use_container_width=True)

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
                st.error(f"Excel export failed (install openpyxl). Falling back to CSV. Error: {e}")
                csv_data = df_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name=f"{result_mode.lower().replace(' ', '_')}_analysis_results.csv",
                    mime="text/csv",
                )

        # Summary (only meaningful when multiple models ran)
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
            st.dataframe(df_summary, use_container_width=True)
            
            # Create bar chart
            chart_data = df_summary.set_index("Model")
            st.bar_chart(chart_data)

if __name__ == "__main__":
    main()
