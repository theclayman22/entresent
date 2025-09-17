"""
Sentiment Analysis Toolbox
A comprehensive tool for analyzing text sentiment using multiple models.
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

# --- Utilities -----------------------------------------------------------------
def _clip01(x: float) -> float:
    try:
        return float(max(0.0, min(1.0, x)))
    except Exception:
        return 0.0


def _safe_json_loads(s: str) -> Dict[str, float]:
    """Parse JSON object safely, returning a dict with positive/negative/neutral ∈ [0,1]."""
    try:
        data = json.loads(s)
    except Exception:
        # Last resort: try to extract the first {} block
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
    return {
        "positive": _clip01(data.get("positive", 0.0)),
        "negative": _clip01(data.get("negative", 0.0)),
        "neutral": _clip01(data.get("neutral", 0.0)),
    }


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
        """Analyze sentiment using VADER."""
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
        """Analyze sentiment using SiEBERT via Hugging Face Inference API."""
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

            # HF output formats can vary: [[{label,score}]] or [{label,score}]
            result_item = {}
            if isinstance(result, list) and result:
                result_item = result[0][0] if isinstance(result[0], list) and result[0] else result[0]
            elif isinstance(result, dict):
                result_item = result

            label = str(result_item.get("label", "")).lower()
            score = float(result_item.get("score", 0.5))
            score = _clip01(score)

            if "positive" in label:
                return {"positive": score, "negative": _clip01(1 - score), "neutral": 0.0}
            if "negative" in label:
                return {"negative": score, "positive": _clip01(1 - score), "neutral": 0.0}

            # Fallback (unknown label)
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}

        except Exception as e:
            st.error(f"SiEBERT analysis failed: {e}")
            if st.session_state.debug_mode:
                st.exception(e)
            return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    def analyze_bart(self, text: str, api_key: str) -> Dict[str, float]:
        """Analyze sentiment using BART (MNLI) via Hugging Face Inference API."""
        try:
            if not api_key:
                st.error("Hugging Face API key required for BART")
                return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

            api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
            headers = {"Authorization": f"Bearer {api_key}"}
            payload = {
                "inputs": text[:512],
                "parameters": {"candidate_labels": ["positive", "negative", "neutral"], "multi_label": True},
            }
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)

            if response.status_code == 503:
                st.warning("BART model is loading on HF. Try again in a few seconds.")
                return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
            response.raise_for_status()

            result = response.json()
            if st.session_state.debug_mode:
                st.code(f"BART raw response:\n{json.dumps(result, indent=2)}")

            scores_map: Dict[str, float] = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

            # Either {"labels":[...],"scores":[...]} or [{"labels":[...],"scores":[...]}]
            blob = result[0] if isinstance(result, list) and result else result
            if isinstance(blob, dict) and "labels" in blob and "scores" in blob:
                for label, score in zip(blob["labels"], blob["scores"]):
                    scores_map[label] = _clip01(float(score))

            return scores_map

        except Exception as e:
            st.error(f"BART analysis failed: {e}")
            if st.session_state.debug_mode:
                st.exception(e)
            return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    # ---------------- DeepSeek (OpenAI-compatible) ----------------
    def analyze_deepseek(self, text: str, api_key: str) -> Dict[str, float]:
        """Analyze sentiment using DeepSeek API (OpenAI-compatible)."""
        try:
            if not api_key:
                st.error("DeepSeek API key required for DeepSeek")
                return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

            prompt = (
                "Analyze the sentiment of the following text passage. "
                'Return ONLY a JSON object: {"positive": 0.x, "negative": 0.y, "neutral": 0.z} '
                "with each value in [0,1]. No prose.\n\n"
                f"Text: {text[:1000]}"
            )

            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis assistant. Return only JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=60,
            )

            result_text = resp.choices[0].message.content or "{}"
            return _safe_json_loads(result_text)

        except Exception as e:
            st.error(f"DeepSeek analysis failed: {e}")
            if st.session_state.debug_mode:
                st.exception(e)
            return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    # ---------------- OpenAI (GPT-5 nano) ----------------
    def analyze_gpt5nano(self, text: str, api_key: str) -> Dict[str, float]:
        """
        Analyze sentiment using OpenAI GPT-5 nano via Chat Completions.
        Uses response_format=json_object to guarantee valid JSON.
        """
        try:
            if not api_key:
                st.error("OpenAI API key required for GPT-5 nano")
                return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

            client = OpenAI(api_key=api_key)

            user_prompt = (
                "Analyze the sentiment of the following text. "
                'Return ONLY a JSON object {"positive": x, "negative": y, "neutral": z} '
                "with x,y,z in [0,1]. No explanation, no code fences.\n\n"
                f"Text: {text[:1000]}"
            )

            completion = client.chat.completions.create(
                model="gpt-5-nano",  # ✅ correct model ID
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis assistant. Return only valid JSON."},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=60,
            )

            payload = completion.choices[0].message.content or "{}"
            return _safe_json_loads(payload)

        except Exception as e:
            st.error(f"GPT-5 nano analysis failed: {e}")
            if st.session_state.debug_mode:
                st.exception(e)
            return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    # ---------------- Dispatcher ----------------
    def analyze_text(self, text: str, model: str, api_keys: Dict[str, str]) -> Dict[str, float]:
        """Dispatch to the selected model."""
        if model == "VADER":
            return self.analyze_vader(text)
        if model == "SiEBERT":
            return self.analyze_siebert(text, api_keys.get("Entresent_HF_API", ""))
        if model == "BART":
            return self.analyze_bart(text, api_keys.get("Entresent_HF_API", ""))
        if model == "DeepSeek":
            return self.analyze_deepseek(text, api_keys.get("Entresent_DS_API", ""))
        if model == "GPT-5 nano":
            return self.analyze_gpt5nano(text, api_keys.get("Entresent_OAI_API", ""))
        return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}


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
        st.subheader("API Keys")
        st.info(
            "ℹ️ VADER runs locally. SiEBERT & BART use Hugging Face Inference API. "
            "DeepSeek and GPT-5 nano require their respective API keys."
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

        st.subheader("Model Selection")
        available_models = ["VADER", "SiEBERT", "BART", "DeepSeek", "GPT-5 nano"]

        models_status = [
            "✅ VADER (ready)",
            "✅ SiEBERT" if api_keys.get("Entresent_HF_API") else "❌ SiEBERT (HF API key required)",
            "✅ BART" if api_keys.get("Entresent_HF_API") else "❌ BART (HF API key required)",
            "✅ DeepSeek" if api_keys.get("Entresent_DS_API") else "⚠️ DeepSeek (API key missing)",
            "✅ GPT-5 nano" if api_keys.get("Entresent_OAI_API") else "⚠️ GPT-5 nano (API key missing)",
        ]
        with st.expander("Model Availability", expanded=False):
            for status in models_status:
                st.write(status)

        benchmark_mode = st.checkbox("🏁 Benchmark Mode (Run all models)", value=False)

        with st.expander("🔧 Advanced Settings", expanded=False):
            debug_mode = st.checkbox(
                "Enable Debug Mode", value=False, help="Show detailed API responses for troubleshooting"
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
            if not api_keys.get("Entresent_HF_API"):
                missing_keys.append("Hugging Face API key (for SiEBERT & BART)")
            if not api_keys.get("Entresent_DS_API"):
                missing_keys.append("DeepSeek API key")
            if not api_keys.get("Entresent_OAI_API"):
                missing_keys.append("OpenAI API key")

        if missing_keys:
            st.warning(f"⚠️ Missing: {', '.join(missing_keys)}")

        run_button = st.button("🚀 Analyze Sentiment", type="primary", disabled=len(texts) == 0)

    # ---------------- Execution ----------------
    if run_button and texts:
        analyzer = SentimentAnalyzer()
        results: Dict[str, List[Dict[str, float]]] = {}

        total_ops = max(1, len(models_to_run) * len(texts))
        progress = st.progress(0)
        status = st.empty()
        op = 0

        for model in models_to_run:
            status.text(f"Running {model}...")
            model_results: List[Dict[str, float]] = []
            for t in texts:
                try:
                    scores = analyzer.analyze_text(t, model, api_keys)
                except Exception as e:
                    st.error(f"{model} failed on a text: {e}")
                    if st.session_state.debug_mode:
                        st.exception(e)
                    scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
                model_results.append(scores)
                op += 1
                progress.progress(op / total_ops)
            results[model] = model_results

        progress.empty()
        status.empty()
        st.session_state.results = (texts, results)
        st.success("✅ Analysis complete!")

    # ---------------- Results ----------------
    if st.session_state.results:
        texts, results = st.session_state.results
        st.divider()
        st.subheader("📊 Results")

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
                file_name="sentiment_analysis_results.csv",
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
                    file_name="sentiment_analysis_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as e:
                st.error(f"Excel export failed (install openpyxl). Falling back to CSV. Error: {e}")
                csv_data = df_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv",
                )

        # Summary (only meaningful when multiple models ran)
        if len(results) > 1:
            st.divider()
            st.subheader("📈 Summary Statistics")
            summary_rows = []
            for model, scores in results.items():
                avg_pos = float(np.mean([s.get("positive", 0.0) for s in scores])) if scores else 0.0
                avg_neg = float(np.mean([s.get("negative", 0.0) for s in scores])) if scores else 0.0
                avg_neu = float(np.mean([s.get("neutral", 0.0) for s in scores])) if scores else 0.0
                summary_rows.append(
                    {"Model": model, "Avg Positive": round(avg_pos, 3), "Avg Negative": round(avg_neg, 3), "Avg Neutral": round(avg_neu, 3)}
                )
            df_summary = pd.DataFrame(summary_rows)
            st.dataframe(df_summary, use_container_width=True)
            st.bar_chart(df_summary.set_index("Model"))


if __name__ == "__main__":
    main()
