"""
Sentiment Analysis Toolbox
A comprehensive tool for analyzing text sentiment using multiple models
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
import json
import re
import requests
from typing import List, Dict, Tuple, Optional

# Import required libraries
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon', quiet=True)

from openai import OpenAI

# Page config
st.set_page_config(
    page_title="Sentiment Analysis Toolbox",
    page_icon="🎭",
    layout="wide"
)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None

class SentimentAnalyzer:
    """Main class for handling different sentiment analysis models"""
    
    def __init__(self):
        self.models_initialized = {}
        self.setup_models()
    
    def setup_models(self):
        """Initialize models that don't require API keys"""
        try:
            # Initialize VADER
            self.vader = SentimentIntensityAnalyzer()
            self.models_initialized['VADER'] = True
        except Exception as e:
            st.warning(f"VADER initialization failed: {e}")
            self.models_initialized['VADER'] = False
    
    def analyze_vader(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER"""
        try:
            scores = self.vader.polarity_scores(text)
            return {
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'compound': scores['compound']
            }
        except Exception as e:
            st.error(f"VADER analysis failed: {e}")
            return {'positive': 0, 'negative': 0, 'neutral': 0, 'compound': 0}
    
    @st.cache_resource
    def get_siebert_pipeline():
        """Load SiEBERT model (cached)"""
        try:
            return pipeline("sentiment-analysis", 
                          model="siebert/sentiment-roberta-large-english",
                          device=0 if torch.cuda.is_available() else -1)
        except Exception as e:
            st.error(f"SiEBERT loading failed: {e}")
            return None
    
    def analyze_siebert(self, text: str, api_key: str) -> Dict[str, float]:
        """Analyze sentiment using SiEBERT via Hugging Face Inference API"""
        try:
            if not api_key:
                st.error("Hugging Face API key required for SiEBERT")
                return {'positive': 0, 'negative': 0, 'neutral': 0}
            
            API_URL = "https://api-inference.huggingface.co/models/siebert/sentiment-roberta-large-english"
            headers = {"Authorization": f"Bearer {api_key}"}
            
            response = requests.post(API_URL, headers=headers, json={
                "inputs": text[:512],  # Truncate to model's max length
            })
            
            if response.status_code == 503:
                st.warning("SiEBERT model is loading. Please wait a moment and try again.")
                return {'positive': 0.5, 'negative': 0.5, 'neutral': 0.0}
            elif response.status_code != 200:
                raise Exception(f"API error: {response.text}")
            
            result = response.json()
            
            # Debug output if enabled
            if st.session_state.get('debug_mode', False):
                st.code(f"SiEBERT raw response: {json.dumps(result, indent=2)}")
            
            # Handle different response formats
            # The response is typically a list with nested lists
            if isinstance(result, list):
                if len(result) > 0:
                    if isinstance(result[0], list):
                        # Response format: [[{"label": "POSITIVE", "score": 0.99}]]
                        result_item = result[0][0] if len(result[0]) > 0 else {}
                    else:
                        # Response format: [{"label": "POSITIVE", "score": 0.99}]
                        result_item = result[0]
                else:
                    result_item = {}
            elif isinstance(result, dict):
                result_item = result
            else:
                result_item = {}
            
            # Extract label and score
            label = result_item.get('label', '').lower()
            score = result_item.get('score', 0.5)
            
            # Convert to our standard format
            sentiment_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
            if 'positive' in label:
                sentiment_scores['positive'] = score
                sentiment_scores['negative'] = 1 - score
                sentiment_scores['neutral'] = 0.0
            elif 'negative' in label:
                sentiment_scores['negative'] = score
                sentiment_scores['positive'] = 1 - score
                sentiment_scores['neutral'] = 0.0
            else:
                # If label is not recognized, distribute equally
                sentiment_scores['positive'] = 0.33
                sentiment_scores['negative'] = 0.33
                sentiment_scores['neutral'] = 0.34
            
            return sentiment_scores
            
        except Exception as e:
            st.error(f"SiEBERT analysis failed: {e}")
            if st.session_state.get('debug_mode', False):
                st.exception(e)
            return {'positive': 0, 'negative': 0, 'neutral': 0}
    
    def analyze_bart(self, text: str, api_key: str) -> Dict[str, float]:
        """Analyze sentiment using BART via Hugging Face Inference API"""
        try:
            if not api_key:
                st.error("Hugging Face API key required for BART")
                return {'positive': 0, 'negative': 0, 'neutral': 0}
            
            API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
            headers = {"Authorization": f"Bearer {api_key}"}
            
            response = requests.post(API_URL, headers=headers, json={
                "inputs": text[:512],  # Truncate to model's max length
                "parameters": {
                    "candidate_labels": ["positive", "negative", "neutral"],
                    "multi_label": True  # Allow independent scores
                }
            })
            
            if response.status_code == 503:
                st.warning("BART model is loading. Please wait a moment and try again.")
                return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
            elif response.status_code != 200:
                raise Exception(f"API error: {response.text}")
            
            result = response.json()
            
            # Debug output if enabled
            if st.session_state.get('debug_mode', False):
                st.code(f"BART raw response: {json.dumps(result, indent=2)}")
            
            # Parse the result based on the response structure
            if isinstance(result, dict) and 'labels' in result and 'scores' in result:
                scores = {}
                for label, score in zip(result['labels'], result['scores']):
                    scores[label] = score
            elif isinstance(result, list) and len(result) > 0:
                # Handle alternative list format
                first_result = result[0]
                if isinstance(first_result, dict) and 'labels' in first_result and 'scores' in first_result:
                    scores = {}
                    for label, score in zip(first_result['labels'], first_result['scores']):
                        scores[label] = score
                else:
                    scores = {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
            else:
                # Default scores if format is unexpected
                scores = {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
            
            # Ensure all sentiment types are present
            for sentiment in ['positive', 'negative', 'neutral']:
                if sentiment not in scores:
                    scores[sentiment] = 0.0
            
            return scores
            
        except Exception as e:
            st.error(f"BART analysis failed: {e}")
            if st.session_state.get('debug_mode', False):
                st.exception(e)
            return {'positive': 0, 'negative': 0, 'neutral': 0}
    
    def analyze_deepseek(self, text: str, api_key: str) -> Dict[str, float]:
        """Analyze sentiment using DeepSeek API"""
        try:
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
            
            prompt = """Analyze the sentiment of the following text passage. 
            Classify it into three independent sentiment categories: positive, negative, and neutral. 
            Assign each category a separate confidence value (from 0 to 1) that is independent and not evaluated as a probability distribution with a sum of 1. 
            Return ONLY a JSON object with the format: {"positive": 0.X, "negative": 0.Y, "neutral": 0.Z}"""
            
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis assistant. Return only JSON."},
                    {"role": "user", "content": f"{prompt}\n\nText: {text[:1000]}"}
                ],
                temperature=0.1
            )
            
            # Parse the response
            result_text = response.choices[0].message.content
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
                return scores
            else:
                return {'positive': 0, 'negative': 0, 'neutral': 0}
                
        except Exception as e:
            st.error(f"DeepSeek analysis failed: {e}")
            return {'positive': 0, 'negative': 0, 'neutral': 0}
    
    def analyze_gpt5nano_responses(self, text: str, api_key: str) -> Dict[str, float]:
    if not api_key:
        st.error("OpenAI API key required for GPT-5 nano")
        return {'positive': 0, 'negative': 0, 'neutral': 0}

    client = OpenAI(api_key=api_key)

    resp = client.responses.create(
        model="gpt-5-nano",
        input=[
            {"role": "system", "content": "You are a sentiment analysis assistant. Return only valid JSON."},
            {"role": "user", "content":
                ("Analyze the sentiment of the following text. "
                 'Return ONLY {"positive": x, "negative": y, "neutral": z} with x,y,z ∈ [0,1]. '
                 f"\n\nText: {text[:1000]}")
            },
        ],
        response_format={"type": "json_object"},
    )

    # Extract plain text from the response in a version-agnostic way
    text_chunks = []
    for item in getattr(resp, "output", []):
        if getattr(item, "type", "") == "message":
            for c in getattr(item, "content", []):
                if getattr(c, "type", "") in ("output_text", "text"):
                    text_chunks.append(getattr(c, "text", ""))
    result_text = "".join(text_chunks) or getattr(resp, "output_text", "")

    try:
        scores = json.loads(result_text)
        return {
            'positive': float(scores.get('positive', 0.0)),
            'negative': float(scores.get('negative', 0.0)),
            'neutral':  float(scores.get('neutral',  0.0)),
        }
    except Exception as e:
        st.error(f"GPT-5 nano parse error: {e}")
        return {'positive': 0, 'negative': 0, 'neutral': 0}
    
    def analyze_text(self, text: str, model: str, api_keys: Dict[str, str]) -> Dict[str, float]:
        """Main method to analyze text with specified model"""
        if model == "VADER":
            return self.analyze_vader(text)
        elif model == "SiEBERT":
            return self.analyze_siebert(text, api_keys.get('Entresent_HF_API', ''))
        elif model == "BART":
            return self.analyze_bart(text, api_keys.get('Entresent_HF_API', ''))
        elif model == "DeepSeek":
            return self.analyze_deepseek(text, api_keys.get('Entresent_DS_API', ''))
        elif model == "GPT-5 nano":
            return self.analyze_gpt5nano(text, api_keys.get('Entresent_OAI_API', ''))
        else:
            return {'positive': 0, 'negative': 0, 'neutral': 0}

def load_texts_from_file(uploaded_file) -> List[str]:
    """Load texts from uploaded CSV or Excel file"""
    texts = []
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format")
            return texts
        
        # Assume first column contains texts or look for 'text' column
        if 'text' in df.columns:
            texts = df['text'].dropna().tolist()
        elif 'Text' in df.columns:
            texts = df['Text'].dropna().tolist()
        else:
            # Use first column
            texts = df.iloc[:, 0].dropna().tolist()
        
        return [str(text) for text in texts]
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return []

def create_results_dataframe(texts: List[str], results: Dict) -> pd.DataFrame:
    """Create a formatted DataFrame from analysis results"""
    df_data = []
    
    for i, text in enumerate(texts):
        row = {'Text': text[:100] + '...' if len(text) > 100 else text}
        for model, scores in results.items():
            if i < len(scores):
                for sentiment, value in scores[i].items():
                    if sentiment != 'compound':  # Skip compound score for cleaner display
                        row[f"{model}_{sentiment}"] = round(value, 3)
        df_data.append(row)
    
    return pd.DataFrame(df_data)

def main():
    """Main Streamlit application"""
    st.title("🎭 Sentiment Analysis Toolbox")
    st.markdown("Analyze text sentiment using multiple state-of-the-art models")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys section
        st.subheader("API Keys")
        st.info("ℹ️ VADER runs locally. SiEBERT and BART require Hugging Face API. DeepSeek and GPT-5 nano require their respective API keys.")
        
        with st.expander("📚 How to get API keys"):
            st.markdown("""
            **Hugging Face API Key** (required for SiEBERT & BART):
            1. Go to https://huggingface.co/settings/tokens
            2. Create a free account if needed
            3. Click "New token" → Give it a name → Copy the token
            
            **DeepSeek API Key**:
            1. Visit https://platform.deepseek.com/
            2. Sign up and go to API Keys section
            3. Create and copy your API key
            
            **OpenAI API Key** (for GPT-5 nano):
            1. Go to https://platform.openai.com/api-keys
            2. Sign in or create an account
            3. Create new secret key and copy it
            """)
        
        use_secrets = st.checkbox("Use Streamlit Secrets", value=True)
        
        api_keys = {}
        if use_secrets:
            try:
                api_keys['Entresent_HF_API'] = st.secrets.get("Entresent_HF_API", "")
                api_keys['Entresent_DS_API'] = st.secrets.get("Entresent_DS_API", "")
                api_keys['Entresent_OAI_API'] = st.secrets.get("Entresent_OAI_API", "")
                st.success("Using API keys from Streamlit secrets")
            except Exception:
                st.warning("Secrets not configured. Please add API keys manually.")
                api_keys['Entresent_HF_API'] = st.text_input("Hugging Face API Key (required for SiEBERT/BART)", type="password")
                api_keys['Entresent_DS_API'] = st.text_input("DeepSeek API Key", type="password")
                api_keys['Entresent_OAI_API'] = st.text_input("OpenAI API Key (for GPT-5 nano)", type="password")
        else:
            api_keys['Entresent_HF_API'] = st.text_input("Hugging Face API Key (required for SiEBERT/BART)", type="password")
            api_keys['Entresent_DS_API'] = st.text_input("DeepSeek API Key", type="password")
            api_keys['Entresent_OAI_API'] = st.text_input("OpenAI API Key (for GPT-5 nano)", type="password")
        
        st.divider()
        
        # Model selection
        st.subheader("Model Selection")
        available_models = ["VADER", "SiEBERT", "BART", "DeepSeek", "GPT-5 nano"]
        
        # Check which models are available based on API keys
        models_status = []
        models_status.append("✅ VADER (ready)")
        models_status.append("✅ SiEBERT" if api_keys.get('Entresent_HF_API') else "❌ SiEBERT (HF API key required)")
        models_status.append("✅ BART" if api_keys.get('Entresent_HF_API') else "❌ BART (HF API key required)")
        models_status.append("✅ DeepSeek" if api_keys.get('Entresent_DS_API') else "⚠️ DeepSeek (API key missing)")
        models_status.append("✅ GPT-5 nano" if api_keys.get('Entresent_OAI_API') else "⚠️ GPT-5 nano (API key missing)")
        
        with st.expander("Model Availability", expanded=False):
            for status in models_status:
                st.write(status)
        
        benchmark_mode = st.checkbox("🏁 Benchmark Mode (Run all models)", value=False)
        
        # Debug mode toggle (optional)
        with st.expander("🔧 Advanced Settings", expanded=False):
            debug_mode = st.checkbox("Enable Debug Mode", value=False, help="Show detailed API responses for troubleshooting")
            st.session_state.debug_mode = debug_mode
        
        if not benchmark_mode:
            selected_model = st.selectbox("Select Model", available_models)
            models_to_run = [selected_model]
        else:
            models_to_run = available_models
            st.info(f"Running all {len(models_to_run)} models")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Input Text")
        input_method = st.radio("Input Method", ["Text Area", "File Upload"])
        
        texts = []
        if input_method == "Text Area":
            text_input = st.text_area(
                "Enter texts (one per line)",
                height=200,
                placeholder="Enter your first text here\nEnter your second text here\n..."
            )
            if text_input:
                texts = [t.strip() for t in text_input.split('\n') if t.strip()]
        else:
            uploaded_file = st.file_uploader(
                "Upload CSV or Excel file",
                type=['csv', 'xlsx', 'xls']
            )
            if uploaded_file:
                texts = load_texts_from_file(uploaded_file)
                st.success(f"Loaded {len(texts)} texts from file")
    
    with col2:
        st.subheader("🎯 Analysis Settings")
        st.info(f"**Texts to analyze:** {len(texts)}")
        st.info(f"**Models to run:** {', '.join(models_to_run)}")
        
        # Check if required API keys are present for selected models
        missing_keys = []
        if not benchmark_mode:
            if selected_model in ["SiEBERT", "BART"] and not api_keys.get('Entresent_HF_API'):
                missing_keys.append("Hugging Face API key (required for " + selected_model + ")")
            elif selected_model == "DeepSeek" and not api_keys.get('Entresent_DS_API'):
                missing_keys.append("DeepSeek API key")
            elif selected_model == "GPT-5 nano" and not api_keys.get('Entresent_OAI_API'):
                missing_keys.append("OpenAI API key")
        else:
            if not api_keys.get('Entresent_HF_API'):
                missing_keys.append("Hugging Face API key (for SiEBERT & BART)")
            if not api_keys.get('Entresent_DS_API'):
                missing_keys.append("DeepSeek API key")
            if not api_keys.get('Entresent_OAI_API'):
                missing_keys.append("OpenAI API key")
        
        if missing_keys:
            st.warning(f"⚠️ Missing: {', '.join(missing_keys)}")
        
        if st.button("🚀 Analyze Sentiment", type="primary", disabled=len(texts) == 0):
            if len(texts) > 0:
                analyzer = SentimentAnalyzer()
                results = {}
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_operations = len(models_to_run) * len(texts)
                current_operation = 0
                
                for model in models_to_run:
                    status_text.text(f"Running {model}...")
                    model_results = []
                    
                    for text in texts:
                        scores = analyzer.analyze_text(text, model, api_keys)
                        model_results.append(scores)
                        
                        current_operation += 1
                        progress_bar.progress(current_operation / total_operations)
                    
                    results[model] = model_results
                
                progress_bar.empty()
                status_text.empty()
                st.session_state.results = (texts, results)
                st.success("✅ Analysis complete!")
    
    # Results section
    if st.session_state.results:
        texts, results = st.session_state.results
        st.divider()
        st.subheader("📊 Results")
        
        # Create and display results DataFrame
        df_results = create_results_dataframe(texts, results)
        
        # Display options
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            show_full_text = st.checkbox("Show full text", value=False)
        with col2:
            highlight_max = st.checkbox("Highlight max values", value=True)
        with col3:
            download_format = st.selectbox("Download format", ["CSV", "Excel"])
        
        # Apply styling if requested
        if highlight_max:
            # Get all score columns (excluding 'Text' column)
            score_columns = [col for col in df_results.columns if col != 'Text']
            styled_df = df_results.style.highlight_max(
                subset=score_columns,
                color='lightgreen',
                axis=1
            )
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.dataframe(df_results, use_container_width=True)
        
        # Download button
        if download_format == "CSV":
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv"
            )
        else:
            # For Excel download, we need to use a different approach
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_results.to_excel(writer, index=False, sheet_name='Results')
            excel_data = output.getvalue()
            st.download_button(
                label="📥 Download Results as Excel",
                data=excel_data,
                file_name="sentiment_analysis_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Summary statistics
        if len(results) > 1:  # Only show if benchmark mode
            st.divider()
            st.subheader("📈 Summary Statistics")
            
            # Calculate average scores per model
            summary_data = []
            for model, scores in results.items():
                avg_pos = np.mean([s.get('positive', 0) for s in scores])
                avg_neg = np.mean([s.get('negative', 0) for s in scores])
                avg_neu = np.mean([s.get('neutral', 0) for s in scores])
                summary_data.append({
                    'Model': model,
                    'Avg Positive': round(avg_pos, 3),
                    'Avg Negative': round(avg_neg, 3),
                    'Avg Neutral': round(avg_neu, 3)
                })
            
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True)
            
            # Visualization
            st.bar_chart(df_summary.set_index('Model'))

if __name__ == "__main__":
    main()
