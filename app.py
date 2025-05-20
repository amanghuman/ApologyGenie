import streamlit as st
import google.generativeai as genai
import logging
from functools import lru_cache
from datetime import datetime
from transformers import pipeline

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# Configure app settings
CACHE_SIZE = 100
MAX_RETRIES = 2
GPT2_MODEL = "gpt2"

def configure_models():
    """Initialize both models with error handling"""
    models = {
        'gemini': None,
        'gpt2': pipeline('text-generation', model=GPT2_MODEL)
    }
    
    try:
        genai.configure(api_key=st.secrets.gemini.api_key)
        models['gemini'] = genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        st.error("Failed to configure Gemini API. Using GPT-2 as fallback.")
        logging.error(f"Gemini config error: {str(e)}")
    
    return models

@lru_cache(maxsize=CACHE_SIZE)
def generate_with_gemini(situation: str, tone: str) -> str:
    """Generate apology using Gemini"""
    model = st.session_state.models['gemini']
    prompt = f"Write a {tone} apology note for: {situation}. Keep it under 3 sentences."
    
    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.7 if tone == "funny" else 0.3,
                    "max_output_tokens": 100
                }
            )
            return response.text
        except genai.APIError as e:
            if "quota" in str(e).lower():
                st.session_state.gemini_over_limit = True
                raise e
            logging.warning(f"API error (attempt {attempt+1}): {str(e)}")
            if attempt == MAX_RETRIES - 1:
                raise e
    return ""

def generate_with_gpt2(situation: str, tone: str) -> str:
    """Fallback to GPT-2 when Gemini fails"""
    try:
        prompt = f"Write a {tone} apology for {situation}:"
        response = st.session_state.models['gpt2'](
            prompt,
            max_length=100,
            num_return_sequences=1
        )
        return response[0]['generated_text'].replace(prompt, "").strip()
    except Exception as e:
        logging.error(f"GPT-2 error: {str(e)}")
        return "Apology generation failed. Please try again."

def handle_feedback(entry_id, is_positive):
    """Update feedback in session state"""
    for idx, entry in enumerate(st.session_state.history):
        if entry['id'] == entry_id:
            st.session_state.history[idx]['feedback'] = is_positive
            break

def main():
    st.title("🤖 AI Apology Generator")
    st.markdown("### Powered by Gemini 1.5 Flash with GPT-2 Fallback")
    
    # Initialize models and session state
    if 'models' not in st.session_state:
        st.session_state.models = configure_models()
    if 'gemini_over_limit' not in st.session_state:
        st.session_state.gemini_over_limit = False
    if 'history' not in st.session_state:
        st.session_state.history = []

    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. Describe your situation
        2. Choose a tone
        3. Generate and rate the apology!
        """)

    # User inputs
    situation = st.text_input("What happened? ✍️", placeholder="Describe the situation...")
    tone = st.selectbox("Choose tone 🎭", ["formal", "friendly", "funny", "sincere"])

    if st.button("Generate Apology ✨", type="primary"):
        if not situation.strip():
            st.error("Please describe your situation first!")
            return

        apology = ""
        model_used = ""
        
        try:
            if not st.session_state.gemini_over_limit and st.session_state.models['gemini']:
                with st.spinner("Crafting the perfect apology with Gemini..."):
                    apology = generate_with_gemini(situation.strip().lower(), tone.lower())
                    model_used = "Gemini"
        except Exception as e:
            if "quota" in str(e).lower():
                st.warning("Gemini limit reached. Switching to GPT-2...")
                st.session_state.gemini_over_limit = True

        if not apology and st.session_state.models['gpt2']:
            with st.spinner("Generating with GPT-2..."):
                apology = generate_with_gpt2(situation.strip().lower(), tone.lower())
                model_used = "GPT-2"

        if apology:
            entry_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
            st.session_state.history.append({
                "id": entry_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "situation": situation,
                "tone": tone,
                "apology": apology,
                "model": model_used,
                "feedback": None
            })

            st.subheader(f"Your {model_used}-Generated Apology:")
            st.write(apology)
            st.download_button(
                label="Copy to Clipboard 📋",
                data=apology,
                file_name="apology.txt",
                mime="text/plain"
            )

            # Feedback buttons
            col1, col2 = st.columns(2)
            with col1:
                st.button(
                    "👍 Good Response",
                    on_click=handle_feedback,
                    args=(entry_id, True),
                    key=f"good_{entry_id}"
                )
            with col2:
                st.button(
                    "👎 Needs Improvement",
                    on_click=handle_feedback,
                    args=(entry_id, False),
                    key=f"bad_{entry_id}"
                )
        else:
            st.error("Failed to generate apology. Please try again.")

    # Display history
    if st.session_state.history:
        st.divider()
        st.subheader("Apology History 📚")
        for entry in reversed(st.session_state.history):
            with st.expander(f"{entry['timestamp']} - {entry['situation']} ({entry['model']})"):
                st.caption(f"Tone: {entry['tone'].capitalize()}")
                st.write(entry['apology'])
                
                # Show feedback status
                if entry['feedback'] is not None:
                    status = "👍" if entry['feedback'] else "👎"
                    st.markdown(f"**Your Feedback:** {status}")
                else:
                    st.caption("No feedback submitted yet")

    # Footer
    st.markdown("---")
    st.caption("""
    🔒 Note:
    - Gemini: 1,500 requests/day limit
    - Feedback helps improve the service!
    - GPT-2 responses may be less polished
    """)

if __name__ == "__main__":
    main()