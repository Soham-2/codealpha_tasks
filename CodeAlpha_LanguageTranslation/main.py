import streamlit as st
from googletrans import Translator

translator = Translator()

def translate_text(target_language, text):
    """Translates text into the target language using googletrans."""
    translated = translator.translate(text, dest=target_language)
    return translated.text, translated.src

st.title("Language Translation Tool")

text_to_translate = st.text_area("Enter text to translate:", "Hello, how are you?")

languages = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Chinese (Simplified)": "zh-CN",
}

col1, col2 = st.columns(2)

with col1:
    source_language_display = st.selectbox("Source Language:", list(languages.keys()), index=0)
    source_language_code = languages[source_language_display]

with col2:
    target_language_display = st.selectbox("Target Language:", list(languages.keys()), index=1)
    target_language_code = languages[target_language_display]

if st.button("Translate"):
    if text_to_translate:
        translated_text, detected_source_language = translate_text(target_language_code, text_to_translate)
        st.subheader("Translated Text (Detected Source Language: " + detected_source_language + "):")
        st.write(translated_text)
    else:
        st.warning("Please enter some text to translate.")

st.markdown("---")
st.markdown("Developed with Streamlit and Google Cloud Translation API")

