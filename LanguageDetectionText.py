import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline, M2M100ForConditionalGeneration, \
    M2M100Tokenizer

# Dictionary to map language codes to full names
language_map = {
    'ar': 'Arabic',
    'bg': 'Bulgarian',
    'de': 'German',
    'el': 'Modern Greek',
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'hi': 'Hindi',
    'it': 'Italian',
    'ja': 'Japanese',
    'nl': 'Dutch',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'sw': 'Swahili',
    'th': 'Thai',
    'tr': 'Turkish',
    'ur': 'Urdu',
    'vi': 'Vietnamese',
    'zh': 'Chinese'
}


# Cache the model loading functions using Streamlit's caching mechanism
@st.cache_resource
def load_language_detection_model(model_path=None):
    """Load and cache the language detection model."""
    model = AutoModelForSequenceClassification.from_pretrained(
        "papluca/xlm-roberta-base-language-detection", cache_dir=model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "papluca/xlm-roberta-base-language-detection", cache_dir=model_path
    )
    return pipeline("text-classification", model=model, tokenizer=tokenizer)


@st.cache_resource
def load_translation_model(model_path=None):
    """Load and cache the translation model."""
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M", cache_dir=model_path)
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", cache_dir=model_path)
    return model, tokenizer


class LanguageDetector:
    def __init__(self, model_path=None, language_map=None):
        self.model_path = model_path
        self.language_map = language_map
        self.language_detector = load_language_detection_model(self.model_path)

    def detect_language(self, text):
        """Detect the language of the given text."""
        result = self.language_detector(text)
        language_code = result[0]['label']
        return self.language_map.get(language_code, language_code)


class Translator:
    def __init__(self, model_path=None):
        self.translation_model, self.tokenizer = load_translation_model(model_path)

    def translate_to_english(self, text, source_language):
        """Translate the given text to English."""
        self.tokenizer.src_lang = source_language
        encoded_text = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.translation_model.generate(**encoded_text,
                                                           forced_bos_token_id=self.tokenizer.get_lang_id("en"))
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]


# Initialize the detector and translator
detector = LanguageDetector(model_path="D:\PythonProjects\Text Detection\models\\xlm-roberta",
                            language_map=language_map)
translator = Translator(model_path="D:\PythonProjects\Text Detection\models\\m2m100")

# Streamlit app
st.title("🌍 Language Detection and Translation App")
st.subheader("Detect the language and translate text to English in real-time")

# Instructions
st.markdown("### Instructions:")
st.markdown(
    "Enter or paste text in the box below and click **Detect & Translate** to see the results. The text will be detected and translated into English!")

# Text input
user_input = st.text_area("Enter text to detect language and translate", placeholder="Type here...", height=150)

# Real-time detection and translation
if st.button("Detect & Translate"):
    if user_input.strip():
        detected_language = detector.detect_language(user_input)
        st.success(f"Detected Language: **{detected_language}**")

        if detected_language.lower() != 'english':
            source_language_code = list(language_map.keys())[list(language_map.values()).index(detected_language)]
            translation = translator.translate_to_english(user_input, source_language_code)
            st.info(f"Translation to English: **{translation}**")
        else:
            st.info("No translation needed, the text is already in English.")
    else:
        st.warning("Please enter some text.")

# Example input suggestions
st.sidebar.header("Language Input Examples")
st.sidebar.markdown("- **Spanish**: Mi hermosa amiga")
st.sidebar.markdown("- **French**: Bonjour, comment ça va?")
st.sidebar.markdown("- **German**: Guten Morgen!")
st.sidebar.markdown("- **Chinese**: 你好")
st.sidebar.markdown("- **Arabic**: مرحبا")
