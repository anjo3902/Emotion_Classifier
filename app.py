import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ------------------- NLTK Downloads -------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ------------------- Text Preprocessing -------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# ------------------- Load Pickle Files -------------------
@st.cache_resource
def load_models():
    with open('svm_model_bert_best.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('bert_encoder.pkl', 'rb') as f:
        bert_model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    with open('label_to_emotion.pkl', 'rb') as f:
        label_to_emotion = pickle.load(f)
    return model, bert_model, label_encoder, label_to_emotion

# ------------------- Streamlit App -------------------
st.set_page_config(page_title="Emotion Classifier", layout="centered")
st.title("🧠 Emotion Classifier using BERT + SVM")
st.markdown("Enter a sentence to detect the underlying emotion.")

# Load models
try:
    model, bert_model, label_encoder, label_to_emotion = load_models()
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# Input from user
text_input = st.text_area("Enter your text here", height=150)

if st.button("Predict Emotion"):
    if not text_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        # Preprocess
        clean_text = preprocess_text(text_input)
        embedding = bert_model.encode([clean_text])
        
        # Predict
        prediction = model.predict(embedding)[0]
        emotion = label_to_emotion[prediction]

        st.success(f"✅ **Predicted Emotion:** `{emotion}`")
        st.markdown("---")
        st.markdown(f"**Original Text:** _{text_input}_")
        st.markdown(f"**Cleaned Text:** _{clean_text}_")

# Optional footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit, BERT, and SVM")
