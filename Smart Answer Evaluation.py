import streamlit as st
import pytesseract
from PIL import Image
import nltk
import numpy as np
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.util import bigrams
from nltk.stem import WordNetLemmatizer
import language_tool_python
import io

# Tesseract path (update if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# OCR text extractor from PDF or image (without Poppler)
def extract_text(file):
    if file.type == "application/pdf":
        text = ""
        pdf_data = file.read()
        pdf_doc = fitz.open(stream=pdf_data, filetype="pdf")
        for page_index in range(len(pdf_doc)):
            pix = pdf_doc[page_index].get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text += pytesseract.image_to_string(img)
        return text
    elif file.type.startswith("image/"):
        image = Image.open(file)
        return pytesseract.image_to_string(image)
    else:
        return ""

# Preprocess text
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    return [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]

# Similarity metrics
def compute_cosine_similarity(text1, text2):
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform([text1, text2])
    return cosine_similarity(matrix[0:1], matrix[1:2])[0][0]

def jaccard_similarity(set1, set2):
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0

def bigram_similarity(tokens1, tokens2):
    b1, b2 = set(bigrams(tokens1)), set(bigrams(tokens2))
    return len(b1 & b2) / len(b1 | b2) if b1 | b2 else 0

def synonym_similarity(tokens1, tokens2):
    count = 0
    for word in tokens1:
        synonyms = set(lemma.name() for syn in wordnet.synsets(word) for lemma in syn.lemmas())
        if any(token in synonyms for token in tokens2):
            count += 1
    return count / len(tokens1) if tokens1 else 0

def grammar_error_count(text):
    tool = language_tool_python.LanguageTool('en-US')
    return len(tool.check(text))

# Evaluation
def evaluate_answer(student_file, model_file):
    student_text = extract_text(student_file)
    key_text = extract_text(model_file)

    student_tokens = preprocess(student_text)
    key_tokens = preprocess(key_text)

    cos_sim = compute_cosine_similarity(student_text, key_text)
    jac_sim = jaccard_similarity(set(student_tokens), set(key_tokens))
    bigr_sim = bigram_similarity(student_tokens, key_tokens)
    syn_sim = synonym_similarity(student_tokens, key_tokens)
    grammar_errors = grammar_error_count(student_text)

    final_score = (
        0.3 * cos_sim +
        0.2 * jac_sim +
        0.2 * bigr_sim +
        0.2 * syn_sim
    )

    return {
        "Cosine Similarity": round(cos_sim, 2),
        "Jaccard Similarity": round(jac_sim, 2),
        "Bigram Similarity": round(bigr_sim, 2),
        "Synonym Similarity": round(syn_sim, 2),
        "Grammar Errors": grammar_errors,
        "Final Score": round(final_score * 100, 2)
    }

# Streamlit UI
st.set_page_config(page_title="Smart Answer Evaluator", layout="centered")
st.title("🧠 Smart Answer Evaluator (PDF or Image Upload)")

student_file = st.file_uploader("Upload Student Answer (PDF or Image)", type=["pdf", "jpg", "jpeg", "png"])
model_file = st.file_uploader("Upload Model Answer (PDF or Image)", type=["pdf", "jpg", "jpeg", "png"])

if student_file and model_file:
    if st.button("Evaluate"):
        with st.spinner("Evaluating..."):
            results = evaluate_answer(student_file, model_file)

        st.success("✅ Evaluation Completed")
        st.metric("Final Score (%)", results["Final Score"])
        st.markdown("### 🔍 Detailed Metrics")
        st.write(f"🧮 Cosine Similarity: {results['Cosine Similarity']}")
        st.write(f"📊 Jaccard Similarity: {results['Jaccard Similarity']}")
        st.write(f"🔗 Bigram Similarity: {results['Bigram Similarity']}")
        st.write(f"🔁 Synonym Similarity: {results['Synonym Similarity']}")
        st.write(f"✍️ Grammar Errors: {results['Grammar Errors']}")

        if results["Final Score"] >= 85:
            st.markdown("🌟 **Performance: Excellent**")
        elif results["Final Score"] >= 70:
            st.markdown("👍 **Performance: Good**")
        elif results["Final Score"] >= 50:
            st.markdown("🧐 **Performance: Fair**")
        else:
            st.markdown("⚠️ **Performance: Needs Improvement**")
