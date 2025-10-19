"""
Medical FAQ Chatbot - Streamlit Application
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Medical FAQ Chatbot",
    page_icon="🏥",
    layout="wide"
)

# Dark theme CSS
st.markdown("""
<style>
    .stApp {
        background-color: #1e1e1e;
        color: #ffffff;
        max-width: 100%;
    }
    .stTextArea textarea {
        background-color: #2d2d2d;
        color: #ffffff;
        border: 1px solid #444;
    }
    .stSelectbox > div > div {
        background-color: #2d2d2d;
        color: #ffffff;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff;
        font-size: 24px;
    }
    [data-testid="stMetricLabel"] {
        color: #b0b0b0;
    }
    .info-card {
        background-color: #2d2d2d;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        margin: 10px 0;
    }
    h1 {
        text-align: center;
        color: #ffffff;
    }
    h2, h3 {
        color: #ffffff;
    }
    .subtitle {
        text-align: center;
        color: #b0b0b0;
        font-size: 16px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Text preprocessing
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s\?\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        medical_keep_words = {'what', 'when', 'where', 'who', 'how', 'why', 'which'}
        self.stop_words = self.stop_words - medical_keep_words
    
    def preprocess(self, text):
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token.isalnum() and token not in self.stop_words]
        return ' '.join(tokens)

# Chatbot class
class MedicalChatbot:
    def __init__(self, df, tfidf_vectorizer, tfidf_matrix, 
                 sentence_model, faiss_index, preprocessor):
        self.df = df.reset_index(drop=True)
        self.tfidf_vectorizer = tfidf_vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.sentence_model = sentence_model
        self.faiss_index = faiss_index
        self.preprocessor = preprocessor
    
    def get_answer_tfidf(self, question, top_k=3):
        processed_q = self.preprocessor.preprocess(clean_text(question))
        question_vec = self.tfidf_vectorizer.transform([processed_q])
        similarities = cosine_similarity(question_vec, self.tfidf_matrix)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'question': self.df.iloc[idx]['question'],
                'answer': self.df.iloc[idx]['answer'],
                'focus_area': self.df.iloc[idx]['focus_area'],
                'similarity': float(similarities[idx])
            })
        return results
    
    def get_answer_semantic(self, question, top_k=3):
        question_embedding = self.sentence_model.encode([clean_text(question)])
        faiss.normalize_L2(question_embedding)
        similarities, indices = self.faiss_index.search(question_embedding, top_k)
        
        results = []
        for idx, sim in zip(indices[0], similarities[0]):
            results.append({
                'question': self.df.iloc[idx]['question'],
                'answer': self.df.iloc[idx]['answer'],
                'focus_area': self.df.iloc[idx]['focus_area'],
                'similarity': float(sim)
            })
        return results
    
    def get_answer_hybrid(self, question, top_k=3, weights=(0.4, 0.6)):
        tfidf_results = self.get_answer_tfidf(question, top_k=10)
        semantic_results = self.get_answer_semantic(question, top_k=10)
        
        combined_scores = {}
        
        for result in tfidf_results:
            q = result['question']
            combined_scores[q] = {
                'tfidf': result['similarity'],
                'semantic': 0,
                'answer': result['answer'],
                'focus_area': result['focus_area']
            }
        
        for result in semantic_results:
            q = result['question']
            if q in combined_scores:
                combined_scores[q]['semantic'] = result['similarity']
            else:
                combined_scores[q] = {
                    'tfidf': 0,
                    'semantic': result['similarity'],
                    'answer': result['answer'],
                    'focus_area': result['focus_area']
                }
        
        for q in combined_scores:
            tfidf_score = combined_scores[q]['tfidf']
            semantic_score = combined_scores[q]['semantic']
            combined_scores[q]['final_score'] = (
                weights[0] * tfidf_score + weights[1] * semantic_score
            )
        
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1]['final_score'],
            reverse=True
        )[:top_k]
        
        results = []
        for q, data in sorted_results:
            results.append({
                'question': q,
                'answer': data['answer'],
                'focus_area': data['focus_area'],
                'similarity': data['final_score']
            })
        return results
    
    def answer(self, question, method='hybrid', top_k=3):
        if method == 'tfidf':
            return self.get_answer_tfidf(question, top_k)
        elif method == 'semantic':
            return self.get_answer_semantic(question, top_k)
        else:
            return self.get_answer_hybrid(question, top_k)

# Load models
@st.cache_resource
def load_models():
    try:
        df = pd.read_csv('processed_medquad.csv')
        
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        
        with open('tfidf_matrix.pkl', 'rb') as f:
            tfidf_matrix = pickle.load(f)
        
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        
        faiss_index = faiss.read_index('faiss_index.bin')
        
        with open('config.pkl', 'rb') as f:
            config = pickle.load(f)
        
        sentence_model = SentenceTransformer(config['model_name'])
        
        return df, tfidf_vectorizer, tfidf_matrix, preprocessor, faiss_index, sentence_model
    
    except FileNotFoundError as e:
        st.error(f"❌ Error: Model files not found!")
        st.info("Please run the Jupyter notebook first to generate model files.")
        st.stop()

@st.cache_resource
def initialize_chatbot():
    df, tfidf_vectorizer, tfidf_matrix, preprocessor, faiss_index, sentence_model = load_models()
    
    chatbot = MedicalChatbot(
        df=df,
        tfidf_vectorizer=tfidf_vectorizer,
        tfidf_matrix=tfidf_matrix,
        sentence_model=sentence_model,
        faiss_index=faiss_index,
        preprocessor=preprocessor
    )
    
    return chatbot, df

# Main app
def main():
    # Load models
    with st.spinner("⏳ Loading chatbot..."):
        chatbot, df = initialize_chatbot()
    
    # Header with info
    st.title("🏥 Medical FAQ Chatbot")
    st.markdown('<p class="subtitle">Get instant answers to medical questions from trusted National Institutes of Health (NIH) sources</p>', unsafe_allow_html=True)
    
    # Dataset info cards
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    
    with info_col1:
        st.metric("📚 Total Questions", f"{len(df):,}")
    
    with info_col2:
        st.metric("🏷️ Focus Areas", f"{df['focus_area'].nunique()}")
    
    with info_col3:
        st.metric("🔍 Search Methods", "3")
    
    with info_col4:
        st.metric("✅ Status", "Ready")
    
    st.markdown("---")
    
    # Main content layout
    col_left, col_center, col_right = st.columns([1, 2, 1])
    
    with col_left:
        st.subheader("⚙️ Search Options")
        
        search_method = st.selectbox(
            "Method",
            ["Hybrid", "Semantic", "TF-IDF"],
            help="Hybrid: Best accuracy\nSemantic: Context understanding\nTF-IDF: Keyword matching"
        )
        
        num_results = st.slider(
            "Results",
            min_value=1,
            max_value=5,
            value=3
        )
        
        st.markdown("---")
        st.subheader("💡 Quick Start")
        
        sample_questions = [
            "What is diabetes?",
            "What is glaucoma?",
            "Hypertension prevention",
            "Heart disease causes",
            "Cancer symptoms"
        ]
        
        for i, sq in enumerate(sample_questions):
            if st.button(sq, key=f"sample_{i}", use_container_width=True):
                st.session_state.sample_question = sq
                st.rerun()
    
    with col_center:
        st.subheader("💬 Your Medical Question")
        
        # Check for sample question
        default_question = st.session_state.get('sample_question', '')
        
        question = st.text_area(
            label="Type your question below",
            height=150,
            value=default_question,
            placeholder="Example: What are the symptoms of diabetes?",
            key="question_input",
            label_visibility="collapsed"
        )
        
        # Clear sample question after use
        if 'sample_question' in st.session_state:
            del st.session_state.sample_question
        
        search_clicked = st.button("🔍 Search Answer", type="primary", use_container_width=True)
        
        st.markdown("---")
        
        # Search logic
        if search_clicked:
            if question and question.strip():
                st.subheader("📋 Search Results")
                
                with st.spinner("🔍 Searching for answers..."):
                    method = search_method.lower()
                    results = chatbot.answer(question, method=method, top_k=num_results)
                
                # Display results
                for i, result in enumerate(results, 1):
                    # Confidence
                    similarity = result['similarity']
                    if similarity > 0.7:
                        conf_emoji = "🟢"
                        conf_text = "High Confidence"
                    elif similarity > 0.4:
                        conf_emoji = "🟡"
                        conf_text = "Medium Confidence"
                    else:
                        conf_emoji = "🔴"
                        conf_text = "Low Confidence"
                    
                    # Display result
                    st.markdown(f"### {conf_emoji} Result {i}")
                    
                    res_col1, res_col2 = st.columns([1, 1])
                    with res_col1:
                        st.markdown(f"**Category:** `{result['focus_area']}`")
                    with res_col2:
                        st.markdown(f"**{conf_text}** `{similarity:.3f}`")
                    
                    st.markdown("**💡 Answer:**")
                    st.info(result['answer'])
                    
                    with st.expander("📌 View Related Question"):
                        st.write(result['question'])
                    
                    if i < len(results):
                        st.markdown("---")
                
                st.warning("⚠️ **Medical Disclaimer:** This information is for educational purposes only. Always consult healthcare professionals for medical advice.")
            
            else:
                st.error("❌ Please enter a question to search!")
        else:
            # Show placeholder when no search
            st.info("👆 Enter your medical question above and click 'Search Answer' to get started!")
    
    with col_right:
        st.subheader("📊 Dataset Details")
        
        st.markdown("""
        <div class="info-card">
            <b>Source:</b><br>
            National Institutes of Health (NIH)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <b>Coverage:</b><br>
            12 NIH Websites
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <b>Topics Include:</b><br>
            • Diabetes<br>
            • Heart Disease<br>
            • Cancer<br>
            • Glaucoma<br>
            • Hypertension<br>
            • And more...
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("🎯 How It Works")
        
        st.markdown("""
        <div class="info-card">
            <b>1. Ask Question</b><br>
            Type your medical question
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <b>2. AI Search</b><br>
            Advanced algorithms find best matches
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <b>3. Get Answers</b><br>
            Receive trusted NIH information
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888; padding: 10px;'>"
        "<p><b>Medical FAQ Chatbot</b> NLP Assignment | Pavithra L022</p>"
        "<p style='font-size: 12px;'>Powered by MedQuAD Dataset from NIH | Pavithra L022</p>"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    if 'sample_question' not in st.session_state:
        st.session_state.sample_question = ''
    
    main()