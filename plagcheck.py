import streamlit as st
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import pandas as pd
import sklearn

# -------------------------------
# Function to extract text from PDF
# -------------------------------
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# -------------------------------
# Text cleaning
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return text

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="PDF Plagiarism Checker", page_icon="📑")

st.title("📑 Multi-PDF Plagiarism Checker")
st.write("Upload multiple PDFs to check pairwise similarity. The model refreshes **every run**.")

# File uploader (accept multiple files)
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files and len(uploaded_files) > 1:
    with st.spinner("Extracting and processing PDFs..."):
        # Extract and clean text for all PDFs
        docs = []
        file_names = []
        for f in uploaded_files:
            text = extract_text_from_pdf(f)
            cleaned = clean_text(text)
            docs.append(cleaned)
            file_names.append(f.name)

        # TF-IDF Vectorization (fresh every run)
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(docs)

        # Cosine similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)


        st.subheader("📊 Pairwise Similarity Scores")

        

        # Collect pairwise results
        results = []
        for (i, j) in itertools.combinations(range(len(file_names)), 2):
            score = similarity_matrix[i, j] * 100
            results.append({
                "PDF 1": file_names[i],
                "PDF 2": file_names[j],
                "Similarity (%)": round(score, 2)
            })

        # Convert to DataFrame for neat table
        df_results = pd.DataFrame(results)

        # Sort by similarity (highest first)
        df_results = df_results.sort_values(by="Similarity (%)", ascending=False).reset_index(drop=True)

        # Display as  table
        st.dataframe(df_results, use_container_width=True)

        
        import seaborn as sns
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(similarity_matrix, annot=True, fmt=".2f",
                    xticklabels=file_names, yticklabels=file_names,
                    cmap="YlGnBu", cbar=True, ax=ax)

        st.subheader("🔥 Similarity Heatmap")
        st.pyplot(fig)


        
        st.subheader("📖 Preview of Extracted Texts")
        for i, f in enumerate(uploaded_files):
            st.text_area(f"Document: {file_names[i]}", docs[i][:1000] + "...", height=100)

st.write("🔍 Library Versions:")
st.write(f"• Streamlit: {st.__version__}")
st.write(f"• PyPDF2: {PyPDF2.__version__}")
st.write(f"• scikit-learn: {sklearn.__version__}")
st.write(f"• pandas: {pd.__version__}")
st.write(f"• re: built-in (no version)")
