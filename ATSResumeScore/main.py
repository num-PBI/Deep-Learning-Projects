import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
import fitz

st.title("Get your Resume Score")
pdf_file =st.file_uploader("Upload your resume here",type=["pdf"])
if pdf_file is not None:
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    resume_text = ""

    # Extract text from each page
    for page in doc:
        resume_text += page.get_text()
    job_description=st.text_input("Enter the job decription:")
    nltk.download("stopwords")
    stop_words=set(stopwords.words("english"))
    def preprocess_text(text):
        text=text.lower()
        text=re.sub(r"[^\w\s]","",text)
        text=" ".join(word for word in text.split() if word not in stop_words)
        return text
    clean_resume=preprocess_text(resume_text)
    clean_job_description=preprocess_text(job_description)
    button0=st.button("Generate Score")# Load pre-trained Sentence-BERT model
    if button0:
        with st.spinner("Analyzing resume and job description..."):
            model = SentenceTransformer('all-MiniLM-L6-v2')
    
            # Encode resume and job description
            resume_vector = model.encode(clean_resume).reshape(1, -1)
            job_desc_vector = model.encode(clean_job_description).reshape(1, -1)
    
            # Compute similarity
            matching_score = cosine_similarity(resume_vector, job_desc_vector)[0][0] * 100
            st.header(f"Your score is : {int(matching_score)}/100")
            st.write("Get Suggestions of Keywords to be added in your resume for better matching score. Score above 90 gets selected.")
            with st.spinner("Extracting important keywords..."):
                kw_model = KeyBERT()
        
                # Function to extract top N keywords using KeyBERT
                def extract_keywords_bert(text, top_n=10):
                    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
                    return {word[0] for word in keywords}  # Extract just the words
        
                # Extract keywords from job description and resume
                job_keywords = extract_keywords_bert(clean_job_description, top_n=10)
                resume_keywords = extract_keywords_bert(clean_resume, top_n=10)
        
                # Find missing keywords (important terms present in JD but not in Resume)
                missing_keywords = job_keywords - resume_keywords
        
            # Provide suggestions
                if missing_keywords:
                    st.write(f"Add these important keywords to your resume : \n {', '.join(missing_keywords)}")
