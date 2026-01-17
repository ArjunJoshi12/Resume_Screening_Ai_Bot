import streamlit as st
from utils import read_pdf, read_docx
from embedding import get_embedding
from ranking import rank_resumes
from openai import OpenAI
import os

st.set_page_config(page_title="AI Resume Ranker", layout="wide")
st.title("🤖 AI Resume Screening System")

jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
resume_files = st.file_uploader("Upload Resumes (PDF / DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

if st.button("Analyze Resumes") and jd_file and resume_files:
    st.info("Reading Job Description...")
    jd_text = read_pdf(jd_file)
    print(jd_text)
    jd_vector = get_embedding(jd_text)
    print(jd_vector)    

    resume_vectors = []
    resume_names = []
    resume_texts = {}

    st.info("Processing Resumes...")

    for file in resume_files:
        if file.name.endswith(".pdf"):
            text = read_pdf(file)
        else:
            text = read_docx(file)

        resume_texts[file.name] = text
        resume_vectors.append(get_embedding(text))
        resume_names.append(file.name)
        
        print(resume_texts)
        print(resume_names)
        print(resume_vectors)

    ranked = rank_resumes(jd_vector, resume_vectors, resume_names, top_k=len(resume_names))

    st.success("Ranking Completed!")

    if "OPENAI_API_KEY" in os.environ:
        client = OpenAI()

    for rank, name, score in ranked:
        st.subheader(f"🏆 Rank {rank}: {name}")

        if "OPENAI_API_KEY" in os.environ:
            prompt = f"""
            Job Description:
            {jd_text}

            Resume:
            {resume_texts[name]}

            Give:
            1. Match Percentage
            2. Strengths
            3. Missing Skills
            4. Final Verdict
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )

            st.write(response.choices[0].message.content)
        else:
            st.write("Similarity Score:", score)

        st.divider()
