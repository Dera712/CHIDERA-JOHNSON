# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 12:26:41 2026

@author: CHIDERA
"""

import os
import re
import pandas as pd

from docx import Document
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# =====================================================================
# STAGE 1: DATA ACQUISITION
# =====================================================================

# ---------------------------------------------------------------------
# File Paths
# ---------------------------------------------------------------------

JOB_DESCRIPTION =r"C:\Users\CHIDERA\OneDrive\Documents\FSD JD\Job Description.docx"

CV_FOLDER = r"C:\Users\CHIDERA\OneDrive\Documents\FSD_Resumes"
# ---------------------------------------------------------------------
# Read Job Description
# ---------------------------------------------------------------------

def read_docx(file_path):

    document = Document(file_path)

    return "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
    )

job_description = read_docx(JOB_DESCRIPTION)

print("="*80)
print("FULL STACK JOB DESCRIPTION")
print("="*80)

print(job_description[:1000])

# ---------------------------------------------------------------------
# Read CVs
# ---------------------------------------------------------------------

candidate_names = []

candidate_texts = []

for filename in os.listdir(CV_FOLDER):

    if filename.endswith(".docx"):

        filepath = os.path.join(
            CV_FOLDER,
            filename
        )

        candidate_names.append(filename)

        candidate_texts.append(
            read_docx(filepath)
        )

print("\n")

print("="*80)
print("DATASET SUMMARY")
print("="*80)

print(f"Number of Resumes : {len(candidate_names)}")

print("\nFirst Ten Resumes")

print(candidate_names[:10])


# =====================================================================
# STAGE 2: TEXT PREPROCESSING
# =====================================================================

# ---------------------------------------------------------------------
# Text Preprocessing Function
# ---------------------------------------------------------------------

def preprocess(text):
    """
    Performs basic text preprocessing.
    """

    # Convert to lowercase
    text = text.lower()

    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # Remove numbers
    text = re.sub(r'\d+', ' ', text)

    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ---------------------------------------------------------------------
# Preprocess Job Description
# ---------------------------------------------------------------------

processed_job_description = preprocess(job_description)

# ---------------------------------------------------------------------
# Preprocess Candidate Resumes
# ---------------------------------------------------------------------

processed_cvs = []

for cv in candidate_texts:
    processed_cvs.append(preprocess(cv))

# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

print("\n")
print("=" * 90)
print("PREPROCESSING COMPLETE")
print("=" * 90)

print(f"Job Description Length (Before): {len(job_description)}")
print(f"Job Description Length (After):  {len(processed_job_description)}")

print("\nFirst 500 Characters of Processed Job Description")
print("-" * 90)

print(processed_job_description[:500])

print("\n")

print("=" * 90)
print("FIRST PREPROCESSED RESUME")
print("=" * 90)

print(processed_cvs[0][:700])

print("\n")

print("=" * 90)
print("TOTAL PREPROCESSED RESUMES")
print("=" * 90)

print(len(processed_cvs))


# =====================================================================
# WORD CLOUD 1: PROCESSED JOB DESCRIPTION
# =====================================================================

# ---------------------------------------------------------------------
# Custom Stopwords
# ---------------------------------------------------------------------

custom_stopwords = STOPWORDS.union({

    "developer",
    "participated",
    "worked",
    "full"
    "online",
    "development",
    "experience",
    "required",
    "ability",
    "skills",
    "skill",
    "knowledge",
    "work",
    "working",
    "using",
    "candidate",
    "role",
    "team",
    "years",
    "company",
    "job"
})

# ---------------------------------------------------------------------
# Generate Word Cloud
# ---------------------------------------------------------------------

job_wordcloud = WordCloud(

    width=1400,
    height=700,
    background_color="white",
    stopwords=custom_stopwords,
    collocations=False

).generate(processed_job_description)

# ---------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------

plt.figure(figsize=(16,8))

plt.imshow(job_wordcloud, interpolation="bilinear")

plt.axis("off")

plt.title(
    "Word Cloud of the Processed Full Stack Developer Job Description",
    fontsize=18
)

plt.show()

# ---------------------------------------------------------------------
# Save Results
# ---------------------------------------------------------------------

job_wordcloud.to_file(r"C:\Users\CHIDERA\OneDrive\Documents\DSSR2\Job Decription WordCloud(Baseline).png")


# =====================================================================
# WORD CLOUD 2: PROCESSED RESUME CORPUS
# =====================================================================

resume_corpus = " ".join(processed_cvs)

resume_wordcloud = WordCloud(

    width=1400,
    height=700,
    background_color="white",
    stopwords=custom_stopwords,
    collocations=False

).generate(resume_corpus)

plt.figure(figsize=(16,8))

plt.imshow(resume_wordcloud, interpolation="bilinear")

plt.axis("off")

plt.title(
    "Word Cloud of the Processed Full Stack Developer Resume Corpus",
    fontsize=18
)

plt.show()

# ---------------------------------------------------------------------
# Save Results
# ---------------------------------------------------------------------
resume_wordcloud.to_file(r"C:\Users\CHIDERA\OneDrive\Documents\DSSR2\FullStack_Resume Corpus_WordCloud(Baseline).png")


# =====================================================================
# STAGE 3: TF-IDF FEATURE EXTRACTION
# =====================================================================

# ---------------------------------------------------------------------
# Combine Job Description and CVs
# ---------------------------------------------------------------------

documents = [processed_job_description] + processed_cvs

# ---------------------------------------------------------------------
# TF-IDF Vectorisation
# ---------------------------------------------------------------------

vectorizer = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = vectorizer.fit_transform(documents)

# ---------------------------------------------------------------------
# Feature Names
# ---------------------------------------------------------------------

feature_names = vectorizer.get_feature_names_out()

# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

print("\n")
print("=" * 90)
print("TF-IDF FEATURE EXTRACTION")
print("=" * 90)

print(f"Number of Documents : {len(documents)}")

print(f"Vocabulary Size     : {len(feature_names)}")

print(f"TF-IDF Matrix Shape : {tfidf_matrix.shape}")

print("\n")

print("=" * 90)
print("FIRST 25 FEATURES")
print("=" * 90)

print(feature_names[:25])

print("\n")

print("=" * 90)
print("SAMPLE TF-IDF MATRIX")
print("=" * 90)

print(tfidf_matrix[:5].toarray())


# =====================================================================
# STAGE 4: COSINE SIMILARITY
# =====================================================================
# ---------------------------------------------------------------------
# Calculate Cosine Similarity
# ---------------------------------------------------------------------

similarity_scores = cosine_similarity(
    tfidf_matrix[0:1],      # Job Description
    tfidf_matrix[1:]        # Candidate Resumes
).flatten()

# ---------------------------------------------------------------------
# Create Similarity DataFrame
# ---------------------------------------------------------------------

similarity_df = pd.DataFrame({

    "Candidate": candidate_names,

    "Cosine Similarity": similarity_scores.round(4),

    "Similarity Score": (similarity_scores * 100).round(2)

})

# ---------------------------------------------------------------------
# Rank Candidates
# ---------------------------------------------------------------------

similarity_df = similarity_df.sort_values(
    by="Similarity Score",
    ascending=False
).reset_index(drop=True)

similarity_df["Rank"] = similarity_df.index + 1

# ---------------------------------------------------------------------
# Rearrange Columns
# ---------------------------------------------------------------------

similarity_df = similarity_df[[
    "Rank",
    "Candidate",
    "Cosine Similarity",
    "Similarity Score"
]]

# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

print("\n")
print("=" * 100)
print("COSINE SIMILARITY ANALYSIS")
print("=" * 100)

print("\nTop 10 Candidates")
print("-" * 100)

print(similarity_df.head(10))

print("\n")

print("=" * 100)
print("COSINE SIMILARITY STATISTICS")
print("=" * 100)

print(similarity_df["Similarity Score"].describe())

print("\n")

print("=" * 100)
print("HIGHEST SIMILARITY SCORE")
print("=" * 100)

print(
    similarity_df.nlargest(
        1,
        "Similarity Score"
    )
)

print("\n")

print("=" * 100)
print("LOWEST SIMILARITY SCORE")
print("=" * 100)

print(
    similarity_df.nsmallest(
        1,
        "Similarity Score"
    )
)

# ---------------------------------------------------------------------
# Save Results
# ---------------------------------------------------------------------

similarity_df.to_excel(
    r"C:\Users\CHIDERA\OneDrive\Documents\DSSR2\Cosine_Similarity(Baseline).xlsx",
    index=False
)


# =====================================================================
# STAGE 5: CANDIDATE RANKING
# =====================================================================

# ---------------------------------------------------------------------
# Rank Candidates
# ---------------------------------------------------------------------

baseline_dashboard = similarity_df.copy()

baseline_dashboard = baseline_dashboard.sort_values(
    by="Similarity Score",
    ascending=False
).reset_index(drop=True)

baseline_dashboard["Rank"] = baseline_dashboard.index + 1

# ---------------------------------------------------------------------
# Display Dashboard
# ---------------------------------------------------------------------

print("\n")
print("=" * 100)
print("BASELINE RECRUITMENT DASHBOARD")
print("=" * 100)

print(baseline_dashboard)

# ---------------------------------------------------------------------
# Top 10 Candidates
# ---------------------------------------------------------------------

print("\n")
print("=" * 100)
print("TOP 10 CANDIDATES")
print("=" * 100)

print(baseline_dashboard.head(10))

# ---------------------------------------------------------------------
# Bottom 10 Candidates
# ---------------------------------------------------------------------

print("\n")
print("=" * 100)
print("BOTTOM 10 CANDIDATES")
print("=" * 100)

print(baseline_dashboard.tail(10))

# ---------------------------------------------------------------------
# Summary Statistics
# ---------------------------------------------------------------------

print("\n")
print("=" * 100)
print("SIMILARITY SCORE STATISTICS")
print("=" * 100)

print(baseline_dashboard["Similarity Score"].describe())

# ---------------------------------------------------------------------
# Highest Similarity
# ---------------------------------------------------------------------

print("\n")
print("=" * 100)
print("HIGHEST SIMILARITY SCORE")
print("=" * 100)

print(
    baseline_dashboard.nlargest(
        1,
        "Similarity Score"
    )
)

# ---------------------------------------------------------------------
# Lowest Similarity
# ---------------------------------------------------------------------

print("\n")
print("=" * 100)
print("LOWEST SIMILARITY SCORE")
print("=" * 100)

print(
    baseline_dashboard.nsmallest(
        1,
        "Similarity Score"
    )
)

# ---------------------------------------------------------------------
# Export Dashboard
# ---------------------------------------------------------------------

baseline_dashboard.to_excel(
    r"C:\Users\CHIDERA\OneDrive\Documents\DSSR2\Baseline_FullStack_Dashboard.xlsx",
    index=False
)

# =====================================================================
# HISTOGRAM OF COSINE SIMILARITY SCORES
# =====================================================================

plt.figure(figsize=(10,6))

plt.hist(
    baseline_dashboard["Similarity Score"],
    bins=10
)

plt.title(
    "Distribution of Cosine Similarity Scores",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Similarity Score (%)")

plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "Baseline_Similarity_Distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()






























