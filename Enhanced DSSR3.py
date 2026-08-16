# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 21:05:28 2026

@author: CHIDERA
"""

import os
import docx
import spacy
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# =============================================================================
# STAGE 1: DATA ACQUISITION
# =============================================================================

"""
This stage loads the Software Developer job description and all candidate
CVs from Microsoft Word (.docx) documents.

The job description serves as the reference document, while the CVs form
the candidate dataset used throughout the recruitment pipeline. """

# =============================================================================
# FILE PATHS
# =============================================================================

# Job Description

JD_PATH = r"C:\Users\CHIDERA\OneDrive\Documents\jd\Job Description.docx"

# Folder containing candidate CVs

CV_FOLDER = r"C:\Users\CHIDERA\OneDrive\Documents\Resumes"

# =============================================================================
# READ WORD DOCUMENT
# =============================================================================

def read_docx(file_path):
    """
    Reads a Microsoft Word (.docx) document and returns its text.
        Complete text contained in the document.
    """

    document = docx.Document(file_path)

    text = "\n".join(

        paragraph.text

        for paragraph in document.paragraphs

    )

    return text

# =============================================================================
# LOAD ALL CVS
# =============================================================================

def load_cvs(folder_path):
    """
    Loads all CVs stored as Word (.docx) documents.
    """

    cvs = {}

    for filename in sorted(os.listdir(folder_path)):

        if filename.endswith(".docx") and not filename.startswith("~$"):

            full_path = os.path.join(folder_path, filename)

            cvs[filename] = read_docx(full_path)

    return cvs

# =============================================================================
# LOAD DATA
# =============================================================================

print("\nLoading recruitment documents...\n")

job_description = read_docx(JD_PATH)

cvs_database = load_cvs(CV_FOLDER)

# =============================================================================
# VALIDATION
# =============================================================================

print("=" * 80)
print("DATA ACQUISITION SUMMARY")
print("=" * 80)

print(f"Job Description Loaded : {'Yes' if len(job_description) > 0 else 'No'}")

print(f"Total CVs Loaded       : {len(cvs_database)}")

print("\nFirst Five Candidates:")

for candidate in list(cvs_database.keys())[:5]:

    print(candidate)

print("="*80)
print("SOFTWARE DEVELOPER JOB DESCRIPTION")
print("="*80)

print(job_description[:1000])



# =============================================================================
# STAGE 2: NLP INITIALISATION
# =============================================================================

"""
This stage prepares the Natural Language Processing (NLP) environment
used throughout the enhanced recruitment system.

Functions:
1. Loads the spaCy English language model.
2. Creates an EntityRuler for recognising technical recruitment skills.
3. Registers Software Developer skills as custom entities.
4. Verifies that all skills have been loaded successfully.
"""

# =============================================================================
# LOAD SPACY MODEL
# =============================================================================

nlp = spacy.load("en_core_web_sm")

# =============================================================================
# ADD ENTITY RULER
# =============================================================================

# Prevent duplicate EntityRulers when rerunning the script

if "entity_ruler" in nlp.pipe_names:
    nlp.remove_pipe("entity_ruler")

ruler = nlp.add_pipe("entity_ruler", before="ner")

# =============================================================================
# SOFTWARE DEVELOPER TECHNICAL SKILLS
# =============================================================================

TECHNICAL_SKILLS = [

    # --------------------------------------------------
    # Programming Languages
    # --------------------------------------------------

    "java",
    "javascript",
    "python",
    "go",
    "c",
    "c++",

    # --------------------------------------------------
    # Web Development
    # --------------------------------------------------

    "html5",
    "css3",
    "angular",
    "polymer",
    "closure library",
    "backbone",

    # --------------------------------------------------
    # Software Engineering
    # --------------------------------------------------

    "agile",
    "testing",
    "deployment",
    "software development",
    "software development lifecycle",
    "full-stack",
    "full stack",
    "web application",
    "web applications",
    "architecture",
    "design",
    "analysis",
    "code review",
    "code reviews"

]

# =============================================================================
# CREATE ENTITY PATTERNS
# =============================================================================

patterns = []

for skill in TECHNICAL_SKILLS:

    pattern = {

        "label": "SKILL",

        "pattern": [

            {"LOWER": token}

            for token in skill.split()
        ]

    }
    patterns.append(pattern)

# =============================================================================
# REGISTER PATTERNS
# =============================================================================

ruler.add_patterns(patterns)

# =============================================================================
# VALIDATION
# =============================================================================

print("=" * 80)
print("NLP INITIALISATION SUMMARY")
print("=" * 80)

print(f"spaCy Model Loaded        : en_core_web_sm")

print(f"EntityRuler Added         : {'entity_ruler' in nlp.pipe_names}")

print(f"Technical Skills Loaded   : {len(TECHNICAL_SKILLS)}")

print("\nSample Skills:")

for skill in TECHNICAL_SKILLS[:10]:

    print("-", skill)


# =============================================================================
# STAGE 3: BIAS MITIGATION
# =============================================================================

"""
This stage anonymises sensitive information from candidate CVs before
feature extraction.

The following information is masked:

• Email addresses
• Years
• Geographical locations (GPE)
"""

# =============================================================================
# EXPRESSIONS
# =============================================================================

EMAIL_PATTERN = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

YEAR_PATTERN = r"\b(19\d{2}|20\d{2})\b"

# =============================================================================
# BIAS MITIGATION FUNCTION
# =============================================================================

def apply_bias_mitigation(text):

    # -----------------------------
    # Mask Email Addresses
    # -----------------------------

    text = re.sub(
        EMAIL_PATTERN,
        "[EMAIL]",
        text
    )

    # -----------------------------
    # Mask Years
    # -----------------------------

    text = re.sub(
        YEAR_PATTERN,
        "[YEAR]",
        text
    )

    # -----------------------------
    # Detect Locations
    # -----------------------------

    doc = nlp(text)

    output = []

    for token in doc:

        if token.ent_type_ == "GPE":

            output.append("[LOCATION]")

        else:

            output.append(token.text)

    return " ".join(output)

# =============================================================================
# APPLY BIAS MITIGATION
# =============================================================================

processed_cvs = []

for filename, cv_text in cvs_database.items():

    cleaned_text = apply_bias_mitigation(cv_text)

    processed_cvs.append(cleaned_text)

# =============================================================================
# VALIDATION
# =============================================================================

print("\n")
print("=" * 80)
print("BIAS MITIGATION SUMMARY")
print("=" * 80)

print(f"Original CVs : {len(cvs_database)}")

print(f"Processed CVs: {len(processed_cvs)}")

print("\nSample Processed CV\n")

print(processed_cvs[0][:600])



# =============================================================================
# STAGE 4: FEATURE ENGINEERING
# =============================================================================

"""
This stage extracts job-specific technical features from every anonymised CV.

The extracted features are grouped into:

1. Programming Skills
2. Web Development Skills
3. Software Engineering Skills

For each category the system calculates:

• Number of matched skills
• Percentage match
• Skills identified
"""


# =============================================================================
# SOFTWARE DEVELOPER SKILL CATEGORIES
# =============================================================================

PROGRAMMING_SKILLS = {

    "java",
    "javascript",
    "python",
    "go",
    "c",
    "c++"

}

WEB_DEVELOPMENT_SKILLS = {

    "html5",
    "css3",
    "angular",
    "polymer",
    "closure library",
    "backbone"

}

SOFTWARE_ENGINEERING_SKILLS = {

    "agile",
    "testing",
    "deployment",
    "software development",
    "software development lifecycle",
    "full-stack",
    "full stack",
    "web application",
    "web applications",
    "architecture",
    "design",
    "analysis",
    "code review",
    "code reviews"

}

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

feature_list = []

for filename, cv_text in zip(cvs_database.keys(), processed_cvs):

    doc = nlp(cv_text)

    extracted_skills = {

        ent.text.lower()

        for ent in doc.ents

        if ent.label_ == "SKILL"

    }

    # ----------------------------------------------------
    # Programming
    # ----------------------------------------------------

    programming_matches = extracted_skills & PROGRAMMING_SKILLS

    programming_percentage = round(

        (len(programming_matches) / len(PROGRAMMING_SKILLS)) * 100,

        2

    )

    # ----------------------------------------------------
    # Web Development
    # ----------------------------------------------------

    web_matches = extracted_skills & WEB_DEVELOPMENT_SKILLS

    web_percentage = round(

        (len(web_matches) / len(WEB_DEVELOPMENT_SKILLS)) * 100,

        2

    )

    # ----------------------------------------------------
    # Software Engineering
    # ----------------------------------------------------

    software_matches = extracted_skills & SOFTWARE_ENGINEERING_SKILLS

    software_percentage = round(

        (len(software_matches) / len(SOFTWARE_ENGINEERING_SKILLS)) * 100,

        2

    )

    # ----------------------------------------------------
    # Store Candidate Features
    # ----------------------------------------------------

    feature_list.append({

        "Candidate": filename,

        "Programming Match (%)": programming_percentage,

        "Web Development Match (%)": web_percentage,

        "Software Engineering Match (%)": software_percentage,

        "Programming Skills Found":
            ", ".join(sorted(programming_matches)),

        "Web Skills Found":
            ", ".join(sorted(web_matches)),

        "Software Engineering Skills Found":
            ", ".join(sorted(software_matches))

    })

# =============================================================================
# CREATE DATAFRAME
# =============================================================================
print("CVs:", len(cvs_database))
print("Processed:", len(processed_cvs))
print("feature_list:", len(feature_list))
print(feature_list[-3:])

features_df = pd.DataFrame(feature_list)

# =============================================================================
# VALIDATION
# =============================================================================

print("\n")
print("=" * 90)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 90)

print(f"CVs Loaded              : {len(cvs_database)}")

print(f"Processed CVs           : {len(processed_cvs)}")

print(f"Feature Records Created : {len(features_df)}")

if len(features_df) == len(cvs_database):

    print("\n✓ Validation Successful")

else:

    print("\n✗ Validation Failed")

print("\n")

print(features_df.head())

# =============================================================================
# EXPORT
# =============================================================================

features_df.to_excel(
    r"C:\Users\CHIDERA\OneDrive\Documents\DSSR4\Feature_Engineering_Output.xlsx",
    index=False
)

# Stage 4 Diagnostics

feature_list = []

for i, (filename, cv_text) in enumerate(zip(cvs_database.keys(), processed_cvs), start=1):

    print(f"\nProcessing {i}: {filename}")

    doc = nlp(cv_text)

    extracted_skills = {

        ent.text.lower()

        for ent in doc.ents

        if ent.label_ == "SKILL"

    }

    feature_list.append({

        "Candidate": filename

    })

    print("Length =", len(feature_list))

print("\nFinal Length =", len(feature_list))

print("\nSTAGE 4 COLUMNS")
print(features_df.columns.tolist())

# =============================================================================
# STAGE 5: SIMILARITY SCORING & CANDIDATE SUITABILITY SCORE
# =============================================================================

"""
This stage compares every anonymised CV with the Software Developer
Job Description using TF-IDF and Cosine Similarity.

The semantic similarity score is then combined with the technical
feature scores to calculate the Candidate Suitability Score (CSS).
"""

# =============================================================================
# BUILD TF-IDF MATRIX
# =============================================================================

documents = [job_description] + processed_cvs

vectorizer = TfidfVectorizer(

    stop_words="english"

)

tfidf_matrix = vectorizer.fit_transform(documents)

# =============================================================================
# CALCULATE COSINE SIMILARITY
# =============================================================================

similarity_scores = cosine_similarity(

    tfidf_matrix[0:1],

    tfidf_matrix[1:]

).flatten()

# =============================================================================
# ADD SIMILARITY SCORES
# =============================================================================

features_df["Cosine Similarity"] = similarity_scores.round(4)

features_df["Similarity Score"] = (

    similarity_scores * 100

).round(2)

# =============================================================================
# NORMALISE SIMILARITY
# =============================================================================

min_similarity = features_df["Similarity Score"].min()

max_similarity = features_df["Similarity Score"].max()

features_df["Normalised Similarity"] = (

    (
        features_df["Similarity Score"] - min_similarity
    )

    /

    (
        max_similarity - min_similarity
    )

    * 100

).round(2)

# =============================================================================
# CANDIDATE SUITABILITY SCORE (CSS)
# =============================================================================
print(features_df.columns.tolist())

features_df["Candidate Suitability Score"] = (

      0.40 * features_df["Similarity Score"]

    + 0.30 * features_df["Programming Match (%)"]

    + 0.15 * features_df["Web Development Match (%)"]

    + 0.15 * features_df["Software Engineering Match (%)"]

).round(2)

# =============================================================================
# RANK CANDIDATES
# =============================================================================

features_df = features_df.sort_values(

    by="Candidate Suitability Score",

    ascending=False

).reset_index(drop=True)

features_df["Rank"] = features_df.index + 1

# =============================================================================
# DISPLAY RESULTS
# =============================================================================

print("\n")
print("=" * 100)
print("SIMILARITY SCORING RESULTS")
print("=" * 100)

print(features_df[[

    "Rank",

    "Candidate",

    "Similarity Score",

    "Programming Match (%)",

    "Web Development Match (%)",

    "Software Engineering Match (%)",

    "Candidate Suitability Score"

]].head(20))

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\n")
print("=" * 100)
print("SUMMARY STATISTICS")
print("=" * 100)

print(features_df)[[
    "Similarity Score",
    "Candidate Suitability Score"]].describe()

# =============================================================================
# VALIDATION
# =============================================================================

print("\n")

print(f"Total Candidates Ranked : {len(features_df)}")

print(f"Highest CSS             : {features_df['Candidate Suitability Score'].max():.2f}")

print(f"Lowest CSS              : {features_df['Candidate Suitability Score'].min():.2f}")

# =============================================================================
# EXPORT RESULTS
# =============================================================================

features_df.to_excel(
    r"C:\Users\CHIDERA\OneDrive\Documents\DSSR4\Similarity_CSS.xlsx",
    index=False
)


# =============================================================================
# STAGE 6: GROUND TRUTH & RANDOM FOREST VALIDATION
# =============================================================================

"""
This stage validates the enhanced recruitment framework using a
Random Forest classifier.

The Ground Truth is created by shortlisting the highest-ranked
candidates according to the Candidate Suitability Score (CSS).

The Random Forest then learns whether the engineered recruitment
features can reproduce these shortlisting decisions.
"""


# =============================================================================
# CREATE LABELS FOR RF
# =============================================================================

TOP_PERCENT = 0.25

top_candidates = int(

    len(features_df) * TOP_PERCENT

)

features_df["Ground Truth"] = "Not Suitable"

features_df.loc[

    features_df["Rank"] <= top_candidates,

    "Ground Truth"

] = "Suitable"

print("\n")
print("="*100)
print("GROUND TRUTH DISTRIBUTION")
print("="*100)

print(features_df["Ground Truth"].value_counts())

# =============================================================================
# SELECT FEATURES
# =============================================================================

X = features_df[[

    "Similarity Score",

    "Programming Match (%)",

    "Web Development Match (%)",

    "Software Engineering Match (%)"

]]

y = features_df["Ground Truth"]

# =============================================================================
# TRAIN / TEST SPLIT
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)
print (X_train)
print (X_test)
# =============================================================================
# BUILD RANDOM FOREST
# =============================================================================

rf_model = RandomForestClassifier(

    n_estimators=300,

    random_state=42,

    class_weight="balanced"

)

rf_model.fit(

    X_train,

    y_train

)

# =============================================================================
# PREDICT TEST SET
# =============================================================================

predictions = rf_model.predict(

    X_test

)

# =============================================================================
# EVALUATE MODEL
# =============================================================================

print("\n")
print("="*100)
print("RANDOM FOREST PERFORMANCE")
print("="*100)

print(f"Accuracy : {accuracy_score(y_test,predictions):.3f}")

print(f"Precision: {precision_score(y_test,predictions,pos_label='Suitable',zero_division=0):.3f}")

print(f"Recall   : {recall_score(y_test,predictions,pos_label='Suitable',zero_division=0):.3f}")

print(f"F1 Score : {f1_score(y_test,predictions,pos_label='Suitable',zero_division=0):.3f}")

print("\n")

print("="*100)
print("CLASSIFICATION REPORT")
print("="*100)

print(

    classification_report(

        y_test,

        predictions,

        zero_division=0

    )

)

print("\n")

print("="*100)
print("CONFUSION MATRIX")
print("="*100)

print(
    confusion_matrix(
        y_test,
        predictions
    )
)

# =============================================================================
# PLOT CONFUSION MATRIX
# =============================================================================

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6,5))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Suitable", "Suitable"]
)

disp.plot(cmap="Blues", values_format="d")

plt.title("Random Forest Confusion Matrix")

plt.tight_layout()

plt.savefig(
    r"C:\Users\CHIDERA\OneDrive\Documents\DSSR4\Confusion_Matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": rf_model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print("\n")
print("="*100)
print("FEATURE IMPORTANCE")
print("="*100)

print(importance)

# =============================================================================
# APPLY MODEL TO ALL CANDIDATES
# =============================================================================

features_df["ML Prediction"] = rf_model.predict(X)

features_df["Predicted Suitability"] = features_df["ML Prediction"]

# =============================================================================
# VALIDATION
# =============================================================================

print("\n")
print("Predicted Distribution\n")
print(
    features_df["Predicted Suitability"].value_counts()
)

# =============================================================================
# EXPORT
# =============================================================================

features_df.to_excel(
    r"C:\Users\CHIDERA\OneDrive\Documents\DSSR4\RF_Output.xlsx",
    index=False
)


# =============================================================================
# STAGE 7: DECISION SUPPORT DASHBOARD
# =============================================================================

"""
This stage generates the final recruitment decision support dashboard.

The dashboard combines:

• Candidate Ranking
• Similarity Score
• Technical Competency Scores
• Candidate Suitability Score (CSS)
• Ground Truth
• Random Forest Prediction
• Recruiter Recommendation
"""

# =============================================================================
# CREATE RECOMMENDATION
# =============================================================================

def recommendation(css):

    if css >= 70:
        return "Highly Recommended"

    elif css >= 55:
        return "Recommended"

    elif css >= 40:
        return "Consider"

    else:
        return "Not Recommended"


features_df["Recommendation"] = (

    features_df["Candidate Suitability Score"]

    .apply(recommendation)

)

# =============================================================================
# FINAL DASHBOARD
# =============================================================================

dashboard = features_df[[

    "Rank",

    "Candidate",

    "Similarity Score",

    "Programming Match (%)",

    "Web Development Match (%)",

    "Software Engineering Match (%)",

    "Candidate Suitability Score",

    "Ground Truth",

    "Predicted Suitability",

    "Recommendation"

]]

# =============================================================================
# DISPLAY DASHBOARD
# =============================================================================

print("\n")
print("="*110)
print("FINAL DECISION SUPPORT DASHBOARD")
print("="*110)

print(dashboard.head(20))

# =============================================================================
# RECOMMENDATION DISTRIBUTION
# =============================================================================

print("\n")
print("="*110)
print("RECOMMENDATION DISTRIBUTION")
print("="*110)

print(

    dashboard["Recommendation"]

    .value_counts()

)

# =============================================================================
# TOP 10 CANDIDATES
# =============================================================================

print("\n")
print("="*110)
print("TOP 10 RECOMMENDED CANDIDATES")
print("="*110)

print(

    dashboard.head(10)

)

# =============================================================================
# EXPORT DASHBOARD
# =============================================================================

dashboard.to_excel(
    r"C:\Users\CHIDERA\OneDrive\Documents\DSSR4\Enhanced_Decision_Support_Dashboard.xlsx",
    index=False
)


# =============================================================================
# WORD CLOUD - JOB DESCRIPTION
# =============================================================================

custom_stopwords = STOPWORDS.union({

    "experience",
    "candidate",
    "work",
    "role",
    "team",
    "ability",
    "skills",
    "requirements",
    "using",
    "knowledge",
    "including",
    "development",
    "developer",
    "software"

})

wordcloud = WordCloud(

    width=1200,
    height=600,
    background_color="white",
    colormap="viridis",
    max_words=150

).generate(job_description)

# =============================================================================
# DISPLAY
# =============================================================================

plt.figure(figsize=(14,7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.title(
    "Word Cloud of the Software Developer Job Description",
    fontsize=16
)

plt.tight_layout()

plt.show()

plt.tight_layout()

plt.savefig(
    r"C:\Users\CHIDERA\OneDrive\Documents\DSSR4\Job_Description_WordCloud(1).png",
    dpi=300,
    bbox_inches="tight"
)


# =============================================================================
# TOP 3 ENHANCED CANDIDATES
# =============================================================================

top3_enhanced = features_df.sort_values(

    by="Candidate Suitability Score",

    ascending=False

).head(3)

print(top3_enhanced[["Rank",
                     "Candidate",
                     "Candidate Suitability Score"]])

top3_enhanced_candidates = top3_enhanced["Candidate"].tolist()

print(top3_enhanced_candidates)



# =============================================================================
# CUSTOM STOPWORDS
# =============================================================================

custom_stopwords = STOPWORDS.union({

    "experience",
    "candidate",
    "work",
    "role",
    "team",
    "ability",
    "skills",
    "using",
    "knowledge",
    "including",
    "used",
    "involved",

})

# =============================================================================
# WORD CLOUDS FOR TOP 3 ENHANCED CANDIDATES
# =============================================================================

for candidate in top3_enhanced_candidates:

    text = cvs_database[candidate]

    wordcloud = WordCloud(

        width=1200,
        height=600,
        background_color="white",
        stopwords=custom_stopwords,
        colormap="viridis",
        max_words=150

    ).generate(text)

    plt.figure(figsize=(12,6))

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")

    plt.title(f"Enhanced System - {candidate}")

    plt.tight_layout()

    plt.savefig(

        rf"C:\Users\CHIDERA\OneDrive\Documents\DSSR4\Enhanced_Top3_WordCloud(1).png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.show()





