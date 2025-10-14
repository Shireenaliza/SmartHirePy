import streamlit as st
import os
import io
import pandas as pd
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader
import altair as alt

# --- Configuration & Setup ---
load_dotenv()
st.set_page_config(
    page_title="Smart Hire Resume Screener",
    layout="wide", # Use wide layout for a modern look
    initial_sidebar_state="expanded" 
)

# Initialize the Gemini client
try:
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        st.error("GEMINI_API_KEY not found in environment variables. Please check your .env file.")
        st.stop()
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    st.error(f"Error initializing Gemini client: {e}")
    st.stop()

# --- Pydantic Schemas (Unchanged) ---
class ExtractedResumeData(BaseModel):
    name: str = Field(description="Full name of the candidate.")
    skills: List[str] = Field(description="A comprehensive list of up to 20 technical and soft skills.")
    total_experience_years: float = Field(description="Estimated total professional experience in years.")
    education_summary: str = Field(description="Highest level of education and degree/major.")

class FeatureImportance(BaseModel):
    feature_name: str = Field(description="The resume feature being analyzed (e.g., 'Python Skill', '5 Years Exp', 'AWS Keyword').")
    impact_value: float = Field(description="The numerical magnitude of the feature's influence (0.0 to 1.0).")
    impact_type: str = Field(description="The direction of the impact: 'Positive' (helps score) or 'Negative' (hurts score).")

class ResumeSuggestion(BaseModel):
    type: str = Field(description="Type of suggestion: 'Add Skill', 'Remove Skill', or 'Add Keyword'.")
    content: str = Field(description="The specific skill or keyword recommended to add/remove.")
    reason: str = Field(description="Brief reason for the suggestion to improve fit with the job description.")

class XAIResult(BaseModel):
    match_score: int = Field(description="The final match score between 1 and 10.")
    overall_justification: str = Field(description="A detailed paragraph explaining the final match score and general fit.")
    feature_importance: List[FeatureImportance]
    counterfactual_suggestions: List[ResumeSuggestion]

# --- Helper Functions (Unchanged) ---

def extract_text_from_pdf(uploaded_file):
    try:
        reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
        text = "".join(page.extract_text() or "" for page in reader.pages)
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

# LLM Functions (Unchanged - using @st.cache_data for performance)
@st.cache_data(show_spinner="Extracting structured resume data...")
def extract_data_llm(resume_text: str) -> ExtractedResumeData:
    prompt = f"Extract the key data points from the following raw resume text and format as JSON according to the Pydantic schema. RESUME TEXT:\n---\n{resume_text}"
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ExtractedResumeData,
        )
    )
    return ExtractedResumeData.model_validate_json(response.text)

@st.cache_data(show_spinner="Running Explainable AI (XAI) analysis...")
def analyze_and_explain_llm(extracted_data: ExtractedResumeData, job_description: str) -> XAIResult:
    resume_summary = (
        f"Name: {extracted_data.name}\n"
        f"Skills: {', '.join(extracted_data.skills)}\n"
        f"Total Experience: {extracted_data.total_experience_years} years\n"
        f"Education: {extracted_data.education_summary}\n"
    )
    
    prompt = (
        "You are an **Explainable AI (XAI) Resume Analyst**. Your goal is to analyze the candidate summary against the job description "
        "and output a structured JSON object. Perform the following steps:\n"
        "1. **Score:** Provide a match score (1-10) and an overall justification.\n"
        "2. **SHAP-like Feature Importance:** Identify up to 6 key features (skills, experience, keywords) from the resume that had the strongest influence "
        "on the final score. Assign an `impact_value` (0.0 to 1.0) and specify if the impact is `Positive` (matched JD) or `Negative` (missed JD requirement).\n"
        "3. **Counterfactual Suggestions:** Provide 3 specific, actionable suggestions to improve the resume for this job description. "
        "Suggest adding missing skills, keywords, or removing irrelevant ones.\n\n"
        
        f"JOB DESCRIPTION:\n---\n{job_description}\n---\n\n"
        f"CANDIDATE SUMMARY:\n---\n{resume_summary}\n---\n"
    )
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=XAIResult,
        )
    )
    
    return XAIResult.model_validate_json(response.text)

# --- FIXED: Visualization Functions ---

def radial_gauge_chart(score: int):
    """Generates a radial gauge chart for the match score using Altair (Arc only), 
    with the legend suppressed for a minimalist look."""
    
    # 1. Prepare data for the arc segments
    score_df = pd.DataFrame([
        {'category': 'Score', 'value': score, 'order': 1},
        {'category': 'Remaining', 'value': 10 - score, 'order': 2}
    ])

    # 2. Define the base chart and encoding (theta for pie/arc charts)
    base = alt.Chart(score_df).encode(
        theta=alt.Theta("value", stack=True)
    ).properties(
        width=250, 
        height=250
    )

    # 3. Create the Arc chart (the gauge)
    chart = base.mark_arc(outerRadius=120, innerRadius=90).encode(
        # FIX: Suppress the legend by setting legend=None
        color=alt.Color("category", 
                        scale=alt.Scale(domain=['Score', 'Remaining'], range=['#4CAF50', '#E0E0E0']),
                        legend=None # <--- THIS LINE HIDES THE ICON/LEGEND
                       ),
        order=alt.Order("order", sort="ascending"),
        tooltip=['category', 'value']
    ).properties(
        title="" 
    )

    st.altair_chart(chart, use_container_width=False)
    
    # 4. Use custom Markdown with CSS for centered text overlay
    st.markdown(
        f"<h1 style='text-align: center; color: #4CAF50; margin-top: -150px; margin-bottom: 120px; font-size: 50px;'>{score}/10</h1>",
        unsafe_allow_html=True
    )


# --- Streamlit UI Main Logic ---

def main():
    st.title("📄 Smart Hire")
    st.markdown(
        """
        <style>
        .tagline {
            font-size: 4.5em; /* Increase font size */
            font-weight: bold; /* Make it bold */
            color: #A0A0A0; /* Optional: Adjust color for contrast */
        }
        </style>
        <p class='tagline'>Unlock AI-Powered Resume Screening with Full Transparency and Actionable Advice.</p>
        """, 
        unsafe_allow_html=True
    )
    st.divider()

    # --- INPUT SECTION ---
    if 'run_analysis' not in st.session_state:
        st.session_state['run_analysis'] = False
        
    input_container = st.container(border=True)
    with input_container:
        st.subheader("Define the Target Role and Candidate")
        
        col_jd, col_upload = st.columns([3, 2])
        
        with col_jd:
            job_description = st.text_area(
                "1. Paste Job Description (JD)",
                height=250,
                value="Software Engineer III (Python/Backend)\n\nWe seek a seasoned backend engineer with 5+ years of experience in Python, FastAPI, and asynchronous programming. Must have strong expertise in distributed systems, PostgreSQL, and AWS (S3, Lambda). Experience with LLMs or NLP is a significant plus. Excellent communication is required."
            )

        with col_upload:
            uploaded_file = st.file_uploader(
                "2. Upload Candidate Resume (PDF only)",
                type=["pdf"],
                accept_multiple_files=False,
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("Run Analysis", type="primary", use_container_width=True, key='run_button_visible'):
                st.session_state['run_analysis'] = True            
    st.markdown("---")

    if st.session_state.get('run_analysis'):
        if not job_description or not uploaded_file:
            st.error("Please provide both a Job Description and a Resume to run the analysis.")
            st.session_state['run_analysis'] = False
            return

        # Extract text from PDF
        resume_text = extract_text_from_pdf(uploaded_file)
        if not resume_text:
            st.session_state['run_analysis'] = False
            return

        try:
            # Extract structured data
            extracted_data = extract_data_llm(resume_text)
            
            # Perform XAI Analysis
            xai_result = analyze_and_explain_llm(extracted_data, job_description)

            # --- Results Display ---
            st.header(f"Candidate Match Report: {extracted_data.name}")

            score_col, justification_col = st.columns([1, 3])
            
            with score_col:
                st.subheader("Match Score")
                radial_gauge_chart(xai_result.match_score) 

            with justification_col:
                st.subheader("Summary")
                with st.expander("Detailed Match Justification (Click to Expand)", expanded=True):
                    st.markdown(f"**Candidate:** {extracted_data.name}")
                    st.markdown(f"**Experience:** {extracted_data.total_experience_years} years")
                    st.markdown("---")
                    st.info(xai_result.overall_justification)
            
            st.markdown("---")

            # Analysis Report
            importance_col, counterfactual_col = st.columns([1, 1])

            with importance_col:
                st.markdown("### 🔍 Feature Importance")
                st.markdown("*(Resume features that positively or negatively impacted the score)*")

                # Bar chart
                importance_data = []
                for item in xai_result.feature_importance:
                    score = item.impact_value if item.impact_type == "Positive" else -item.impact_value
                    importance_data.append({
                        "Feature": item.feature_name,
                        "Impact Score": score,
                        "Color": item.impact_type 
                    })
                
                # Sort by absolute score for better visualization
                df_importance = pd.DataFrame(importance_data)
                df_importance['Abs Score'] = df_importance['Impact Score'].abs()
                df_importance = df_importance.sort_values(by="Abs Score", ascending=False)
                
                chart = alt.Chart(df_importance).mark_bar().encode(
                    x=alt.X('Impact Score', title='Impact Magnitude'),
                    y=alt.Y('Feature', sort='x', title='Feature'),
                    color=alt.Color('Color', 
                                    scale=alt.Scale(domain=['Positive', 'Negative'], range=['#4CAF50', '#F44336']),
                                    legend=alt.Legend(title="Impact")
                                   ),
                    tooltip=['Feature', 'Impact Score', 'Color']
                ).properties(
                    title="Feature Contribution Analysis"
                ).interactive() 

                st.altair_chart(chart, use_container_width=True)

            with counterfactual_col:
                st.markdown("### 🔄 Resume Optimization Steps")
                st.markdown("*(Actionable changes to significantly improve the match score)*")
                
                suggestions_data = []
                for suggestion in xai_result.counterfactual_suggestions:
                    icon = "➕" if suggestion.type in ["Add Skill", "Add Keyword"] else "➖"
                    suggestions_data.append({
                        "Action": f"{icon} {suggestion.type}",
                        "Content": suggestion.content,
                        "Reason": suggestion.reason
                    })
                    
                df_suggestions = pd.DataFrame(suggestions_data)
                
                st.dataframe(
                    df_suggestions,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Action": st.column_config.Column(width="small"),
                        "Content": st.column_config.Column(width="medium"),
                        "Reason": st.column_config.Column(width="large")
                    },
                )
                
            st.markdown("---")
           
            with st.expander("View Extracted Resume Data", expanded=False):
                st.subheader("Key Extracted Fields")

                data_dict = extracted_data.model_dump()
                
                table_data = [
                    {"Field": "Candidate Name", "Value": data_dict["name"]},
                    {"Field": "Total Experience (Years)", "Value": f"{data_dict['total_experience_years']} years"},
                    {"Field": "Education Summary", "Value": data_dict["education_summary"]},
                    {"Field": "Skills", "Value": ", ".join(data_dict["skills"])},
                ]

                df_raw = pd.DataFrame(table_data)

                st.dataframe(
                    df_raw,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Field": st.column_config.Column(width="medium", help="The field extracted by the LLM."),
                        "Value": st.column_config.Column(width="large", help="The extracted value from the resume."),
                    }
                )                        

            # Reset button
            if st.button("Start New Analysis", key='reset_button'):
                st.session_state['run_analysis'] = False
                st.rerun()


        except Exception as e:
            st.error(f"An error occurred during LLM processing: {e}")
            st.session_state['run_analysis'] = False 

if __name__ == "__main__":
    main()