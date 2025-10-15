# 📄 Smart Hire: Explainable Resume Screener

Unlock intelligent, transparent, and actionable hiring decisions using the Gemini LLM. This application semantically matches a candidate's resume against a job description, providing not just a score, but a full explanation of *why*.

* **Live Demo Video (YouTube):** https://youtu.be/u7qs6oL8x8w?si=hAS69wOis0GOvX9k
* **GitHub Repository (Codebase):** https://github.com/Shireenaliza/SmartHirePy
  
## Features

This tool goes beyond keyword matching to provide a comprehensive analysis:

1.  **Semantic Match Scoring (1-10):** Generates an accurate fit score between the candidate and the role.
2.  **SHAP-like Feature Importance:** Visualizes which specific factors (skills, experience, keywords) positively (Green 🟢) or negatively (Red 🔴) impacted the final score.
3.  **Actionable Improvement Plan:** Provides clear, counterfactual suggestions for optimizing the resume to achieve a higher score for the given role.
4.  **Minimalist UI:** Uses Streamlit and Altair for a clean, dynamic, and intuitive user experience.

## How to set up;

Follow these steps to set up and run the application locally.

### 1. Prerequisites

* Python 3.8+
* A Google AI API Key (required for the Gemini model)

### 2. Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd SmartHirePy
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use: venv\Scripts\activate
    pip install -r requirements.txt
    ```
    *The `requirements.txt` file should contain:*
    ```
    streamlit
    google-genai
    pypdf
    pydantic
    python-dotenv
    altair
    pandas
    ```

3.  **Configure API Key:**
    Create a file named `.env` in the root directory and add your Gemini API key:
    ```
    GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```

### 3. Run the App

Execute the Streamlit command to launch the application:

```bash
streamlit run app.py
```


## Technology Stack

* **Core Logic:** Python
* **AI/LLM:** Google Gemini (via `google-genai`)
* **Frontend/UI:** Streamlit
* **Data Validation:** Pydantic
* **Visualization:** Altair (for the dynamic gauges and feature importance charts)


