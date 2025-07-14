import streamlit as st
import pandas as pd
import PyPDF2
from docx import Document
import json
import os
from openai import OpenAI
import zipfile
import io

def get_llm_response(transcript_text, features, model):
    """Gets the LLM response for the transcript."""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=st.secrets["OPENROUTER_API_KEY"],
    )
    
    prompt = f"""
  Given the following clinical transcript and list of features, determine whether each feature has been explicitly addressed in the transcript.

A feature is considered **"Addressed"** if the transcript provides any relevant information affirming, denying, or otherwise commenting on the feature — even if only partially or indirectly (e.g., "no difficulty swallowing" would Address "dysphagia").

A feature is considered **"Not Addressed"** if it is not mentioned or inferable at all. Importantly, if the transcript provides information about a related feature but not the one in question (e.g., mentions improvement with cold foods but no mention of hot foods), then the unmentioned feature is still marked as **"Not Addressed"**.

Respond in JSON format, where each key is a feature and each value is either `"Addressed"` or `"Not Addressed"`.

Transcript:
{transcript_text}

Features:
{features}

    """

    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "https://streamlit.io", 
            "X-Title": "Transcript Feature Checker",
        },
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(completion.choices[0].message.content)

def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def create_grading_docx(transcript_name, features, transcript_text, model):
    """Creates a DOCX file with a grading table."""
    document = Document()
    document.add_heading(f'Grading for {transcript_name}', 0)

    results = get_llm_response(transcript_text, features, model)

    # Create a table
    table = document.add_table(rows=1, cols=2)
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Feature'
    hdr_cells[1].text = 'Status'

    for feature, status in results.items():
        row_cells = table.add_row().cells
        row_cells[0].text = feature
        row_cells[1].text = status

    # Save the document
    file_path = f"draft_grading_{transcript_name}.docx"
    document.save(file_path)
    return file_path

def check_password():
    """Returns `True` if the user had a correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password():
    st.title('Transcript Feature Checker')

    st.sidebar.header('Model Selection')
    model_options = [
        "google/gemini-2.5-flash",
        "google/gemini-2.5-pro",
        "openai/gpt-4.1",
        "openai/gpt-4.1-mini",
        "anthropic/claude-3.7-sonnet"
    ]
    selected_model = st.sidebar.selectbox("Choose a model", model_options, index=0)


    st.header('1. Upload Topics List')
    uploaded_features_file = st.file_uploader("Upload an Excel or CSV file with a list of topics.", type=['csv', 'xlsx'])

    if uploaded_features_file:
        if uploaded_features_file.name.endswith('.csv'):
            features_df = pd.read_csv(uploaded_features_file)
        else:
            features_df = pd.read_excel(uploaded_features_file)
        
        if not features_df.empty:
            features = features_df.iloc[:, 0].tolist()
            st.write("Features to check for:")
            st.write(features)

            st.header('2. Upload Transcripts')
            uploaded_transcripts = st.file_uploader("Upload student transcripts (PDF files).", type=['pdf'], accept_multiple_files=True)

            if uploaded_transcripts:
                if st.button('Generate Grading Documents'):
                    with st.spinner('Generating grading documents...'):
                        zip_buffer = io.BytesIO()
                        docx_files = []
                        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                            for transcript_pdf in uploaded_transcripts:
                                transcript_text = extract_text_from_pdf(transcript_pdf)
                                transcript_name = os.path.splitext(transcript_pdf.name)[0]
                                
                                docx_file_path = create_grading_docx(transcript_name, features, transcript_text, selected_model)
                                docx_files.append(docx_file_path)
                                st.success(f"Generated grading document for {transcript_pdf.name}")
                                zip_file.write(docx_file_path)
                        
                        for file_path in docx_files:
                            os.remove(file_path)

                        st.download_button(
                            label="Download All as ZIP",
                            data=zip_buffer.getvalue(),
                            file_name="graded_transcripts.zip",
                            mime="application/zip"
                        )
        else:
            st.warning("The uploaded file is empty.")
