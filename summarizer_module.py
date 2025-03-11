import streamlit as st
from google import genai

def generate_summary_with_gemini(summaries):
    """
    Generate a cohesive summary from multiple summaries using Google Gemini.

    Args:
        summaries (list of str): List of document summaries.

    Returns:
        str: The final summarized text.
    """
    # Configure API key
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

    # Initialize Gemini model
    model = genai.GenerativeModel("gemini-2.0-flash-lite")

    # Construct the prompt using summaries
    combined_input = "Summarize the following information concisely within 300 tokens:\n"
    for i, summary in enumerate(summaries, 1):
        combined_input += f"{i}. {summary}\n"

    # Call the Gemini API
    response = model.generate_content(combined_input)

    # Extract and return the summarized text
    return response.text.strip() if hasattr(response, "text") else response.candidates[0].content.strip()

def summarize_documents(documents):
    """
    Wrapper for generate_summary_with_gemini to maintain compatibility with app.py.

    Args:
        documents (list of str): List of document texts to summarize.

    Returns:
        str: Final summarized text.
    """
    return generate_summary_with_gemini(documents)
