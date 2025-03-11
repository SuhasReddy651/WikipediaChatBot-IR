import os
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
    # Retrieve secrets from Streamlit Cloud
    api_key = st.secrets["GEMINI_API_KEY"]

    # Initialize Google Gemini client
    client = genai.Client(api_key=api_key)

    # Construct the prompt using summaries
    combined_input = "Summarize the following information concisely within 300 tokens:\n"
    for i, summary in enumerate(summaries, 1):
        combined_input += f"{i}. {summary}\n"

    # Call the Gemini API
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=combined_input,
        generation_config={
            "max_tokens": 500,
            "temperature": 0.2,
            "top_p": 1.0,
        }
    )

    # Extract the summarized text from the response
    summarized_text = response.text if hasattr(response, "text") else response["candidates"][0]["content"]
    
    return summarized_text.strip()

def summarize_documents(documents):
    """
    Wrapper for generate_summary_with_gemini to maintain compatibility with app.py.

    Args:
        documents (list of str): List of document texts to summarize.

    Returns:
        str: Final summarized text.
    """
    return generate_summary_with_gemini(documents)
