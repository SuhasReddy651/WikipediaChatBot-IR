import streamlit as st
from google import genai

class ChitChatSystem:
    def __init__(self):
        """
        Initialize the Chit-Chat system with Google Gemini client using secrets from Streamlit.
        """
        self.client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        self.model = "gemini-2.0-flash-lite"  # Use Gemini model for fast responses

    def generate_chitchat_response(self, user_input, chat_history=[]):
        """
        Generate a chit-chat response using Google Gemini API.

        Args:
            user_input (str): The user's query or input.
            chat_history (list): Previous conversation history.

        Returns:
            str: The response generated by the model.
        """
        # Construct chat input with history
        messages = "\n".join(chat_history) + f"\nUser: {user_input}"

        # Call the Gemini API
        response = self.client.generate_content(messages)

        # Extract and return the response content
        return response.text.strip() if hasattr(response, "text") else response["candidates"][0]["content"].strip()
