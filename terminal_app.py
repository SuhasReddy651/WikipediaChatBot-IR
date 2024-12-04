import json
from pathlib import Path
from chitchat_module import ChitChatSystem
from classifier_module import classify_query
from summarizer_module import summarize_documents
from wiki_qna_module import fetch_relevant_documents

# Paths to resources
DATA_DIR = Path("data")
MODEL_DIR = Path("model/topic_classifier_model.joblib")

# Initialize the ChitChat system
chit_chat_system = ChitChatSystem()

# Load required resources
def load_resources():
    print("Loading resources...")
    try:
        # Load inverted index and scraped data
        with open(DATA_DIR / "inverted_index.json", "r") as idx_file:
            inverted_index = json.load(idx_file)
        with open(DATA_DIR / "scraped_data.json", "r") as data_file:
            scraped_data = json.load(data_file)
        print("Resources loaded successfully!")
        return inverted_index, scraped_data
    except Exception as e:
        print(f"Error loading resources: {e}")
        return None, None

# Chatbot loop with continuation
def chatbot_interface():
    # Load resources
    inverted_index, scraped_data = load_resources()
    if not inverted_index or not scraped_data:
        print("Failed to load resources. Exiting.")
        return

    print("\nWelcome to the IR Chatbot!")
    print("You can ask questions about topics or chat casually.")
    print("Type 'exit' to end the chat.\n")

    chat_history = []

    while True:
        user_query = input("You: ").strip()
        if user_query.lower() == "exit":
            print("Goodbye!")
            break

        # Add user's query to the chat history
        chat_history.append({"role": "user", "content": user_query})

        try:
            # Classify the query
            query_topic = classify_query(user_query, MODEL_DIR)
        except Exception as e:
            response = f"Error classifying query: {e}"
            print(f"Bot: {response}")
            chat_history.append({"role": "assistant", "content": response})
            continue

        if query_topic == "General":
            # Handle chit-chat
            try:
                # Pass the properly formatted chat history to the chit-chat system
                response = chit_chat_system.generate_chitchat_response(
                    user_input=user_query,
                    chat_history=[msg["content"] for msg in chat_history if msg["role"] == "user"]
                )
            except Exception as e:
                response = f"Error generating chit-chat response: {e}"
        else:
            # Handle topic-specific queries
            try:
                # Fetch relevant documents and answers
                response, answers = fetch_relevant_documents(user_query, query_topic, inverted_index, scraped_data)
                
                # Check if a response was generated
                if not response:
                    response = (
                        "Sorry, I couldn't find any relevant information. "
                        "Please try refining your query or provide more specific details."
                    )
                else:
                    # Beautify the response and format answers
                    formatted_answers = "\n\n".join([f"- {answer.strip()}" for answer in answers]) if answers else "No specific answers found."
                    
                    response = (
                        "Here are the most relevant details based on your query:\n\n"
                        f"{response.strip()}\n\n"
                        "Details from the sources:\n"
                        f"{formatted_answers}\n\n"
                        "If you need more information, feel free to ask!"
                    )

            except Exception as e:
                # Handle and format exceptions gracefully
                response = (
                    "Oops! Something went wrong while processing your request. "
                    "Here's the error details:\n\n"
                    f"{str(e)}\n\n"
                    "Please check your query and try again."
                )

            # Add bot's response to the chat history and display it
        print(f"Bot: {response}")
        chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    chatbot_interface()
