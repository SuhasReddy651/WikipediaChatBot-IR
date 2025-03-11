import os
import streamlit as st
from pathlib import Path
import json
import sqlite3
import uuid
import time
import base64
import warnings
from chitchat_module import ChitChatSystem
from classifier_module import classify_query
from wiki_qna_module import fetch_relevant_documents
import sqlite3
import pandas as pd
import json
import base64
import uuid
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="Tried to instantiate class '__path__._path'")

# Initialize the ChitChat system
chit_chat_system = ChitChatSystem()
MODEL_DIR = Path("model/topic_classifier_model.joblib")

DB_PATH = "chatbot.db"
DATA_DIR = Path("data")
BOT_IMAGE_PATH = str(DATA_DIR / "pic.jpeg")  # GIF for bot
USER_IMAGE_PATH = str(DATA_DIR / "photo.jpeg")  # Static image for user

# Database setup
def init_db():
    """Initialize the database and create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Chat messages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            session_id TEXT,
            role TEXT,
            message TEXT,
            topic TEXT,
            relevance REAL,
            rating INTEGER,
            query_type TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Feedback table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            session_id TEXT,
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            message TEXT,
            rating INTEGER,
            feedback_text TEXT,
            feedback_type TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

def save_message(session_id, role, message, topic=None, relevance=None, rating=None, query_type=None):
    """Save a chat message to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO chat_messages (session_id, role, message, topic, relevance, rating, query_type)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        role,
        message,
        topic if topic is not None else "General",
        relevance if relevance is not None else None,
        rating if rating is not None else None,
        query_type if query_type is not None else "Unknown"
    ))
    
    # Get the last inserted message_id to use in feedback tracking
    message_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return message_id

def render_feedback_widget(message_id):
    """Render the Streamlit feedback widget for a message."""
    sentiment_mapping = ["Very Bad", "Bad", "Neutral", "Good", "Excellent"]
    selected = st.feedback("stars", key=f"feedback_{message_id}")

    if selected is not None:
        rating = selected + 1  # Adjust rating to range from 1 to 5
        save_feedback_to_db(message_id, rating)
        success_placeholder = st.empty()
        success_placeholder.success(f"Feedback saved: {sentiment_mapping[selected]} ({rating} stars)")
        time.sleep(2)  # Show the success message for 2 seconds
        success_placeholder.empty()

def save_feedback_to_db(message_id, rating):
    """Save feedback (rating) directly to the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE chat_messages 
            SET rating = ? 
            WHERE rowid = ?
        """, (rating, message_id))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")

def get_feedback(session_id=None):
    """Retrieve feedback, optionally filtered by session_id."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if session_id:
        cursor.execute("""
            SELECT * FROM feedback 
            WHERE session_id = ? 
            ORDER BY timestamp DESC
        """, (session_id,))
    else:
        cursor.execute("SELECT * FROM feedback ORDER BY timestamp DESC")
    
    feedback_list = cursor.fetchall()
    conn.close()
    return feedback_list

def get_messages(session_id):
    """Retrieve all chat messages for a specific session."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT rowid, role, message, topic, relevance, rating, query_type
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """, (session_id,))
        messages = cursor.fetchall()
        conn.close()
        return messages
    except Exception as e:
        st.error(f"Error retrieving messages: {e}")
        return []



def load_resources():
    """Load the required resources."""
    try:
        with open(DATA_DIR / "inverted_index.json", "r") as idx_file:
            inverted_index = json.load(idx_file)
        with open(DATA_DIR / "scraped_data.json", "r") as data_file:
            scraped_data = json.load(data_file)
        return inverted_index, scraped_data
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None


def encode_image(image_path):
    """Encode an image or GIF to base64 for rendering in HTML."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def render_message(role, message, message_id=None):
    """Render a message with alignment, images, and optional feedback."""
    if role == "assistant":
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <img src="data:image/jpeg;base64,{encode_image(BOT_IMAGE_PATH)}" alt="Bot Logo" style="width: 50px; height: 50px; margin-right: 10px; border-radius: 50%;">
                <div style="background-color: #e3f2fd; padding: 10px; border-radius: 10px; text-align: left; max-width: 70%; box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);">
                    {message}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if message_id:  # Render feedback for all assistant messages except the welcome message
            render_feedback_widget(message_id)
    else:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; align-items: center; margin-bottom: 10px;">
                <div style="background-color: #d1e7dd; padding: 10px; border-radius: 10px; text-align: right; max-width: 70%; box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);">
                    {message}
                </div>
                <img src="data:image/jpeg;base64,{encode_image(USER_IMAGE_PATH)}" alt="User Logo" style="width: 50px; height: 50px; margin-left: 10px; border-radius: 50%;">
            </div>
            """,
            unsafe_allow_html=True,
        )

def render_typing_animation():
    """Render typing animation"""
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <img src="data:image/jpeg;base64,{encode_image(BOT_IMAGE_PATH)}" alt="Bot Logo" style="width: 50px; height: 50px; margin-right: 10px; border-radius: 50%;">
            <div style="background-color: #e3f2fd; padding: 10px 15px; border-radius: 15px; text-align: left; display: inline-block; box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);">
                <style>
                    .typing-dots {{
                        display: flex;
                        align-items: center;
                        gap: 4px;
                    }}
                    .typing-dots span {{
                        animation: bounce 1.4s infinite ease-in-out;
                        background-color: #2196F3;
                        border-radius: 50%;
                        display: block;
                        height: 7px;
                        width: 7px;
                        opacity: 0.6;
                    }}
                    .typing-dots span:nth-child(1) {{ animation-delay: -0.32s; }}
                    .typing-dots span:nth-child(2) {{ animation-delay: -0.16s; }}
                    @keyframes bounce {{
                        0%, 80%, 100% {{ transform: scale(0.6); }}
                        40% {{ transform: scale(1); opacity: 1; }}
                    }}
                </style>
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def reset_chat(selected_option="Automatic"):
    """Reset the chat for a new session."""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.selected_option = selected_option
    st.session_state["messages"] = []
    st.session_state["chat_ended"] = False
    if "topic" not in st.session_state:
        st.session_state.topic = "Automatic"


def format_response(relevant_docs, answers):
    """Format the response with relevant documents and answers."""
    response = ""
    if relevant_docs:
        response += "\n\n**Relevant Documents Summary:**\n\n" + relevant_docs.strip() + "\n\n"
    else:
        response += "No relevant documents found.\n\n"

    if answers:
        response += "\n\n**Detailed Answers:**\n" + "\n".join([f"- {ans.strip()}" for ans in answers])
    else:
        response += "No detailed answers found."
    return response

def chatbot_interface():
    # Initialize the database
    init_db()
    
    # Session initialization
    if "session_id" not in st.session_state:
        reset_chat()
    if "topic" not in st.session_state:
        st.session_state.topic = "Automatic"

    # Sidebar for options
    st.sidebar.header("Options")

    # Add "New Chat" button in the sidebar
    if st.sidebar.button("New Chat", key="new_chat"):
        st.session_state.topic = "Automatic"  # Reset topic
        reset_chat()
        message_placeholder = st.sidebar.empty()
        message_placeholder.success("New chat started!")
        time.sleep(1)
        message_placeholder.empty()
    
    # Add "Visualize Analytics" button in the sidebar
    if st.sidebar.button("Visualize Analytics"):
        st.title("Chatbot Analytics Dashboard")
        visualize_data()
        return  # Exit chatbot interface when visualizing data

    session_id = st.session_state.session_id

    # Topic selection in sidebar
    options_list = [
        "Automatic",
        "Economy",
        "Education", 
        "Entertainment",
        "Environment",
        "Food",
        "Health",
        "Politics",
        "Sports",
        "Technology",
        "Travel", 
        "Food and Travel",
        "General"
    ]

    selected_option = st.sidebar.radio(
        "Select a Topic",
        options_list,
        key="topic",
        index=options_list.index(st.session_state.topic)
    )

    # Reset chat if the topic changes
    if selected_option != st.session_state.get("selected_option", "Automatic"):
        reset_chat(selected_option)
        message_placeholder = st.sidebar.empty()
        message_placeholder.success(f"New chat started with topic: {selected_option}!")
        time.sleep(1)
        message_placeholder.empty()

    # Header for chatbot UI
    st.markdown(
        """
        <div style="text-align: center;">
            <h1> lol bot 🤖 </h1>
            <p>by suhas and chandrahas</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Load chat history
    session_id = st.session_state.session_id
    chat_history = get_messages(session_id)

    for message_id, role, message, topic, relevance, rating, query_type in chat_history:
        render_message(role, message, message_id)

    # Display welcome message if chat is empty
    if not chat_history:
        welcome_msg = "Welcome to Lol Bot! Ask me anything or just chat casually. 😊"
        render_message("assistant", welcome_msg)  # No feedback for the welcome message
        save_message(session_id, "assistant", welcome_msg)

    if not st.session_state["chat_ended"]:
        # Chat input
        chat_input_placeholder = st.empty()
        user_input = chat_input_placeholder.chat_input("Type your message here...")
        
        if user_input:
            # Render the user's message first
            render_message("user", user_input)
            
            # Save user message with message_id tracking
            user_message_id = save_message(session_id, "user", user_input, topic=selected_option, query_type="informational")
            
            # End chat if user types exit commands
            if user_input.lower() in ["exit chat", "exit", "end", "end chat", "stop", "stop chat"]:
                # Set chat_ended to True and render the bot's end message
                st.session_state["chat_ended"] = True
                response = "The chat has ended. Click on the 'New Chat' button to start a new chat."
                render_message("assistant", response)
                save_message(session_id, "assistant", response)
                st.rerun()
                
            else:
            # Bot response generation
                typing_placeholder = st.empty()
                with typing_placeholder:
                    render_typing_animation()
                
                inverted_index, scraped_data = load_resources()
                if not inverted_index or not scraped_data:
                    response = "Failed to load resources. Please check your data files."
                else:
                    try:
                        if st.session_state.selected_option == "Automatic":
                            query_topic = classify_query(user_input, MODEL_DIR)
                            if query_topic == "General":
                                response = chit_chat_system.generate_chitchat_response(
                                    user_input=user_input,
                                    chat_history=[msg[1] for msg in chat_history if msg[0] == "user"]
                                )
                            else:
                                relevant_docs, answers, relevance_scores, _ = fetch_relevant_documents(
                                    user_input, query_topic, inverted_index, scraped_data
                                )
                                response = format_response(relevant_docs, answers)
                                for doc, score in zip(relevant_docs.split("\n\n"), relevance_scores):
                                    save_message(session_id, "assistant", doc, topic=query_topic, relevance=score)
                        
                        elif st.session_state.selected_option == "Food and Travel":
                            food_relevant_docs, food_answers, food_relevance_scores, _ = fetch_relevant_documents(
                                user_input, "Food", inverted_index, scraped_data
                            )
                            travel_relevant_docs, travel_answers, travel_relevance_scores, _ = fetch_relevant_documents(
                                user_input, "Food", inverted_index, scraped_data
                            )
                            relevant_docs, answers, relevance_scores = food_relevant_docs and travel_relevant_docs, food_answers and travel_answers, food_relevance_scores and travel_relevance_scores
                            response = format_response(relevant_docs, answers)
                            for doc, score in zip(relevant_docs.split("\n\n"), relevance_scores):
                                save_message(session_id, "assistant", doc, topic=selected_option, relevance=score)
                        
                        else:
                            relevant_docs, answers, relevance_scores, _ = fetch_relevant_documents(
                                user_input, st.session_state.selected_option, inverted_index, scraped_data
                            )
                            response = format_response(relevant_docs, answers)
                            for doc, score in zip(relevant_docs.split("\n\n"), relevance_scores):
                                save_message(session_id, "assistant", doc, topic=selected_option, relevance=score)

                    except Exception as e:
                        response = f"An error occurred: {e}"

                typing_placeholder.empty()
                
                # Save assistant message and track its message_id
                assistant_message_id = save_message(session_id, "assistant", response, topic=selected_option)
                render_message("assistant", response, assistant_message_id)  # Render feedback for all other messages
    

def visualize_data():
    # Load data from the database
    conn = sqlite3.connect(DB_PATH)
    messages_df = pd.read_sql_query("SELECT * FROM chat_messages", conn)
    feedback_df = pd.read_sql_query("SELECT * FROM chat_messages", conn)
    conn.close()



    # Debugging: Display the DataFrame
    st.write("Loaded Messages DataFrame:", messages_df)

    if messages_df.empty:
        st.warning("No data available for visualization.")
        return

    # Bar chart: Messages per topic
    if "topic" in messages_df.columns and not messages_df["topic"].isnull().all():
        topic_counts = messages_df['topic'].value_counts()
        st.subheader("Messages per Topic")
        st.bar_chart(topic_counts)
    else:
        st.warning("No topic data available for visualization.")

    # Pie chart: Response relevance (using feedback data)
    if not feedback_df.empty and "rating" in feedback_df.columns:
        rating_distribution = feedback_df['rating'].value_counts()
        st.subheader("Response Rating Distribution")
        st.bar_chart(rating_distribution)
    else:
        st.warning("No rating data available for visualization.")

    # Line chart: Average rating over time
    if not feedback_df.empty:
        feedback_df['timestamp'] = pd.to_datetime(feedback_df['timestamp'])
        
        # Cumulative count of ratings
        feedback_df['rating_count'] = feedback_df['rating'].notnull().cumsum()
        
        # Plot cumulative ratings over time
        st.subheader("Cumulative Ratings Over Time")
        st.line_chart(feedback_df.set_index('timestamp')['rating_count'])
    else:
        st.warning("No data available for visualization.")


    # User vs Assistant Messages Per Session
    if "role" in messages_df.columns:
        user_vs_bot = messages_df.groupby(['session_id', 'role']).size().unstack(fill_value=0)
        st.subheader("User vs Assistant Messages Per Session")
        st.bar_chart(user_vs_bot)
    else:
        st.warning("No data available to compare User vs Assistant messages.")

    # Average Relevance Scores by Topic
    if "relevance" in messages_df.columns and not messages_df['relevance'].isnull().all():
        relevance_by_topic = messages_df[messages_df['relevance'].notnull()].groupby('topic')['relevance'].mean()
        st.subheader("Average Relevance Scores by Topic")
        st.bar_chart(relevance_by_topic)
    else:
        st.warning("No relevance score data available for visualization.")

    # Response Latency (Time Between User Message and Bot Response)
    if "timestamp" in messages_df.columns:
        messages_df['timestamp'] = pd.to_datetime(messages_df['timestamp'])
        messages_df['latency'] = messages_df.groupby('session_id')['timestamp'].diff().dt.total_seconds()
        latency_over_time = messages_df[messages_df['latency'].notnull()].groupby(messages_df['timestamp'].dt.date)['latency'].mean()
        st.subheader("Average Response Latency Over Time")
        st.line_chart(latency_over_time)
    else:
        st.warning("No timestamp data available for latency analysis.")

    # Query Type Distribution
    if "query_type" in messages_df.columns and not messages_df["query_type"].isnull().all():
        query_type_distribution = messages_df['query_type'].value_counts()
        st.subheader("Query Type Distribution")
        fig, ax = plt.subplots()
        ax.pie(
            query_type_distribution,
            labels=query_type_distribution.index,
            autopct='%1.1f%%',
            startangle=140
        )
        ax.set_title("User Query Types")
        st.pyplot(fig)
    else:
        st.warning("No query type data available for visualization.")

    # Sentiment Confidence Score Distribution
    if "relevance" in messages_df.columns:
        relevance_scores = messages_df['relevance'].dropna()  # Exclude None values
        if not relevance_scores.empty:
            st.subheader("Sentiment Confidence Score Distribution")
            fig, ax = plt.subplots()
            ax.hist(relevance_scores, bins=10, edgecolor="k", alpha=0.7)
            ax.set_title("Confidence Score Distribution")
            ax.set_xlabel("Confidence Score")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        else:
            st.warning("No confidence score data available for visualization.")

    # Messages Over Time
    if "timestamp" in messages_df.columns:
        if not messages_df['timestamp'].isnull().all():
            messages_over_time = messages_df.groupby(messages_df['timestamp'].dt.date).size()
            st.subheader("Messages Over Time")
            st.line_chart(messages_over_time)
        else:
            st.warning("No timestamp data available for visualization.")





if __name__ == "__main__":
    chatbot_interface()