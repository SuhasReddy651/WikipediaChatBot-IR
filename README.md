# 📚 Overview

The Wikipedia QnA Chatbot is an advanced Information Retrieval (IR) system designed to answer questions and chit-chat. It integrates web scraping, topic classification, document retrieval, summarization, and conversational AI into an intuitive interface powered by Streamlit.

# 🚀 Key Features
1.	**Wiki Scraper**: Efficiently collects data from 50,000+ Wikipedia articles across diverse topics.
2.	**Topic Classifier**: Automatically assigns user queries to specific categories.
3.	**Chit-Chat Module**: Engaging conversational AI powered by Azure OpenAI’s GPT-4o.
4.	**QnA System**: Retrieves relevant Wikipedia documents and generates precise answers.
5.	**Dynamic Summarization**: Delivers concise summaries using Azure OpenAI’s GPT-4o.
6.	**Interactive UI**: Built with Streamlit for seamless interaction and visualization.
7.	**Database Integration**: Persistent storage using SQLite for conversation history and analytics.

# 🛠️ Tech Stack
- Programming Language: Python.
- Frameworks & Libraries: Streamlit, OpenAI API, NLTK, TF-IDF, SVM, SQLite.
- Cloud Services: Azure, OpenAI.
- Hosting Service - Streamlit Community Cloud

# 🏗️ Project Structure
- `scraper.py` → Wikipedia article scraper  
- `classifier.py` → Topic classification module  
- `chitchat.py` → Chit-chat functionality with GPT-4o  
- `qna.py` → Document retrieval and summarization  
- `app.py` → Streamlit-based user interface  
- `database.db` → SQLite database for persistence

# 📊 How to Run the Project Locally
1.	Clone the Repository:
   ```bash
  git clone https://github.com/SuhasReddy651/Wiki-Chat-Bot.git
  cd wiki-qna-chatbot
```
2.	Set Up Environment Variables:
Add the following to `.streamlit/secrets.toml`:
```bash
AZURE_OPENAI_ENDPOINT="https://your-endpoint"
AZURE_OPENAI_API_KEY="your-api-key"
AZURE_OPENAI_API_VERSION="2024-08-01-preview"
```

3. Install Dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Application:
```bash
streamlit run app.py
```

# 🧠 Team Members
•	**Surya Suhas Reddy Sathi**: Scraping, Classifier Implementation, QnA Module, UI Development.  
•	**Chandrahas Reddy Gurram**: Chit-Chat Implementation, Summarizer, Analysis & Visualization.


