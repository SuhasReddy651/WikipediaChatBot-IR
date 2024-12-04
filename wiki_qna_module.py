import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess_index import Preprocessor
from summarizer_module import generate_summary_with_gpt

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess_index import Preprocessor
from summarizer_module import generate_summary_with_gpt


class QASystem:
    def __init__(self, index_file, documents_file):
        # Load documents
        with open(index_file, 'r') as file:
            self.inverted_index = json.load(file)
        with open(documents_file, 'r') as file:
            raw_data = json.load(file)
            # Flatten documents from all topics
            self.documents = [
                doc for topic_docs in raw_data.values() for doc in topic_docs
            ]
        self.preprocessor = Preprocessor()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self._prepare_tfidf()

    def _prepare_tfidf(self):
        """Prepare TF-IDF vectors for all documents."""
        # Use 'summary' field as the document content
        self.document_texts = [doc.get('summary', '') for doc in self.documents]
        # Filter out documents without a summary
        self.documents = [doc for doc in self.documents if doc.get('summary', '')]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.document_texts)

    def search_query(self, query):
        """Search for query terms using TF-IDF similarity."""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        # Retrieve document IDs of the top-matching documents with relevance scores
        top_indices = similarities.argsort()[-10:][::-1]  # Top 10 results
        return [
            {"doc_id": self.documents[i]['revision_id'], "relevance": similarities[i]}
            for i in top_indices if similarities[i] > 0
        ]

    def fetch_documents(self, doc_ids):
        """Fetch full documents using their IDs."""
        return [doc for doc in self.documents if doc['revision_id'] in doc_ids]

    def extract_answers(self, query, documents, top_n=3):
        """Extract or summarize answers from the top documents."""
        ranked_docs = sorted(documents, key=lambda doc: query in doc['summary'], reverse=True)
        return [doc['summary'] for doc in ranked_docs[:top_n]]

    def fetch_relevant_documents(self, query):
        """Fetch and summarize relevant documents for a query."""
        doc_results = self.search_query(query)
        doc_ids = [result["doc_id"] for result in doc_results]
        relevance_scores = [result["relevance"] for result in doc_results]
        documents = self.fetch_documents(doc_ids)
        answers = self.extract_answers(query, documents)
        return answers, relevance_scores



def fetch_relevant_documents(query, topic, inverted_index, scraped_data):
    """
    Wrapper function for QASystem to fetch and summarize relevant documents.

    Args:
        query (str): User query.
        topic (str): Classified topic of the query.
        inverted_index (dict): The inverted index.
        scraped_data (dict): The scraped data.

    Returns:
        tuple: Final summary, answers, relevance scores, and document IDs.
    """
    qa_system = QASystem('data/inverted_index.json', 'data/scraped_data.json')
    answers, relevance_scores = qa_system.fetch_relevant_documents(query)
    doc_results = qa_system.search_query(query)

    # Combine input for summary generation
    combined_input = f"Query: {query}\n\nSummarize the following answers based on the query:\n"
    for i, answer in enumerate(answers, 1):
        combined_input += f"{i}. {answer}\n"

    summary = generate_summary_with_gpt([combined_input])

    # Include document IDs for tracking purposes
    doc_ids = [result["doc_id"] for result in doc_results]

    return summary, answers, relevance_scores, doc_ids

