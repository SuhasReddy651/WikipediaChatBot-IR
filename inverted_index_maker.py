import json
from tqdm import tqdm
from preprocess_index import Preprocessor, Indexer

class InvertedIndexer:
    def __init__(self, json_file_path, index_output_file):
        self.json_file_path = json_file_path
        self.index_output_file = index_output_file
        self.preprocessor = Preprocessor()
        self.indexer = Indexer()

    def process_and_index(self):
        with open(self.json_file_path, 'r') as file:
            data = json.load(file)

        print("Processing and indexing documents...")
        for topic, documents in tqdm(data.items(), desc="Processing Topics"):
            for doc in documents:
                doc_id = doc['revision_id']
                text = f"{doc['title']} {doc['summary']}"  
                tokenized_text = self.preprocessor.tokenizer(text)
                self.indexer.generate_inverted_index(doc_id, tokenized_text)
                
        self.indexer.sort_terms()
        self.indexer.add_skip_connections()
        self.indexer.calculate_tf_idf()

        print(f"Saving the inverted index to {self.index_output_file}...")
        inverted_index = {term: postings.traverse_list() for term, postings in self.indexer.get_index().items()}
        with open(self.index_output_file, 'w') as file:
            json.dump(inverted_index, file, indent=4)

        print("Indexing complete!")

if __name__ == "__main__":
    json_file_path = 'data/scraped_data.json'
    index_output_file = 'data/inverted_index.json'

    inverted_indexer = InvertedIndexer(json_file_path, index_output_file)
    inverted_indexer.process_and_index()