import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import math
from collections import OrderedDict
import math

# Preprocessor Code

# if stop words are not downloaded, run : "nltk.download('stopwords')"

nltk.data.path.append('data/nltk_data')

class Preprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.ps = PorterStemmer()

    def tokenizer(self, text):
        text = text.lower()
        text = re.sub(r"[^A-Za-z0-9]+", " ", text)
        text = re.sub(r"\s+", " ", text)  
        tokens = [self.ps.stem(word) for word in text.split() if word not in self.stop_words]
        return tokens

# Linked List Code

class Node:

    def __init__(self, value=None, next=None):
        self.value = value
        self.next = next
        self.skip = None
        self.score = 0.0


class LinkedList:
    def __init__(self):
        self.start_node = None
        self.end_node = None
        self.length, self.n_skips, self.idf = 0, 0, 0.0
        self.skip_length = None

    def traverse_list(self):
        result = []
        current = self.start_node
        while current:
            result.append(current.value)
            current = current.next
        return result

    def traverse_skips(self):
        result = []
        current = self.start_node
        while current:
            result.append(current.value)
            if current.skip:
                current = current.skip
            else:
                current = current.next
        return result

    def add_skip_connections(self):
        if self.length <= 2:
            return
            
        n_skips = math.floor(math.sqrt(self.length))
 
        skip_length = math.floor(self.length / (n_skips + 1))
        
        current = self.start_node
        for i in range(self.length):
            if i % skip_length == 0 and i + skip_length < self.length:
                skip_to = current
                steps = skip_length
                while steps > 0 and skip_to:
                    skip_to = skip_to.next
                    steps -= 1
                if skip_to:
                    current.skip = skip_to
            current = current.next

        self.n_skips = n_skips
        self.skip_length = skip_length

    def insert_at_end(self, value):
        new_node = Node(value)
        if self.start_node is None:
            self.start_node = self.end_node = new_node
        else:
            current = self.start_node
            prev = None
            while current and current.value < value:
                prev = current
                current = current.next
            if prev is None:
                new_node.next = self.start_node
                self.start_node = new_node
            else:
                new_node.next = current
                prev.next = new_node
            if new_node.next is None:
                self.end_node = new_node
        self.length += 1

    def to_list(self):
        result = []
        current = self.start_node
        while current:
            result.append(current.value)
            current = current.next
        return result

# Indexer Code

class Indexer:
    def __init__(self):
        self.inverted_index = OrderedDict({})
        self.doc_count = 0

    def get_index(self):
        return self.inverted_index

    def generate_inverted_index(self, doc_id, tokenized_document):
        for t in tokenized_document:
            self.add_to_index(t, doc_id)
        self.doc_count += 1

    def add_to_index(self, term_, doc_id_):
        if term_ not in self.inverted_index:
            self.inverted_index[term_] = LinkedList()
        self.inverted_index[term_].insert_at_end(doc_id_)

    def sort_terms(self):
        sorted_index = OrderedDict({})
        for k in sorted(self.inverted_index.keys()):
            sorted_index[k] = self.inverted_index[k]
        self.inverted_index = sorted_index

    def add_skip_connections(self):
        for term, postings_list in self.inverted_index.items():
            postings_list.add_skip_connections()

    def calculate_tf_idf(self):
        for term in self.inverted_index:
            postings_list = self.inverted_index[term]
            df = postings_list.length
            idf = math.log10(self.doc_count / df)
            
            current = postings_list.start_node
            while current:
                tf = 1
                current.score = tf * idf
                current = current.next
