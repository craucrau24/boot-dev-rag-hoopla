import pickle
import os
import math

from collections import Counter

from data.definitions import BM25_K1, BM25_B

class InvertedIndex:
  def __init__(self, tokenizer):
    self.index = {}
    self.docmap = {}
    self.term_frequencies = {}
    self.doc_lengths = {}

    self.tokenizer = tokenizer

  def __add_document(self, doc_id, text):
    tokens = self.tokenizer.tokenize_str(text)
    count = Counter()

    for tok in tokens:
      self.index.setdefault(tok, set()).add(doc_id)
      count[tok] += 1

    self.term_frequencies[doc_id] = count
    self.doc_lengths[doc_id] = len(tokens)

  def __get_avg_doc_length(self) -> float:
    avg = sum(self.doc_lengths.values())
    if self.doc_lengths:
      avg /= len(self.doc_lengths)
    return avg

  def get_documents(self, term):
    docs = sorted(map(lambda id: self.docmap[id], self.index.get(term, set())), key=lambda elt: elt["id"])
    return docs

  def get_tf(self, doc_id: int, term: str) -> int:
    tokens = self.tokenizer.tokenize_str(term)
    
    try:
      [token] = tokens
    except:
      raise Exception("ill-formed query: either empty or more than one word")

    if doc_id not in self.term_frequencies:
      raise Exception(f"document {doc_id} not found")
    return self.term_frequencies[doc_id].get(token, 0)
    
  def get_bm25_idf(self, term: str) -> float:
    tokens = self.tokenizer.tokenize_str(term)
    try:
      [token] = tokens
    except:
      raise Exception("ill-formed query: either empty or more than one word")

    term_doc_count = len(self.get_documents(token))
    return math.log((len(self.docmap) - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)

  def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    doc_length = self.doc_lengths[doc_id]
    avg_doc_length = self.__get_avg_doc_length()
    length_norm = 1 - b + b * (doc_length / avg_doc_length)

    tf = self.get_tf(doc_id, term)
    result = (tf * (k1 + 1)) / (tf + k1 * length_norm)
    return result

  def build(self, movies):
    for mov in movies:
      self.docmap[mov["id"]] = mov
      self.__add_document(mov["id"], f"{mov["title"]} {mov["description"]}")
  
  def save(self):
    try: os.mkdir("cache")
    except FileExistsError: pass

    with open(os.path.join("cache", "index.pkl"), "bw") as f:
      pickle.dump(self.index, f)

    with open(os.path.join("cache", "docmap.pkl"), "bw") as f:
      pickle.dump(self.docmap, f)

    with open(os.path.join("cache", "term_frequencies.pkl"), "bw") as f:
      pickle.dump(self.term_frequencies, f)

    with open(os.path.join("cache", "doc_lengths.pkl"), "bw") as f:
      pickle.dump(self.doc_lengths, f)
  
  def load(self):
    with open(os.path.join("cache", "index.pkl"), "br") as f:
      self.index = pickle.load(f)

    with open(os.path.join("cache", "docmap.pkl"), "br") as f:
      self.docmap = pickle.load(f)

    with open(os.path.join("cache", "term_frequencies.pkl"), "br") as f:
      self.term_frequencies = pickle.load(f)

    with open(os.path.join("cache", "doc_lengths.pkl"), "br") as f:
      self.doc_lengths = pickle.load(f)