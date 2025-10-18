import pickle
import os
import math

from collections import Counter

class InvertedIndex:
  def __init__(self, tokenizer):
    self.index = {}
    self.docmap = {}
    self.term_frequencies = {}

    self.tokenizer = tokenizer

  def __add_document(self, doc_id, text):
    tokens = self.tokenizer.tokenize_str(text)
    count = Counter()

    for tok in tokens:
      self.index.setdefault(tok, set()).add(doc_id)
      count[tok] += 1

    self.term_frequencies[doc_id] = count

  def get_documents(self, term):
    docs = sorted(map(lambda id: self.docmap[id], self.index.get(term, set())), key=lambda elt: elt["id"])
    return docs

  def get_tf(self, doc_id: str, term: str) -> int:
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

  
  def load(self):
    with open(os.path.join("cache", "index.pkl"), "br") as f:
      self.index = pickle.load(f)

    with open(os.path.join("cache", "docmap.pkl"), "br") as f:
      self.docmap = pickle.load(f)

    with open(os.path.join("cache", "term_frequencies.pkl"), "br") as f:
      self.term_frequencies = pickle.load(f)