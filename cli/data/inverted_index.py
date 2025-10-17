import itertools
import pickle
import os

class InvertedIndex:
  def __init__(self, tokenizer):
    self.index = {}
    self.docmap = {}

    self.tokenizer = tokenizer

  def __add_document(self, doc_id, text):
    tokens = self.tokenizer.tokenize_str(text)
    for tok in tokens:
      self.index.setdefault(tok, set()).add(doc_id)

  def get_documents(self, term):
    print(term)
    docs = sorted(map(lambda id: self.docmap[id], self.index.get(term, set())), key=lambda elt: elt["id"])
    return docs

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

  
  def load(self):
    with open(os.path.join("cache", "index.pkl"), "br") as f:
      self.index = pickle.load(f)

    with open(os.path.join("cache", "docmap.pkl"), "br") as f:
      self.docmap = pickle.load(f)