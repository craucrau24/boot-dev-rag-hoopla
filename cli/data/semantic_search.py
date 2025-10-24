import json
import os

from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticSearch:
  def __init__(self):
    # Load the model (downloads automatically the first time)
    self.model = SentenceTransformer('all-MiniLM-L6-v2')

    self.embeddings = None
    self.documents = None
    self.document_map = {}

  def generate_embedding(self, text: str):
    text = text.strip()
    if text == "":
      raise ValueError("Input text is blank")
    return self.model.encode([text])[0]

  def build_embeddings(self, documents):
    self.documents = documents
    
    doc_strings = []
    for doc in self.documents:
      self.document_map[doc["id"]] = doc
      doc_strings.append(f"{doc['title']}: {doc['description']}")

    self.embeddings = self.model.encode(doc_strings, show_progress_bar=True)
    np.save(os.path.join("cache", "movie_embeddings.npy"), self.embeddings)

    return self.embeddings

  def load_or_create_embeddings(self, documents):
    self.documents = documents
    
    self.document_map = {doc["id"]: doc for doc in self.documents}
    cache_name = os.path.join("cache", "movie_embeddings.npy")
    if os.path.exists(cache_name):
      self.embeddings = np.load(cache_name)
    
      if len(self.embeddings) == len(self.documents):
        return self.embeddings
    
    return self.build_embeddings(documents)


def verify_model():
  sem = SemanticSearch()
  print(f"Model loaded: {sem.model}")
  print(f"Max sequence length: {sem.model.max_seq_length}")

def verify_embeddings():
  sem = SemanticSearch()
  with open(os.path.join("data", "movies.json")) as mov:
    documents = json.load(mov)["movies"]
  embeddings = sem.load_or_create_embeddings(documents)

  print(f"Number of docs:   {len(documents)}")
  print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
    


def embed_text(text: str):
  sem = SemanticSearch()
  embedding = sem.generate_embedding(text)
  print(f"Text: {text}")
  print(f"First 3 dimensions: {embedding[:3]}")
  print(f"Dimensions: {embedding.shape[0]}")