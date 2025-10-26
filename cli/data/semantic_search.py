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

  def search(self, query, limit):
    if self.embeddings is None:
      raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

    embeddings = self.generate_embedding(query)
    scores = []
    for i, doc_embed in enumerate(self.embeddings):
      score = cosine_similarity(embeddings, doc_embed)
      scores.append((score, self.document_map[i + 1]))

    scores.sort(key=lambda elt: elt[0], reverse=True)
    return list(
      map(
        lambda s: {
          "score": s[0],
          "title": s[1]["title"],
          "description": s[1]["description"]
        },
        scores[:limit])
      )
      


def cosine_similarity(vec1, vec2):
  dot_product = np.dot(vec1, vec2)
  norm1 = np.linalg.norm(vec1)
  norm2 = np.linalg.norm(vec2)

  if norm1 == 0 or norm2 == 0:
    return 0.0

  return dot_product / (norm1 * norm2)

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


def search_query(query: str, limit: int):
  sem = SemanticSearch()
  with open(os.path.join("data", "movies.json")) as mov:
    documents = json.load(mov)["movies"]
  sem.load_or_create_embeddings(documents)

  for i, result in enumerate(sem.search(query, limit)):
    print(f"{i + 1}. {result["title"]} (score: {result["score"]: .4f})\n\t{result["description"]}\n")
