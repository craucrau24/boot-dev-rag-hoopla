import os
import json
import numpy as np
from heapq import nlargest

from data.semantic_search import SemanticSearch, cosine_similarity
from data.utils import get_semantic_chunks_from_str

SCORE_PRECISION = 4

class ChunkedSemantticSearch(SemanticSearch):
  def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
    super().__init__(model_name)
    self.chunk_embeddings = None
    self.chunk_metadata = None

  def build_chunk_embeddings(self, documents):
    # self.build_embeddings(documents)
    
    self.chunk_embeddings = []
    self.chunk_metadata = []

    chunks = []
    for doc_idx, doc in enumerate(self.documents):
      if not doc["description"]:
        continue
      doc_chunks = get_semantic_chunks_from_str(doc["description"], 4, 1)

      for chunk_idx, chunk in enumerate(doc_chunks):
        chunks.append(chunk)
        self.chunk_metadata.append({
          "movie_idx": doc_idx,
          "chunk_idx": chunk_idx,
          "total_chunks": len(doc_chunks)
        })
    
    self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)

    np.save(os.path.join("cache", "chunk_embeddings.npy"), self.chunk_embeddings)
    with open(os.path.join("cache", "chunk_metadata.json"), "w") as f:
      json.dump({"chunks": self.chunk_metadata, "total_chunks": len(self.chunk_metadata)}, f)

    return self.chunk_embeddings

  def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
    self.load_or_create_embeddings(documents)
    embedding_cache_name = os.path.join("cache", "chunk_embeddings.npy")
    metadata_cache_name = os.path.join("cache", "chunk_metadata.json")

    if os.path.exists(embedding_cache_name) and os.path.exists(metadata_cache_name):
      self.chunk_embeddings = np.load(embedding_cache_name)
      
      with open(metadata_cache_name) as f:
        self.chunk_metadata = json.load(f)
      
      return self.chunk_embeddings

    return self.build_chunk_embeddings(documents)

  def search_chunks(self, query: str, limit: int=10):
    query_emb = self.generate_embedding(query)
    # print(self.chunk_metadata[:10])
    chunk_scores = [{"chunk_idx": metadata["chunk_idx"], "movie_idx": metadata["movie_idx"], "score": cosine_similarity(query_emb, embed)} for embed, metadata in zip(self.chunk_embeddings, self.chunk_metadata["chunks"], strict=True)]

    score_map = {}
    for score in chunk_scores:
      movie_idx = score["movie_idx"]
      score_map[movie_idx] = max(score_map.setdefault("movie_idx", 0), score["score"])
    
    def format_elt(elt):
      movie = self.document_map[elt[0]]
      return {
        "id": movie["id"],
        "title": movie["title"],
        "document": movie["description"][:100],
        "score": round(elt[1], SCORE_PRECISION),
        "metadata": {}
      }

    print(max(score_map.values()))
    result = list(map(
      format_elt,
      nlargest(limit, score_map.items(), key=lambda elt: elt[1])
    ))

    return result
