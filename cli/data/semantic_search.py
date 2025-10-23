from sentence_transformers import SentenceTransformer

class SemanticSearch:
  def __init__(self):
    # Load the model (downloads automatically the first time)
    self.model = SentenceTransformer('all-MiniLM-L6-v2')

  def generate_embedding(self, text: str):
    text = text.strip()
    if text == "":
      raise ValueError("Input text is blank")
    return self.model.encode([text])[0]

def verify_model():
  sem = SemanticSearch()
  print(f"Model loaded: {sem.model}")
  print(f"Max sequence length: {sem.model.max_seq_length}")

def embed_text(text: str):
  sem = SemanticSearch()
  embedding = sem.generate_embedding(text)
  print(f"Text: {text}")
  print(f"First 3 dimensions: {embedding[:3]}")
  print(f"Dimensions: {embedding.shape[0]}")