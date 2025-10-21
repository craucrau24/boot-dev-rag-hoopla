from sentence_transformers import SentenceTransformer

class SemanticSearch:
  def __init__(self):
    # Load the model (downloads automatically the first time)
    self.model = SentenceTransformer('all-MiniLM-L6-v2')

def verify_model():
  sem = SemanticSearch()
  print(f"Model loaded: {sem.model}")
  print(f"Max sequence length: {sem.model.max_seq_length}")