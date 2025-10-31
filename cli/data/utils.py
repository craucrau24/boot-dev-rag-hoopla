import string
import re
from itertools import count, takewhile

from nltk.stem import PorterStemmer

class Tokenizer:
  def __init__(self) -> None:
    self.stopwords = set()
    self.stemmer = PorterStemmer()

  def load_stop_words(self, fname: str):
    with open(fname) as stop:
        self.stopwords = set(stop.read().splitlines())

  def tokenize_word(self, word: str):
    return self.stemmer.stem(word.lower())

  def tokenize_str(self, s: str):
      table = str.maketrans("", "", string.punctuation)
      new_s = s.translate(table)
      return list(filter(
         lambda w: w not in self.stopwords,
         map(self.tokenize_word, new_s.split())))

def get_chunks_from_str(text: str, chunk_size: int, overlap: int=0) -> list[str]:
  if chunk_size <= 0:
    raise ValueError(f"Chunk size needs to be strictly positive and non null: (actual {chunk_size})")

  words = text.split()
  offset = chunk_size - overlap
  if offset <= 0:
    raise ValueError(f"Overlap must be lower than chunk size: (chunk size {chunk_size}, overlap {overlap})")

  chunks = takewhile(lambda t: t[0] + overlap < len(words), zip(count(0, offset), (count(chunk_size, offset))))

  return list(
     map(lambda t: " ".join(words[t[0]:t[1]]),
         chunks
        )
    )

def get_semantic_chunks_from_str(text: str, chunk_size: int, overlap: int=0) -> list[str]:
  if chunk_size <= 0:
    raise ValueError(f"Chunk size needs to be strictly positive and non null: (actual {chunk_size})")

  sentences = re.split(r"(?<=[.!?])\s+", text)
  offset = chunk_size - overlap
  if offset <= 0:
    raise ValueError(f"Overlap must be lower than chunk size: (chunk size {chunk_size}, overlap {overlap})")

  chunks = takewhile(lambda t: t[0] + overlap < len(sentences), zip(count(0, offset), (count(chunk_size, offset))))

  return list(
     map(lambda t: " ".join(sentences[t[0]:t[1]]),
         chunks
        )
    )