import string
import os

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
         map(self.tokenize_word, new_s.split(" "))))