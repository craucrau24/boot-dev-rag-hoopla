#!/usr/bin/env python3

import argparse

from data.semantic_search import verify_model, verify_embeddings, embed_text

def main():
  parser = argparse.ArgumentParser(description="Semantic Search CLI")
  subparsers = parser.add_subparsers(dest="command", help="Available commands")
  subparsers.add_parser("verify", help="verify model")
  subparsers.add_parser("verify_embeddings", help="verify embeddings for movie documents")

  embed_text_parser = subparsers.add_parser("embed_text", help="Get embeddings for given text using default model")
  embed_text_parser.add_argument("text", type=str, help="Input text for embeddings retrieval")

  args = parser.parse_args()

  match args.command:
    case "verify":
      verify_model()
    
    case "verify_embeddings":
      verify_embeddings()

    case "embed_text":
      embed_text(args.text)

    case _:
        parser.print_help()

if __name__ == "__main__":
  main()