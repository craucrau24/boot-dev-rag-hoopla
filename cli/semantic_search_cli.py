#!/usr/bin/env python3

import argparse

from data.semantic_search import verify_model, verify_embeddings, embed_text, embed_query_text, search_query
from data.utils import get_chunks_from_str

def main():
  parser = argparse.ArgumentParser(description="Semantic Search CLI")
  subparsers = parser.add_subparsers(dest="command", help="Available commands")
  subparsers.add_parser("verify", help="verify model")
  subparsers.add_parser("verify_embeddings", help="verify embeddings for movie documents")

  embed_text_parser = subparsers.add_parser("embed_text", help="Get embeddings for given text using default model")
  embed_text_parser.add_argument("text", type=str, help="Input text for embeddings retrieval")

  search_parser = subparsers.add_parser("search", help="Get embeddings for given text using default model")
  search_parser.add_argument("query", type=str, help="The search query")
  search_parser.add_argument("--limit", type=int, default=5, help="The maximum number of results")

  embed_query_text_parser = subparsers.add_parser("embedquery", help="Get embeddings for given text using default model")
  embed_query_text_parser.add_argument("query", type=str, help="Input text for embeddings retrieval")

  chunk_text_parser = subparsers.add_parser("chunk", help="Split input text into chunks")
  chunk_text_parser.add_argument("text", type=str, help="Input text to be split into chunks")
  chunk_text_parser.add_argument("--chunk-size", type=int, default=200, help="The maximum number of words for each chunk")
  chunk_text_parser.add_argument("--overlap", type=int, default=0, help="Number of words that should overlap over two adjacent chunks")

  args = parser.parse_args()

  match args.command:
    case "verify":
      verify_model()
    
    case "verify_embeddings":
      verify_embeddings()

    case "embed_text":
      embed_text(args.text)

    case "search":
      search_query(args.query, args.limit)

    case "embedquery":
      embed_query_text(args.query)

    case "chunk":
      chunks = get_chunks_from_str(args.text, args.chunk_size, args.overlap)
      print(f"Chunking {len(args.text)} characters")
      for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")

    case _:
        parser.print_help()

if __name__ == "__main__":
  main()