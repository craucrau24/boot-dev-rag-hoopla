#!/usr/bin/env python3

import argparse

from data.semantic_search import SemanticSearch, verify_model

def main():
  parser = argparse.ArgumentParser(description="Semantic Search CLI")
  subparsers = parser.add_subparsers(dest="command", help="Available commands")
  subparsers.add_parser("verify", help="verify model")

  args = parser.parse_args()

  match args.command:
    case "verify":
      verify_model()

    case _:
        parser.print_help()

if __name__ == "__main__":
  main()