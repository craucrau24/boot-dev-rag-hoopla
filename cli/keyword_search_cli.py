#!/usr/bin/env python3

import argparse
import os
import json
import itertools
import sys


from data.utils import Tokenizer
from data.inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build index")
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    with open(os.path.join("data", "movies.json")) as mov:
        movies = json.load(mov)

    tokenizer = Tokenizer()
    tokenizer.load_stop_words(os.path.join("data", "stopwords.txt"))

    match args.command:
        case "search":
            index = InvertedIndex(tokenizer)
            try:
                index.load()
            except IOError as e:
                print(f"Error while loading index files: {e}")
                sys.exit(1)

            print(f"Searching for: {args.query}")


            curr = set()
            result = []
            for tok in tokenizer.tokenize_str(args.query):
                docs = index.get_documents(tok)
                result.extend(filter(lambda doc: doc["id"] not in curr, docs))
                if len(result) > 5:
                    result = result[:5]
                    break

                curr.update(map(lambda doc: doc["id"], docs))

            for i, mov in enumerate(result):
                print(f"{i + 1}. {mov["title"]}")

        case "build":
            index = InvertedIndex(tokenizer)
            index.build(movies["movies"])
            index.save()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
