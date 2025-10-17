#!/usr/bin/env python3

import argparse
import os
import json
import math
import sys


from data.utils import Tokenizer
from data.inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build index")
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    term_freq_parser = subparsers.add_parser("tf", help="get term frequency for give document id and term")
    term_freq_parser.add_argument("doc_id", type=int, help="Document id")
    term_freq_parser.add_argument("term", type=str, help="Term to search")

    inv_doc_freq_parser = subparsers.add_parser("idf", help="get inverse document frequency for given term")
    inv_doc_freq_parser.add_argument("term", type=str, help="Term to search")

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

        case "tf":
            index = InvertedIndex(tokenizer)
            try:
                index.load()
            except IOError as e:
                print(f"Error while loading index files: {e}")
                sys.exit(1)

            print(f"Retrieve term frequency for '{args.term}' in document {args.doc_id}")
            print(f"Count: {index.get_tf(args.doc_id, args.term)}")

        case "idf":
            index = InvertedIndex(tokenizer)
            try:
                index.load()
            except IOError as e:
                print(f"Error while loading index files: {e}")
                sys.exit(1)

            term = tokenizer.tokenize_word(args.term)
            term_doc_count = len(index.get_documents(term))
            idf = math.log((len(index.docmap) + 1) / (term_doc_count + 1))
            print(idf)

            print(f"Inverse document frequency of '{term}': {idf:.2f}")

        case "build":
            index = InvertedIndex(tokenizer)
            index.build(movies["movies"])
            index.save()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
