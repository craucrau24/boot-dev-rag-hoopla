#!/usr/bin/env python3

import argparse
import os
import json
import itertools
import string

from nltk.stem import PorterStemmer

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
            print(f"Searching for: {args.query}")

            stemmer = PorterStemmer()
            
            def check(query, title):
                query = tokenizer.tokenize_str(query)
                title = tokenizer.tokenize_str(title)

                return any(map(lambda tp: tp[0] in tp[1], itertools.product(query, title)))


            result = itertools.islice(
                filter(lambda mov: check(args.query, mov["title"]), movies["movies"]),
                5
            )

            for i, mov in enumerate(result):
                print(f"{i + 1}. {mov["title"]}")

        case "build":
            index = InvertedIndex(tokenizer)
            index.build(movies["movies"])
            index.save()

            docs = index.get_documents("merida")
            print(f"First document for token 'merida' = {docs[0]}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
