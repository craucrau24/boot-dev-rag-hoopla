#!/usr/bin/env python3

import argparse
import os
import json
import itertools
import string

from nltk.stem import PorterStemmer

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    with open(os.path.join("data", "movies.json")) as mov:
        movies = json.load(mov)

    with open(os.path.join("data", "stopwords.txt")) as stop:
        stopwords = set(stop.read().splitlines())


    match args.command:
        case "search":
            print(f"Searching for: {args.query}")

            stemmer = PorterStemmer()
            
            def tokenize(s: str):
                table = str.maketrans("", "", string.punctuation)
                new_s = s.lower().translate(table)
                return list(
                       map(lambda tok: stemmer.stem(tok),
                           filter( lambda word: word not in stopwords, new_s.split())
                        ))

            def check(query, title):
                query = tokenize(query)
                title = tokenize(title)

                return any(map(lambda tp: tp[0] in tp[1], itertools.product(query, title)))


            result = itertools.islice(
                filter(lambda mov: check(args.query, mov["title"]), movies["movies"]),
                5
            )

            for i, mov in enumerate(result):
                print(f"{i + 1}. {mov["title"]}")


        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
