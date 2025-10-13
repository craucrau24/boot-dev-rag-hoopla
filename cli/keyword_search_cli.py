#!/usr/bin/env python3

import argparse
import os
import json
import itertools


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    with open(os.path.join("data", "movies.json")) as mov:
        movies = json.load(mov)

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
        
            result = itertools.islice(
                filter(lambda mov: args.query in mov["title"], movies["movies"]),
                5
            )

            for i, mov in enumerate(result):
                print(f"{i + 1}. {mov["title"]}")


        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
