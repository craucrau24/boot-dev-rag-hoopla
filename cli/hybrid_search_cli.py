import argparse
import os
import json
import re

from dotenv import load_dotenv
from google import genai

from data.utils import normalize,Tokenizer
from data.hybrid_search import HybridSearch

def get_spell_corrected_query(query: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    print(f"Using key {api_key[:6]}...")


    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model='gemini-2.0-flash-001', contents=f"""Fix any spelling errors in this movie search query.

        Only correct obvious typos. Don't change correctly spelled words.

        Query: "{query}"

        If no errors, return the original query.
        Corrected:"""
        )

    m = re.match(r'Corrected:\s*"(.*?)"', response.text)
    if m is not None:
        return m.group(1)
    else:
        return query

def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="normalize inputs")
    normalize_parser.add_argument("values", nargs="+", type=float, help="numeric values to be normalized")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="search using both keyword and semantic search (alpha parameter can be used to modulate search)")
    weighted_search_parser.add_argument("query", type=str, help="The search query")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="the relative weight between semantic (0.0) and keyword (1.0) search")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="maximum number of results")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="search using both keyword and semantic search with reciprocal rank fusion algorithm (k parameter can be used to modulate search)")
    rrf_search_parser.add_argument("query", type=str, help="The search query")
    rrf_search_parser.add_argument("--k", type=int, default=60, help="the k parameter used to define the steepness of the curve")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="maximum number of results")
    rrf_search_parser.add_argument( "--enhance", type=str, choices=["spell"], help="Query enhancement method")

    args = parser.parse_args()

    tokenizer = Tokenizer()
    tokenizer.load_stop_words(os.path.join("data", "stopwords.txt"))

    match args.command:
        case "normalize":
            for v in normalize(args.values):
                print(f"* {v:.4f}")

        case "weighted-search":
            with open(os.path.join("data", "movies.json")) as f:
                movies = json.load(f)
            hyb = HybridSearch(movies["movies"], tokenizer)
            results = hyb.weighted_search(args.query, args.alpha, args.limit)
            for i, r in enumerate(results):
                print(f"{i + 1}. {r["title"]}\nHybrid score: {r["hybrid_score"]: .4f}\nBM25: {r["bm25_score"]: .4f} Semantic: {r["semantic_score"]: .4f}\n{r["document"]}")

        case "rrf-search":
            with open(os.path.join("data", "movies.json")) as f:
                movies = json.load(f)
            hyb = HybridSearch(movies["movies"], tokenizer)
            query = args.query
            if args.enhance == "spell":
                new_query = get_spell_corrected_query(query)
                if query != new_query:
                    print( f"Enhanced query ({args.enhance}): '{query}' -> '{new_query}'\n")
                    query = new_query

            results = hyb.rrf_search(query, args.k, args.limit)
            for i, r in enumerate(results):
                bm25_rank = r['bm25_rank'] if r['bm25_rank'] is not None else "N/A"
                semantic_rank = r['semantic_rank'] if r['semantic_rank'] is not None else "N/A"
                print(f"{i + 1}. {r["title"]}\nRRF score: {r["rrf_score"]: .4f}\nBM25 Rank: {bm25_rank} Semantic Rank: {semantic_rank}\n{r["document"]}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()