import argparse
import os
import json
import re
import time

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

def get_rewritten_query(query: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    print(f"Using key {api_key[:6]}...")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model='gemini-2.0-flash-001', contents=f"""Rewrite this movie search query to be more specific and searchable. Enclose result in <query> XML tags.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

<query>"""        )

    print(response.text)
    m = re.search(r'"?(.*?)"?\s*</query>', response.text)
    if m is not None:
        return m.group(1)
    else:
        return query

def get_expand_query(query: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    print(f"Using key {api_key[:6]}...")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model='gemini-2.0-flash-001', contents=f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"
 
Enclose result in <query> XML tags.
Query: "{query}"
<query>""")

    print(response.text)
    m = re.search(r'"?(.*?)"?\s*</query>', response.text)
    if m is not None:
        return m.group(1)
    else:
        return query

def rerank_individual_results(results, query):
    api_key = os.environ.get("GEMINI_API_KEY")
    print(f"Using key {api_key[:6]}...")

    client = genai.Client(api_key=api_key)
    for doc in results:
        response = client.models.generate_content(
            model='gemini-2.0-flash-001', contents=f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:""")
        doc["rerank-score"] = float(response.text)
        print(".", end="")
        time.sleep(3)

def rerank_batch_results(results, query, hyb):
    api_key = os.environ.get("GEMINI_API_KEY")
    print(f"Using key {api_key[:6]}...")

    doc_list_str = [hyb.get_doc_by_id(r["id"]) for r in results]

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
            model='gemini-2.0-flash-001', contents=f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
""")
    rank_str = "".join(filter(lambda c: c in set("0123456789,[] "), response.text))
    rank_list = json.loads(rank_str)

    ranks = {_id: r + 1 for r, _id in enumerate(rank_list)}
    for r in results:
        r["rerank-score"] = ranks[r["id"]]



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
    rrf_search_parser.add_argument( "--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_search_parser.add_argument( "--rerank-method", type=str, choices=["individual", "batch"], help="Query enhancement method")

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
            limit = args.limit
            if args.enhance == "spell":
                new_query = get_spell_corrected_query(query)
                if query != new_query:
                    print( f"Enhanced query ({args.enhance}): '{query}' -> '{new_query}'\n")
                    query = new_query
            elif args.enhance == "rewrite":
                new_query = get_rewritten_query(query)
                if query != new_query:
                    print( f"Enhanced query ({args.enhance}): '{query}' -> '{new_query}'\n")
                    query = new_query
            elif args.enhance == "expand":
                new_query = get_expand_query(query)
                if query != new_query:
                    print( f"Enhanced query ({args.enhance}): '{query}' -> '{new_query}'\n")
                    query = new_query
                    
            if args.rerank_method in ["individual", "batch"]:
                print("query:", args.query)
                limit *= 5
                print("limit:", args.limit)
                print("new limit", limit)

            results = hyb.rrf_search(query, args.k, limit)
            if args.rerank_method == "individual":
                print(f"Reranking top {args.limit} results using individual method...")
                rerank_individual_results(results, query)
                results.sort(key=lambda r: r["rerank-score"], reverse=True)
            elif args.rerank_method == "batch":
                print(f"Reranking top {args.limit} results using batch method...")
                rerank_batch_results(results, query, hyb)
                results.sort(key=lambda r: r["rerank-score"], reverse=False)

            print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):")
            for i, r in enumerate(results[:args.limit]):
                bm25_rank = r['bm25_rank'] if r['bm25_rank'] is not None else "N/A"
                semantic_rank = r['semantic_rank'] if r['semantic_rank'] is not None else "N/A"
                rerank_score = r.get("rerank-score")
                rerank_str = f"Rerank Score: {rerank_score:.3f}\n" if rerank_score is not None else ""
                print(f"{i + 1}. {r["title"]}\n{rerank_str}RRF score: {r["rrf_score"]: .4f}\nBM25 Rank: {bm25_rank} Semantic Rank: {semantic_rank}\n{r["document"]}")

            

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()