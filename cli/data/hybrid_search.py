import os
from heapq import nlargest

from .inverted_index import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch
from .utils import normalize_dicts, hybrid_score, rrf_score


class HybridSearch:
    def __init__(self, documents, tokenizer):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex(tokenizer)
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        scores = {}
        keywords = self._bm25_search(query, limit * 500)
        print(keywords[0])
        keywords = normalize_dicts(keywords, "score")
        semantic = normalize_dicts(self.semantic_search.search_chunks(query, limit * 500), "score")
        for k in keywords:
            scores.setdefault(k["id"], {})["keyword"] = k["score"]
        for s in semantic:
            scores.setdefault(s["id"], {})["semantic"] = s["score"]
        
        for sc in scores.values():
            sc["hybrid"] = hybrid_score(sc.get("keyword", 0.0), sc.get("semantic", 0.0), alpha)

        def format_elt(elt):
            k, v = elt
            movie = self.semantic_search.document_map[k]
            return {
                "id": movie["id"],
                "title": movie["title"],
                "document": movie["description"][:100],
                "hybrid_score": v["hybrid"],
                "semantic_score": v["semantic"],
                "bm25_score": v["keyword"],
            }

        result = list(map(
        format_elt,
        nlargest(limit, scores.items(), key=lambda elt: elt[1]["hybrid"])
        ))
        return result

    def rrf_search(self, query, k, limit=10):
        scores = {}
        keywords = self._bm25_search(query, limit * 500)
        semantic = self.semantic_search.search_chunks(query, limit * 500)

        for i, kw in enumerate(keywords):
            sc = scores.setdefault(kw["id"], {"rrf": 0})
            rank = i + 1
            sc["bm25"] = rank
            sc["rrf"] += rrf_score(rank, k)

        for i, sem in enumerate(semantic):
            sc = scores.setdefault(sem["id"], {"rrf": 0})
            rank = i + 1
            sc["semantic"] = rank
            sc["rrf"] += rrf_score(rank, k)

        def format_elt(elt):
            k, v = elt
            movie = self.semantic_search.document_map[k]
            return {
                "id": movie["id"],
                "title": movie["title"],
                "document": movie["description"][:100],
                "rrf_score": v["rrf"],
                "semantic_rank": v.get("semantic"),
                "bm25_rank": v.get("bm25"),
            }
        

        result = list(map(
        format_elt,
        nlargest(limit, scores.items(), key=lambda elt: elt[1]["rrf"])
        ))
        return result
