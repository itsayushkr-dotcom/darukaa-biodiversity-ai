"""
Vector Knowledge Store for Darukaa.Earth Biodiversity Intelligence System.
Provides hybrid semantic and lexical retrieval over indexed scientific datasets (FAO, IPCC, IPBES, ICAR).
"""

import os
import json
import re
import math
from typing import List, Dict, Any, Optional
from collections import Counter


class EnvironmentalVectorStore:
    """
    In-memory vector store with cosine semantic-lexical hybrid search,
    supporting domain filtering, variable cross-matching, and citation traceability.
    """

    def __init__(self, raw_docs_dir: Optional[str] = None):
        if raw_docs_dir is None:
            raw_docs_dir = os.path.join(os.path.dirname(__file__), "raw_documents")
        self.raw_docs_dir = raw_docs_dir
        self.documents: List[Dict[str, Any]] = []
        self.vocabulary: Dict[str, int] = {}
        self.doc_vectors: List[List[float]] = []
        self.idf: Dict[str, float] = {}
        self._load_and_index()

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase and extract alphanumeric tokens + scientific terms
        tokens = re.findall(r'[a-zA-Z0-9_\-\.%]+', text.lower())
        # Filter single characters unless numbers or specific symbols
        return [t for t in tokens if len(t) > 1 or t.isdigit()]

    def _build_doc_text(self, doc: Dict[str, Any]) -> str:
        parts = [
            doc.get("title", ""),
            doc.get("source", ""),
            doc.get("domain", ""),
            " ".join(doc.get("environmental_variables", [])),
            doc.get("summary", ""),
            doc.get("scientific_mechanism", ""),
            " ".join(doc.get("recommended_interventions", [])),
            " ".join(doc.get("target_biomes", [])),
        ]
        # Include quantified impact keys and values
        impact = doc.get("quantified_impact", {})
        if isinstance(impact, dict):
            for k, v in impact.items():
                parts.append(f"{k} {v}")
        return " ".join(parts)

    def _load_and_index(self):
        """Loads JSON document files from raw_documents directory and builds inverted index & TF-IDF vectors."""
        self.documents = []
        if not os.path.exists(self.raw_docs_dir):
            return

        for fname in os.listdir(self.raw_docs_dir):
            if fname.endswith(".json") and fname != "agro_ecological_zones.json":
                fpath = os.path.join(self.raw_docs_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            self.documents.extend(data)
                        elif isinstance(data, dict):
                            self.documents.append(data)
                except Exception as e:
                    print(f"Error loading {fpath}: {e}")

        # Compute document frequencies
        doc_count = len(self.documents)
        if doc_count == 0:
            return

        doc_tokens_list = []
        df_counter: Counter = Counter()

        for doc in self.documents:
            text = self._build_doc_text(doc)
            tokens = self._tokenize(text)
            doc_tokens_list.append(tokens)
            unique_tokens = set(tokens)
            for t in unique_tokens:
                df_counter[t] += 1

        # Build vocabulary (top tokens)
        vocab_tokens = [t for t, count in df_counter.items() if count >= 1]
        self.vocabulary = {t: idx for idx, t in enumerate(vocab_tokens)}

        # Compute IDF
        for t, df in df_counter.items():
            self.idf[t] = math.log((doc_count + 1) / (df + 1)) + 1.0

        # Compute L2-normalized TF-IDF vector for each document
        self.doc_vectors = []
        for tokens in doc_tokens_list:
            vec = self._vectorize_tokens(tokens)
            self.doc_vectors.append(vec)

    def _vectorize_tokens(self, tokens: List[str]) -> List[float]:
        tf = Counter(tokens)
        total = len(tokens) or 1
        vec = [0.0] * len(self.vocabulary)
        sum_sq = 0.0

        for t, count in tf.items():
            if t in self.vocabulary:
                idx = self.vocabulary[t]
                weight = (count / total) * self.idf.get(t, 1.0)
                vec[idx] = weight
                sum_sq += weight * weight

        # Normalize
        norm = math.sqrt(sum_sq)
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        return sum(a * b for a, b in zip(v1, v2))

    def search(
        self,
        query: str,
        top_k: int = 3,
        domain_filter: Optional[str] = None,
        environmental_variables: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge base using hybrid semantic-lexical matching.
        Returns top-k matching scientific records with similarity score and citations.
        """
        if not self.documents:
            return []

        q_tokens = self._tokenize(query)
        q_vec = self._vectorize_tokens(q_tokens)

        results = []
        for idx, doc in enumerate(self.documents):
            # Domain filter check
            if domain_filter and doc.get("domain") != domain_filter:
                continue

            sim = self._cosine_similarity(q_vec, self.doc_vectors[idx])

            # Bonus score if matching target environmental variables
            var_match_bonus = 0.0
            if environmental_variables:
                doc_vars = set(doc.get("environmental_variables", []))
                matches = doc_vars.intersection(set(environmental_variables))
                var_match_bonus = len(matches) * 0.15

            total_score = min(1.0, round(sim + var_match_bonus, 4))

            doc_id = doc.get("id", "")
            doc_urls = {
                "FAO_SOC_001": "https://www.fao.org/global-soil-partnership/recsoil/en/",
                "FAO_SOC_002": "https://www.fao.org/conservation-agriculture/en/",
                "IPCC_SRCCL_001": "https://www.ipcc.ch/srccl/chapter/chapter-4/",
                "IPCC_SRCCL_002": "https://www.unccd.int/resources/publications/scientific-conceptual-framework-land-degradation-neutrality",
                "IPBES_BIO_001": "https://www.ipbes.net/assessment-reports/pollinators",
                "IPBES_BIO_002": "https://www.nature.com/articles/s41559-019-0885-4",
                "SOIL_PHYS_001": "https://www.fao.org/global-soil-partnership/areas-of-work/soil-salinity/en/",
                "SOIL_PHYS_002": "https://www.wocat.net/en/global-slm-database/",
                "UNEP_POLLUTION_001": "https://www.unep.org/resources/report/global-assessment-soil-pollution",
                "FAO_FOREST_002": "https://www.fao.org/state-of-forests/en/"
            }
            resolved_url = doc.get("url") or doc_urls.get(doc_id, "https://www.fao.org/global-soil-partnership/en/")

            results.append({
                "document_id": doc_id,
                "title": doc.get("title"),
                "source": doc.get("source"),
                "publication_year": doc.get("publication_year"),
                "domain": doc.get("domain"),
                "environmental_variables": doc.get("environmental_variables", []),
                "summary": doc.get("summary"),
                "scientific_mechanism": doc.get("scientific_mechanism"),
                "quantified_impact": doc.get("quantified_impact", {}),
                "recommended_interventions": doc.get("recommended_interventions", []),
                "relevance_score": total_score,
                "url": resolved_url,
            })

        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:top_k]

    def get_all_documents(self) -> List[Dict[str, Any]]:
        return self.documents


# Module test instance
if __name__ == "__main__":
    store = EnvironmentalVectorStore()
    print(f"Loaded {len(store.documents)} scientific documents across {len(store.vocabulary)} vocabulary terms.")
    sample_query = "low soil organic carbon wheat monoculture semi-arid low rainfall"
    res = store.search(sample_query, top_k=2)
    for r in res:
        print(f"\nMatch: {r['title']} (Score: {r['relevance_score']})")
        print(f"Source: {r['source']} ({r['publication_year']})")
        print(f"Quantified Impact: {r['quantified_impact']}")
