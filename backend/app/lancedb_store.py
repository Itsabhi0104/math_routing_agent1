# backend/app/lancedb_store.py
import asyncio
import logging
import shutil
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from lancedb import connect

logger = logging.getLogger(__name__)

class LanceDBStore:
    """
    LanceDB wrapper for ingesting a Parquet KB and performing semantic search.
    """

    def __init__(self, db_path: str = "lancedb_math", table_name: str = "math_qa", vector_column: str = "embedding"):
        self.db_path = db_path.rstrip("/")
        self.table_name = table_name
        self.vector_column = vector_column
        # connect will create the folder if needed
        self.conn = connect(self.db_path)

    def ingest_from_parquet(
        self,
        parquet_path: str,
        overwrite: bool = False,
        index_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Ingest Parquet file into LanceDB and create a vector index.
        """
        table_dir = os.path.join(self.db_path, f"{self.table_name}.lance")
        if overwrite and os.path.exists(table_dir):
            logger.info("Dropped existing LanceDB folder: %s", table_dir)
            shutil.rmtree(table_dir)

        df = pd.read_parquet(parquet_path)
        logger.info("Ingesting %s → table '%s' (overwrite=%s)", parquet_path, self.table_name, overwrite)
        tbl = self.conn.create_table(self.table_name, df)
        # Optionally create an index on the vector column
        try:
            tbl.create_index(self.vector_column, index_params or {})
            logger.info("Created index on column %s", self.vector_column)
        except Exception as e:
            logger.warning("Could not create vector index: %s", e)

    def find_exact_match(self, query_text: str, question_field: str = "question") -> Optional[Dict[str, Any]]:
        """
        Lightweight exact-match lookup of question text in the table. Returns the first matching row dict.
        NOTE: For large KBs this is O(N). For production, create a normalized text index.
        """
        try:
            tbl = self.conn.open_table(self.table_name)
            df = tbl.to_pandas()
            q_norm = str(query_text).strip().lower()
            for _, row in df.iterrows():
                if str(row.get(question_field, "")).strip().lower() == q_norm:
                    return {
                        "answer": row.get("answer", ""),
                        "question": row.get(question_field, ""),
                        "score": 1.0,
                        "source": "lance_db_exact",
                    }
        except Exception as e:
            logger.exception("Exact match lookup failed: %s", e)
        return None

    async def semantic_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search and return hits whose similarity score >= threshold.
        This method prefers friendly score fields returned by LanceDB and falls back to a guarded
        distance->similarity conversion.
        """

        def _search():
            try:
                tbl = self.conn.open_table(self.table_name)
                # search API expects embedding; returns list of hit dicts
                hits = tbl.search(query_embedding).limit(top_k).to_list()

                logger.info("Raw search returned %d hits", len(hits))
                if hits:
                    logger.info("Sample hit keys: %s", [list(r.keys()) for r in hits[:2]])

                results: List[Dict[str, Any]] = []
                for r in hits:
                    # Prefer an explicit vector score if provided
                    score = None
                    if "_vector_score" in r:
                        score = r.get("_vector_score")
                    elif "_score" in r:
                        score = r.get("_score")

                    if score is None:
                        distance = r.get("_distance", None)
                        if distance is None:
                            similarity_score = 0.0
                        else:
                            try:
                                similarity_score = max(0.0, min(1.0, 1.0 - (float(distance) / 2.0)))
                            except Exception:
                                similarity_score = 0.0
                    else:
                        try:
                            similarity_score = float(score)
                        except Exception:
                            similarity_score = 0.0

                    # Read common metadata fields
                    answer = r.get("answer") or r.get("solution") or r.get("text") or ""
                    steps = r.get("steps")
                    question = r.get("question") or r.get("problem") or None

                    if similarity_score >= threshold:
                        results.append({
                            "answer": answer,
                            "question": question,
                            "steps": steps,
                            "score": similarity_score,
                            "source": r.get("_source", "lance_db"),
                        })
                return results
            except Exception as e:
                logger.exception("LanceDB semantic search failed: %s", e)
                return []

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _search)

    def debug_table_contents(self, limit: int = 5):
        """
        Debug method to inspect table contents.
        """
        try:
            tbl = self.conn.open_table(self.table_name)
            df = tbl.to_pandas().head(limit)
            logger.info("Table has %d total rows", len(tbl.to_pandas()))
            logger.info("Sample rows:")
            for i, row in df.iterrows():
                logger.info("  Row %d: question=%s answer=%s", i, row.get("question"), row.get("answer"))
        except Exception as e:
            logger.exception("Debug read failed: %s", e)
