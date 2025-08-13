# backend/app/lancedb_store.py

import asyncio
import logging
import shutil
import os
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import pandas as pd
import numpy as np
from lancedb import connect, LanceTable
import pyarrow as pa

from app.config import settings

logger = logging.getLogger(__name__)

class LanceDBStore:
    """
    Enhanced LanceDB wrapper for mathematical knowledge base operations.
    Supports ingestion, semantic search, exact matching, and maintenance.
    """

    def __init__(
        self, 
        db_path: str = None, 
        table_name: str = None, 
        vector_column: str = None
    ):
        self.db_path = (db_path or settings.LANCEDB_PATH).rstrip("/")
        self.table_name = table_name or settings.LANCEDB_TABLE
        self.vector_column = vector_column or settings.LANCEDB_VECTOR_COLUMN
        
        # Ensure db directory exists
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        
        # Connection will be established lazily
        self._conn = None
        self._table = None
        
        logger.info(f"LanceDB Store initialized: {self.db_path}/{self.table_name}")
    
    @property
    def conn(self):
        """Lazy connection to LanceDB."""
        if self._conn is None:
            try:
                self._conn = connect(self.db_path)
                logger.debug(f"Connected to LanceDB at {self.db_path}")
            except Exception as e:
                logger.error(f"❌ Failed to connect to LanceDB: {e}")
                raise
        return self._conn
    
    @property 
    def table(self) -> Optional[LanceTable]:
        """Get the table instance if it exists."""
        if self._table is None:
            try:
                self._table = self.conn.open_table(self.table_name)
                logger.debug(f"Opened table: {self.table_name}")
            except Exception as e:
                logger.warning(f"Table {self.table_name} not accessible: {e}")
                return None
        return self._table
    
    def table_exists(self) -> bool:
        """Check if the table exists."""
        try:
            self.conn.open_table(self.table_name)
            return True
        except Exception:
            return False
    
    def ingest_from_parquet(
        self,
        parquet_path: str,
        overwrite: bool = False,
        index_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Ingest Parquet file into LanceDB and create vector index.
        
        Args:
            parquet_path: Path to the parquet file
            overwrite: Whether to overwrite existing table
            index_params: Parameters for vector index creation
        """
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        
        # Handle table overwrite
        table_dir = Path(self.db_path) / f"{self.table_name}.lance"
        if overwrite and table_dir.exists():
            logger.info(f"🗑️ Removing existing table: {table_dir}")
            shutil.rmtree(table_dir)
            self._table = None  # Reset cached table
        
        try:
            # Load parquet data
            df = pd.read_parquet(parquet_path)
            logger.info(f"📊 Loaded {len(df)} records from {parquet_path}")
            
            # Validate required columns
            required_cols = ['question', 'answer', self.vector_column]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Validate embeddings
            if self.vector_column in df.columns:
                # Check embedding format
                sample_embedding = df[self.vector_column].iloc[0]
                if not isinstance(sample_embedding, (list, np.ndarray)):
                    raise ValueError(f"Invalid embedding format in column {self.vector_column}")
                
                embedding_dim = len(sample_embedding)
                logger.info(f"📐 Embedding dimension: {embedding_dim}")
            
            # Create table
            if overwrite or not self.table_exists():
                logger.info(f"🔨 Creating new table: {self.table_name}")
                table = self.conn.create_table(self.table_name, df, mode="overwrite")
            else:
                logger.info(f"➕ Appending to existing table: {self.table_name}")
                table = self.conn.open_table(self.table_name)
                table.add(df)
            
            # Create vector index
            try:
                default_index_params = {
                    "metric": "cosine",
                    "num_partitions": min(256, len(df) // 10 + 1),
                    "num_sub_vectors": 96
                }
                
                final_index_params = {**default_index_params, **(index_params or {})}
                
                logger.info(f"🔍 Creating vector index with params: {final_index_params}")
                table.create_index(self.vector_column, **final_index_params)
                logger.info("✅ Vector index created successfully")
                
            except Exception as e:
                logger.warning(f"⚠️ Vector index creation failed: {e}")
                logger.info("Table created without vector index")
            
            # Update cached table reference
            self._table = table
            
            # Log success
            final_count = len(table.to_pandas())
            logger.info(f"✅ Successfully ingested {final_count} records into {self.table_name}")
            
        except Exception as e:
            logger.error(f"❌ Ingestion failed: {e}")
            raise
    
    def find_exact_match(
        self, 
        query_text: str, 
        question_field: str = "question"
    ) -> Optional[Dict[str, Any]]:
        """
        Find exact text match in the knowledge base.
        
        Args:
            query_text: Text to search for exactly
            question_field: Column name containing questions
            
        Returns:
            Dict with match details or None
        """
        if not self.table:
            logger.warning("Table not available for exact match search")
            return None
        
        try:
            # Get all data (for small tables this is acceptable)
            df = self.table.to_pandas()
            
            # Normalize query
            query_norm = str(query_text).strip().lower()
            
            # Search for exact matches
            for idx, row in df.iterrows():
                question = str(row.get(question_field, "")).strip().lower()
                if question == query_norm:
                    logger.info(f"🎯 Exact match found for: {query_text}")
                    return {
                        "question": row.get(question_field, ""),
                        "answer": row.get("answer", ""),
                        "steps": row.get("steps"),
                        "score": 1.0,
                        "source": "lancedb_exact_match",
                        "topic": row.get("topic"),
                        "metadata": row.get("metadata", {})
                    }
            
            logger.debug(f"No exact match found for: {query_text}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Exact match search failed: {e}")
            return None
    
    async def semantic_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.7,
        filter_condition: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector similarity.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            filter_condition: Optional SQL-like filter
            
        Returns:
            List of matching documents with metadata
        """
        
        def _search():
            if not self.table:
                logger.warning("Table not available for semantic search")
                return []
            
            try:
                # Validate embedding
                if not query_embedding or len(query_embedding) == 0:
                    logger.error("Empty query embedding provided")
                    return []
                
                # Build search query
                search_query = self.table.search(query_embedding).limit(top_k)
                
                # Apply filter if provided
                if filter_condition:
                    search_query = search_query.where(filter_condition)
                
                # Execute search
                hits = search_query.to_list()
                
                logger.info(f"🔍 Semantic search returned {len(hits)} raw hits")
                
                if not hits:
                    return []
                
                # Process results
                results = []
                for hit in hits:
                    # Extract similarity score
                    similarity_score = self._extract_similarity_score(hit)
                    
                    # Apply threshold filter
                    if similarity_score >= threshold:
                        result = {
                            "question": hit.get("question", ""),
                            "answer": hit.get("answer", ""),
                            "steps": hit.get("steps"),
                            "score": similarity_score,
                            "source": "lancedb_semantic",
                            "topic": hit.get("topic"),
                            "metadata": hit.get("metadata", {}),
                            "distance": hit.get("_distance"),
                            "original_hit": hit  # For debugging
                        }
                        results.append(result)
                        
                        logger.debug(f"✅ Hit above threshold: {similarity_score:.3f} - {hit.get('question', '')[:50]}...")
                    else:
                        logger.debug(f"❌ Hit below threshold: {similarity_score:.3f}")
                
                logger.info(f"📊 Returning {len(results)} hits above threshold {threshold}")
                return results
                
            except Exception as e:
                logger.error(f"❌ Semantic search failed: {e}")
                return []
        
        # Run search in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _search)
    
    def _extract_similarity_score(self, hit: Dict[str, Any]) -> float:
        """
        Extract similarity score from LanceDB hit.
        
        Args:
            hit: Search result from LanceDB
            
        Returns:
            Similarity score between 0 and 1
        """
        # Try different score fields LanceDB might return
        score_fields = ["_score", "_vector_score", "score"]
        
        for field in score_fields:
            if field in hit:
                try:
                    return float(hit[field])
                except (ValueError, TypeError):
                    continue
        
        # Fallback: convert distance to similarity
        distance = hit.get("_distance")
        if distance is not None:
            try:
                # Convert L2 distance to similarity (assuming normalized embeddings)
                # For cosine distance: similarity = 1 - distance
                similarity = max(0.0, min(1.0, 1.0 - float(distance)))
                return similarity
            except (ValueError, TypeError):
                pass
        
        # Default fallback
        logger.warning(f"Could not extract similarity score from hit: {list(hit.keys())}")
        return 0.0
    
    def get_table_info(self) -> Dict[str, Any]:
        """Get information about the table."""
        if not self.table:
            return {"error": "Table not available"}
        
        try:
            df = self.table.to_pandas()
            
            info = {
                "table_name": self.table_name,
                "total_records": len(df),
                "columns": list(df.columns),
                "vector_column": self.vector_column,
                "has_vector_index": True,  # Assume true if table exists
                "sample_questions": df["question"].head(3).tolist() if "question" in df.columns else []
            }
            
            # Check embedding dimension
            if self.vector_column in df.columns and len(df) > 0:
                sample_embedding = df[self.vector_column].iloc[0]
                if isinstance(sample_embedding, (list, np.ndarray)):
                    info["embedding_dimension"] = len(sample_embedding)
            
            return info
            
        except Exception as e:
            logger.error(f"❌ Failed to get table info: {e}")
            return {"error": str(e)}
    
    def debug_table_contents(self, limit: int = 5) -> None:
        """
        Debug method to inspect table contents.
        
        Args:
            limit: Number of records to display
        """
        if not self.table:
            logger.error("❌ Table not available for debugging")
            return
        
        try:
            df = self.table.to_pandas()
            total_rows = len(df)
            
            logger.info(f"📊 Table '{self.table_name}' contains {total_rows} total rows")
            logger.info(f"📋 Columns: {list(df.columns)}")
            
            # Show sample records
            sample_df = df.head(limit)
            logger.info(f"🔍 Sample records (first {limit}):")
            
            for i, row in sample_df.iterrows():
                question = str(row.get("question", ""))[:100]
                answer = str(row.get("answer", ""))[:50]
                logger.info(f"  Row {i}: Q='{question}...' A='{answer}...'")
            
            # Check embeddings
            if self.vector_column in df.columns:
                sample_embedding = df[self.vector_column].iloc[0]
                embedding_type = type(sample_embedding).__name__
                embedding_len = len(sample_embedding) if hasattr(sample_embedding, '__len__') else 'unknown'
                logger.info(f"🧮 Embedding info: type={embedding_type}, length={embedding_len}")
            
        except Exception as e:
            logger.error(f"❌ Debug inspection failed: {e}")
    
    def optimize_table(self) -> None:
        """Optimize table performance."""
        if not self.table:
            logger.warning("No table to optimize")
            return
        
        try:
            logger.info("🔧 Optimizing table...")
            self.table.optimize()
            logger.info("✅ Table optimization completed")
        except Exception as e:
            logger.error(f"❌ Table optimization failed: {e}")
    
    def delete_table(self) -> None:
        """Delete the entire table."""
        try:
            table_dir = Path(self.db_path) / f"{self.table_name}.lance"
            if table_dir.exists():
                shutil.rmtree(table_dir)
                logger.info(f"🗑️ Deleted table: {self.table_name}")
                self._table = None
            else:
                logger.warning(f"Table directory not found: {table_dir}")
        except Exception as e:
            logger.error(f"❌ Table deletion failed: {e}")

# Convenience functions for backward compatibility
def create_lancedb_store() -> LanceDBStore:
    """Create a LanceDB store with default settings."""
    return LanceDBStore()

async def search_knowledge_base(query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """Search the knowledge base with default settings."""
    store = create_lancedb_store()
    return await store.semantic_search(query_embedding, top_k=top_k)