import logging
import re
from typing import List, Optional
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

def clean_math_expression(expr: str) -> str:
    """
    Normalize whitespace and LaTeX in a math expression.
    """
    try:
        cleaned = re.sub(r'\s+', ' ', expr).strip()
        cleaned = cleaned.replace('\\\\', '\\')
        return cleaned
    except Exception:
        logger.exception("Failed to clean expression: %s", expr)
        return expr

class KnowledgeBaseBuilder:
    """
    Loads and preprocesses multiple math QA datasets, then computes embeddings.
    """

    def __init__(self, embed_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', batch_size: int = 64):
        self.batch_size = batch_size
        try:
            self.embedder = SentenceTransformer(embed_model_name)
            logger.info("Loaded embedding model %s", embed_model_name)
        except Exception:
            logger.exception("Failed to load embedding model %s", embed_model_name)
            raise

    def load_gsm8k(self) -> pd.DataFrame:
        """
        Load GSM8K dataset and return DataFrame with columns:
        question, answer, source, metadata
        """
        ds = load_dataset("gsm8k", "main")
        df = pd.DataFrame(ds["train"][:])
        df = df.rename(columns={"question": "question", "answer": "answer"})
        df["source"] = "gsm8k"
        df["metadata"] = df.apply(lambda row: {}, axis=1)
        logger.info("Loaded GSM8K: %d examples", len(df))
        return df[["question", "answer", "source", "metadata"]]

    def load_hendrycks(self) -> pd.DataFrame:
        """
        Load all configs from EleutherAI/hendrycks_math and return unified DataFrame.
        """
        configs = [
            'algebra', 'geometry', 'number_theory', 'prealgebra',
            'precalculus', 'counting_and_probability', 'intermediate_algebra'
        ]
        dfs = []
        for cfg in configs:
            try:
                logger.info(f"Loading Hendrycks config: {cfg}")
                ds = load_dataset("EleutherAI/hendrycks_math", name=cfg, split="train")
                df = pd.DataFrame(ds[:])
                df = df.rename(columns={"problem": "question", "solution": "answer"})
                df["source"] = "hendrycks_math"
                df["metadata"] = df.apply(lambda row: {"level": cfg}, axis=1)
                dfs.append(df[["question", "answer", "source", "metadata"]])
                logger.info("Loaded %s: %d examples", cfg, len(df))
            except Exception:
                logger.exception(f"Failed to load Hendrycks config: {cfg}")
        full_df = pd.concat(dfs, ignore_index=True)
        return full_df

    def unify_datasets(self) -> pd.DataFrame:
        """
        Combine GSM8K and Hendrycks Math into one DataFrame,
        cleaning math expressions and standardizing columns.
        """
        try:
            df1 = self.load_gsm8k()
            df2 = self.load_hendrycks()
            df = pd.concat([df1, df2], ignore_index=True)
            df["question"] = df["question"].apply(clean_math_expression)
            df["answer"] = df["answer"].apply(clean_math_expression)
            logger.info("Unified dataset size: %d", len(df))
            return df
        except Exception:
            logger.exception("Failed to unify datasets")
            raise

    def generate_embeddings(self, df: pd.DataFrame, text_column: str = "question") -> pd.DataFrame:
        """
        Compute embeddings for each row in `text_column`, in batches.
        Appends a column 'embedding' with List[float].
        """
        embeddings: List[List[float]] = []
        texts = df[text_column].tolist()
        total = len(texts)
        logger.info("Generating embeddings for %d items", total)

        try:
            for start in range(0, total, self.batch_size):
                end = min(start + self.batch_size, total)
                batch = texts[start:end]
                emb_batch = self.embedder.encode(batch, convert_to_numpy=True)
                embeddings.extend(emb_batch.tolist())
                logger.debug("Processed embeddings %d–%d", start, end)
        except Exception:
            logger.exception("Embedding generation failed at batch starting %d", start)
            raise

        df = df.copy()
        df["embedding"] = embeddings
        return df

    def build(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Full pipeline: unify, embed, and optionally save to disk.
        """
        df = self.unify_datasets()
        df = self.generate_embeddings(df)
        if output_path:
            try:
                df.to_parquet(output_path, index=False)
                logger.info("Saved knowledge base to %s", output_path)
            except Exception:
                logger.exception("Failed to save knowledge base to %s", output_path)
        return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    builder = KnowledgeBaseBuilder(batch_size=128)
    kb_df = builder.build(output_path="knowledge_base.parquet")
    print(f"Built knowledge base with {len(kb_df)} rows")
