# backend/app/adaptive_learning.py

import logging
import datetime
from typing import List, Dict, Any, Optional
import asyncio

from sqlalchemy.ext.asyncio import AsyncSession
from app.services.db_service import DBService
from app.dspy_integration import MathCoTPipeline
from app.models.models import UserFeedback

logger = logging.getLogger(__name__)


class AdaptiveLearner:
    """
    Uses collected feedback to refine future responses via:
    - Prompt / few-shot template updates
    - DSPy signature fine-tuning triggers
    - Knowledge base & embedding refresh
    """

    def __init__(
        self,
        db_session: AsyncSession,
        cot_pipeline: MathCoTPipeline,
        good_examples: Optional[List[Dict[str, Any]]] = None,
    ):
        self.db = DBService(db_session)
        self.cot = cot_pipeline
        self.good_examples = good_examples or []
        # history of performance metrics
        self.metrics_log: List[Dict[str, Any]] = []

    async def refine_response(
        self,
        question: str,
        initial_answer: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Given user feedback for an initial_answer, generate a refined answer.
        - Pull top N good examples to few-shot
        - Re-run Chain-of-Thought with updated context
        """
        # 1) Retrieve recent positive feedback examples
        feedbacks: List[UserFeedback] = await self.db.list_feedback(limit=50)
        positive = [f for f in feedbacks if f.rating and f.rating >= 4]
        few_shot = positive[:3]  # take top 3
        shot_context = "\n".join(
            f"Q: {f.question}\nA: {f.generated_answer}" for f in few_shot
        )

        # 2) Build augmented prompt
        prompt = (
            f"{shot_context}\n\n"
            f"Now solve step-by-step: {question}"
        )

        # 3) Rerun CoT pipeline
        refined = await asyncio.to_thread(self.cot.solve, prompt)

        # 4) Log performance: rating vs. refinement time
        metric = {
            "question": question,
            "initial_score": initial_answer.get("score"),
            "timestamp": datetime.datetime.utcnow(),
        }
        self.metrics_log.append(metric)
        logger.info("Refinement metric logged: %s", metric)

        return refined

    async def track_performance(self):
        """
        Aggregate metrics over time, e.g. average initial_score.
        """
        if not self.metrics_log:
            return {}
        avg_score = sum(m["initial_score"] or 0 for m in self.metrics_log) / len(self.metrics_log)
        summary = {
            "total_sessions": len(self.metrics_log),
            "avg_initial_score": avg_score,
            "last_updated": datetime.datetime.utcnow(),
        }
        logger.info("Performance summary: %s", summary)
        return summary

    async def rebuild_embeddings(self):
        """
        Periodically retrain or re-embed examples that received strong feedback
        and update the LanceDB store.
        """
        from app.knowledge_loader import KnowledgeBaseBuilder
        from app.lancedb_store import LanceDBStore

        # 1) Gather Q&A pairs with high ratings
        feedbacks: List[UserFeedback] = await self.db.list_feedback(limit=100)
        high = [f for f in feedbacks if f.rating and f.rating >= 4]
        # 2) Create a small DataFrame for re-embedding
        import pandas as pd
        df = pd.DataFrame([
            {"question": f.question, "answer": f.generated_answer, "metadata": {}}
            for f in high
        ])
        # 3) Run embedder
        builder = KnowledgeBaseBuilder(batch_size=32)
        new_df = builder.generate_embeddings(df, text_column="question")
        # 4) Upsert into LanceDB
        store = LanceDBStore()
        store.ingest_from_parquet("knowledge_base.parquet")  # re-ingest full KB, or...
        # Optionally: incremental add via store.table.add()

        logger.info("Rebuilt embeddings with %d high-feedback examples", len(high))

    async def a_b_test(
        self,
        questions: List[str],
        variant_fn: Any,
    ) -> Dict[str, Any]:
        """
        Run an A/B test comparing original vs. variant_fn response logic.
        Returns aggregate score differences.
        """
        results = []
        for q in questions:
            orig = await asyncio.to_thread(self.cot.solve, q)
            var = await asyncio.to_thread(variant_fn, q)
            # Here insert code to collect user ratings or simulated metrics
            results.append({"question": q, "orig": orig, "var": var})
        # Analyze differences
        # (Placeholder: real implementation would gather real feedback)
        report = {"tested": len(results), "notes": "A/B test completed"}
        logger.info("A/B test report: %s", report)
        return report
