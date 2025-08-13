#!/usr/bin/env python3
"""
Knowledge Base Setup Script
Creates and populates the LanceDB knowledge base with GSM8K and Hendrycks Math datasets.
"""

import logging
import os
import sys
import asyncio
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.knowledge_loader import KnowledgeBaseBuilder
from app.lancedb_store import LanceDBStore
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_knowledge_base():
    """
    Complete knowledge base setup pipeline:
    1. Load GSM8K and Hendrycks Math datasets
    2. Generate embeddings
    3. Create LanceDB store
    4. Populate with data
    """
    try:
        logger.info("🚀 Starting Knowledge Base Setup...")
        
        # Step 1: Initialize Knowledge Builder
        logger.info("📚 Initializing Knowledge Base Builder...")
        builder = KnowledgeBaseBuilder(
            embed_model_name='sentence-transformers/all-MiniLM-L6-v2',
            batch_size=64
        )
        
        # Step 2: Build unified knowledge base
        logger.info("🔄 Building unified knowledge base from datasets...")
        kb_df = builder.build(output_path="knowledge_base.parquet")
        logger.info(f"✅ Knowledge base built with {len(kb_df)} examples")
        
        # Step 3: Setup LanceDB
        logger.info("🗄️  Setting up LanceDB store...")
        store = LanceDBStore(
            db_path=settings.LANCEDB_PATH,
            table_name="math_qa",
            vector_column="embedding"
        )
        
        # Step 4: Ingest data into LanceDB
        logger.info("📥 Ingesting data into LanceDB...")
        store.ingest_from_parquet(
            parquet_path="knowledge_base.parquet",
            overwrite=True,
            index_params={"metric": "cosine", "num_partitions": 256}
        )
        
        # Step 5: Test the setup
        logger.info("🧪 Testing knowledge base setup...")
        await test_knowledge_base(store)
        
        logger.info("✅ Knowledge Base Setup Complete!")
        
    except Exception as e:
        logger.error(f"❌ Knowledge Base Setup Failed: {e}")
        raise

async def test_knowledge_base(store: LanceDBStore):
    """Test the knowledge base with sample queries."""
    test_queries = [
        "What is 2+2?",
        "Solve x + 5 = 10",
        "Find the area of a circle with radius 3",
        "What is the derivative of x^2?"
    ]
    
    logger.info("Testing knowledge base with sample queries...")
    
    for query in test_queries:
        try:
            from app.embedding_client import get_embedding
            embedding = get_embedding(query)
            results = await store.semantic_search(
                query_embedding=embedding,
                top_k=2,
                threshold=0.3
            )
            logger.info(f"Query: '{query}' -> {len(results)} results")
            
        except Exception as e:
            logger.error(f"Test failed for query '{query}': {e}")

def verify_prerequisites():
    """Verify all required dependencies and API keys are available."""
    logger.info("🔍 Verifying prerequisites...")
    
    # Check API keys
    if not settings.GOOGLE_API_KEY or settings.GOOGLE_API_KEY == "your-google-api-key":
        logger.error("❌ GOOGLE_API_KEY not set or using placeholder value")
        return False
    
    if not settings.TAVILY_API_KEY or settings.TAVILY_API_KEY == "your-tavily-api-key":
        logger.warning("⚠️ TAVILY_API_KEY not set - web search will not work")
    
    # Check directories
    os.makedirs(os.path.dirname(settings.LANCEDB_PATH), exist_ok=True)
    
    logger.info("✅ Prerequisites verified")
    return True

def cleanup_existing_data():
    """Clean up existing knowledge base data."""
    logger.info("🧹 Cleaning up existing data...")
    
    # Remove existing parquet file
    if os.path.exists("knowledge_base.parquet"):
        os.remove("knowledge_base.parquet")
        logger.info("Removed existing knowledge_base.parquet")
    
    # Remove existing LanceDB directory
    import shutil
    if os.path.exists(settings.LANCEDB_PATH):
        shutil.rmtree(settings.LANCEDB_PATH)
        logger.info(f"Removed existing LanceDB at {settings.LANCEDB_PATH}")

async def main():
    """Main setup function."""
    try:
        print("🔧 Math Routing Agent - Knowledge Base Setup")
        print("=" * 50)
        
        # Verify prerequisites
        if not verify_prerequisites():
            sys.exit(1)
        
        # Clean up existing data
        cleanup_existing_data()
        
        # Setup knowledge base
        setup_knowledge_base()
        
        print("\n🎉 Setup completed successfully!")
        print(f"Knowledge base location: {settings.LANCEDB_PATH}")
        print(f"Parquet file: knowledge_base.parquet")
        
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())