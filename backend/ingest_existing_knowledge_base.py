#!/usr/bin/env python3
"""
Script to ingest your existing knowledge_base.parquet into LanceDB.
"""

import logging
import sys
from pathlib import Path

# Add backend to path so we can import our modules
sys.path.append('backend')

from backend.app.core.lancedb_store import LanceDBStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Ingest the existing knowledge_base.parquet into LanceDB.
    """
    # Check if the parquet file exists
    parquet_path = "knowledge_base.parquet"
    if not Path(parquet_path).exists():
        logger.error("❌ Could not find %s", parquet_path)
        logger.info("Make sure you're running this from the directory containing knowledge_base.parquet")
        return False
    
    logger.info("📁 Found %s", parquet_path)
    
    # Create LanceDB store
    store = LanceDBStore(
        db_path="backend/lancedb_math",
        table_name="math_qa",
        vector_column="embedding",  # This matches your knowledge_loader.py
    )
    
    try:
        # Ingest the data
        logger.info("🚀 Starting ingestion...")
        store.ingest_from_parquet(parquet_path, overwrite=True)
        logger.info("✅ Ingestion completed successfully!")
        
        # Test the table contents
        logger.info("🔍 Checking table contents...")
        if store.debug_table_contents():
            logger.info("✅ Table contents verified!")
        else:
            logger.warning("⚠️  Could not verify table contents")
        
        # Test search functionality
        logger.info("🔍 Testing search functionality...")
        if store.test_search():
            logger.info("✅ Search test passed!")
        else:
            logger.warning("⚠️  Search test failed")
            
        return True
        
    except Exception as e:
        logger.error("❌ Ingestion failed: %s", e)
        logger.exception("Full error details:")
        return False

if __name__ == "__main__":
    print("🚀 Ingesting existing knowledge base into LanceDB...")
    
    success = main()
    
    if success:
        print("\n🎉 Setup complete! Your knowledge base is ready.")
        print("\nNow start your server and test with:")
        print("  cd backend")
        print("  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
        print("\nTest queries:")
        print("  curl -X POST 'http://127.0.0.1:8000/api/v1/solve' \\")
        print("    -H 'Content-Type: application/json' \\")
        print("    -d '{\"question\": \"What is 15 * 3?\"}'")
    else:
        print("\n❌ Setup failed. Check the logs above for details.")
        sys.exit(1)