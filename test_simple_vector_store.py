"""
Simple test script to verify basic LanceDB vector storage functionality
"""
import asyncio
import sys
import logging
import numpy as np

# Add src to path
sys.path.append('src')

from fluid_server.config import ServerConfig
from fluid_server.storage.lancedb_client import LanceDBClient, VectorDocument

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_simple_vector_store():
    """Test basic vector storage and search with fake embeddings"""
    try:
        # Create config
        config = ServerConfig()
        logger.info(f"Using data root: {config.data_root}")
        
        # Create LanceDB client
        lancedb_client = LanceDBClient(
            db_path=config.embeddings_db_path,
            db_name=config.embeddings_db_name
        )
        await lancedb_client.initialize()
        
        # Create fake embeddings (384 dimensions like MiniLM)
        np.random.seed(42)  # For reproducibility
        dimension = 384
        
        # Test documents with fake embeddings
        test_docs = [
            "The quick brown fox jumps over the lazy dog",
            "Python is a powerful programming language for AI",
            "Machine learning models can process natural language",
            "Vector databases store high-dimensional embeddings",
            "Similarity search finds related documents efficiently"
        ]
        
        # Generate fake embeddings (normally distributed)
        embeddings = []
        for i in range(len(test_docs)):
            embedding = np.random.normal(0, 1, dimension).tolist()
            embeddings.append(embedding)
        
        logger.info(f"Generated {len(embeddings)} fake embeddings with dimension {len(embeddings[0])}")
        
        # Create collection
        collection_name = "test_documents_simple"
        logger.info(f"Creating collection '{collection_name}'...")
        await lancedb_client.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            content_type="text",
            overwrite=True
        )
        
        # Create vector documents
        vector_docs = []
        for i, (text, embedding) in enumerate(zip(test_docs, embeddings)):
            vector_doc = VectorDocument(
                id=f"doc_{i}",
                content=text,
                vector=embedding,
                metadata={},
                content_type="text"
            )
            vector_docs.append(vector_doc)
        
        # Insert documents
        logger.info("Inserting documents into vector store...")
        await lancedb_client.insert_documents(collection_name, vector_docs)
        logger.info(f"Inserted {len(vector_docs)} documents")
        
        # Test search with a fake query vector (similar to first doc)
        query_vector = np.array(embeddings[0]) + np.random.normal(0, 0.1, dimension)  # Add small noise
        query_vector = query_vector.tolist()
        
        logger.info("Searching with fake query vector...")
        
        # Search
        results = await lancedb_client.search_vectors(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=3
        )
        
        logger.info(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            similarity = result.get('similarity_score', 'N/A')
            logger.info(f"  {i+1}. '{result['content'][:50]}...' (similarity: {similarity})")
        
        # Test retrieval by ID
        logger.info("Testing document retrieval by ID...")
        doc = await lancedb_client.get_document(collection_name, "doc_1")
        if doc:
            logger.info(f"Retrieved doc_1: '{doc.content[:50]}...'")
        
        # Get collection stats
        stats = await lancedb_client.get_collection_stats(collection_name)
        logger.info(f"Collection stats: {stats['num_documents']} documents")
        
        # Test collection listing
        collections = await lancedb_client.list_collections()
        logger.info(f"Available collections: {collections}")
        
        # Clean up
        await lancedb_client.close()
        logger.info("Simple vector store test completed successfully!")
        
    except Exception as e:
        logger.error(f"Simple vector store test failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(test_simple_vector_store())