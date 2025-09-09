"""
Test script to verify LanceDB vector storage and search functionality
"""
import asyncio
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append('src')

from fluid_server.config import ServerConfig
from fluid_server.managers.embedding_manager import EmbeddingManager
from fluid_server.storage.lancedb_client import LanceDBClient, VectorDocument

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_vector_store_and_search():
    """Test complete vector storage and search workflow"""
    try:
        # Create config
        config = ServerConfig()
        logger.info(f"Using data root: {config.data_root}")
        
        # Create embedding manager
        embedding_manager = EmbeddingManager(config)
        await embedding_manager.initialize()
        
        # Create LanceDB client
        lancedb_client = LanceDBClient(
            db_path=config.embeddings_db_path,
            db_name=config.embeddings_db_name
        )
        await lancedb_client.initialize()
        
        # Test documents
        test_docs = [
            "The quick brown fox jumps over the lazy dog",
            "Python is a powerful programming language for AI",
            "Machine learning models can process natural language",
            "Vector databases store high-dimensional embeddings",
            "Similarity search finds related documents efficiently"
        ]
        
        # Generate embeddings
        logger.info("Generating embeddings for test documents...")
        embeddings = await embedding_manager.get_text_embeddings(test_docs)
        logger.info(f"Generated {len(embeddings)} embeddings with dimension {len(embeddings[0])}")
        
        # Create collection
        collection_name = "test_documents"
        logger.info(f"Creating collection '{collection_name}'...")
        await lancedb_client.create_collection(
            collection_name=collection_name,
            dimension=len(embeddings[0]),  # 384 for MiniLM
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
                metadata={},  # Keep metadata simple for now
                content_type="text"
            )
            vector_docs.append(vector_doc)
        
        # Insert documents
        logger.info("Inserting documents into vector store...")
        await lancedb_client.insert_documents(collection_name, vector_docs)
        logger.info(f"Inserted {len(vector_docs)} documents")
        
        # Test search
        query = "programming language for artificial intelligence"
        logger.info(f"Searching for: '{query}'")
        
        # Generate query embedding
        query_embeddings = await embedding_manager.get_text_embeddings([query])
        query_vector = query_embeddings[0]
        
        # Search
        results = await lancedb_client.search_vectors(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=3
        )
        
        logger.info(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            logger.info(f"  {i+1}. '{result['content']}' (similarity: {result.get('similarity_score', 'N/A'):.4f})")
        
        # Test retrieval by ID
        logger.info("Testing document retrieval by ID...")
        doc = await lancedb_client.get_document(collection_name, "doc_1")
        if doc:
            logger.info(f"Retrieved doc_1: '{doc.content}'")
        
        # Get collection stats
        stats = await lancedb_client.get_collection_stats(collection_name)
        logger.info(f"Collection stats: {stats['num_documents']} documents")
        
        # Clean up
        await embedding_manager.shutdown()
        await lancedb_client.close()
        logger.info("Vector store test completed successfully!")
        
    except Exception as e:
        logger.error(f"Vector store test failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(test_vector_store_and_search())