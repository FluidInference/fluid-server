"""
Simple test script to verify embedding model download and basic functionality
"""
import asyncio
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append('src')

from fluid_server.config import ServerConfig
from fluid_server.managers.embedding_manager import EmbeddingManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_embedding_download_and_usage():
    """Test downloading and using embedding models"""
    try:
        # Create config
        config = ServerConfig()
        logger.info(f"Using embedding model: {config.embedding_model}")
        logger.info(f"Data root: {config.data_root}")
        logger.info(f"Model path: {config.model_path}")
        
        # Create embedding manager
        embedding_manager = EmbeddingManager(config)
        
        # Initialize (this should trigger download if needed)
        logger.info("Initializing embedding manager...")
        await embedding_manager.initialize()
        
        # Test text embeddings
        logger.info("Testing text embeddings...")
        test_texts = [
            "Hello world",
            "This is a test sentence for embeddings",
            "AI and machine learning are fascinating topics"
        ]
        
        embeddings = await embedding_manager.get_text_embeddings(test_texts)
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        for i, embedding in enumerate(embeddings):
            logger.info(f"Text {i+1}: '{test_texts[i]}' -> embedding dimension: {len(embedding)}")
            logger.info(f"First 5 values: {embedding[:5]}")
        
        # Test similarity between first two texts
        if len(embeddings) >= 2:
            import numpy as np
            emb1 = np.array(embeddings[0])
            emb2 = np.array(embeddings[1])
            
            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            logger.info(f"Similarity between first two texts: {similarity:.4f}")
        
        # Clean up
        await embedding_manager.shutdown()
        logger.info("Test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(test_embedding_download_and_usage())