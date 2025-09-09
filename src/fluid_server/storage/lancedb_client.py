"""
LanceDB client for vector storage and retrieval
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    import lancedb
    from lancedb.pydantic import LanceModel, Vector
    from lancedb.embeddings import get_registry
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False
    lancedb = None  # type: ignore
    LanceModel = None  # type: ignore
    Vector = None  # type: ignore
    get_registry = None  # type: ignore

logger = logging.getLogger(__name__)


class VectorDocument:
    """Base document class for vector storage"""
    
    def __init__(
        self, 
        id: str, 
        content: str, 
        vector: List[float], 
        metadata: Optional[Dict[str, Any]] = None,
        content_type: str = "text"
    ):
        self.id = id
        self.content = content
        self.vector = vector
        self.metadata = metadata or {}
        self.content_type = content_type

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary for storage"""
        return {
            "id": self.id,
            "content": self.content,
            "vector": self.vector,
            "metadata": self.metadata,
            "content_type": self.content_type
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorDocument":
        """Create document from dictionary"""
        return cls(
            id=data["id"],
            content=data["content"],
            vector=data["vector"],
            metadata=data.get("metadata", {}),
            content_type=data.get("content_type", "text")
        )


class LanceDBClient:
    """Client for LanceDB vector database operations"""

    # Class-level thread pool for database operations
    _db_executor: ThreadPoolExecutor | None = None

    @classmethod
    def get_executor(cls) -> ThreadPoolExecutor:
        """Get or create dedicated database thread pool"""
        if cls._db_executor is None:
            cls._db_executor = ThreadPoolExecutor(
                max_workers=2, 
                thread_name_prefix="LanceDB"
            )
        return cls._db_executor

    def __init__(self, db_path: Path, db_name: str = "embeddings"):
        """
        Initialize LanceDB client
        
        Args:
            db_path: Path to database directory
            db_name: Name of the database
        """
        if not LANCEDB_AVAILABLE:
            raise ImportError("LanceDB is not available. Install with: pip install lancedb")
            
        self.db_path = db_path
        self.db_name = db_name
        self.db: Any = None
        self._tables: Dict[str, Any] = {}
        
        # Ensure database directory exists
        self.db_path.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        """Initialize database connection"""
        try:
            loop = asyncio.get_event_loop()
            self.db = await loop.run_in_executor(
                self.get_executor(),
                self._initialize_sync
            )
            logger.info(f"LanceDB initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize LanceDB: {e}")
            raise

    def _initialize_sync(self) -> Any:
        """Synchronous database initialization"""
        db_uri = str(self.db_path / self.db_name)
        return lancedb.connect(db_uri)

    async def create_collection(
        self, 
        collection_name: str, 
        dimension: int,
        content_type: str = "text",
        overwrite: bool = False
    ) -> None:
        """
        Create a new collection/table
        
        Args:
            collection_name: Name of the collection
            dimension: Vector dimension
            content_type: Type of content stored
            overwrite: Whether to overwrite existing collection
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.get_executor(),
                self._create_collection_sync,
                collection_name,
                dimension,
                content_type,
                overwrite
            )
            logger.info(f"Created collection '{collection_name}' with dimension {dimension}")
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            raise

    def _create_collection_sync(
        self, 
        collection_name: str, 
        dimension: int,
        content_type: str,
        overwrite: bool
    ) -> None:
        """Synchronous collection creation"""
        # For now, use simple dict-based schema instead of dynamic Pydantic model
        # This avoids complex type checking issues with dynamic class creation

        # Check if collection exists
        existing_tables = self.db.table_names()
        
        if collection_name in existing_tables:
            if overwrite:
                # Drop existing table
                self.db.drop_table(collection_name)
                logger.info(f"Dropped existing collection '{collection_name}'")
            else:
                # Load existing table
                self._tables[collection_name] = self.db.open_table(collection_name)
                return

        # Create new table with simple schema
        # Start with empty data that matches the schema
        initial_data = [{
            "id": "placeholder",
            "content": "placeholder",
            "vector": [0.0] * dimension,
            "metadata": {},
            "content_type": content_type
        }]
        
        table = self.db.create_table(
            collection_name, 
            data=initial_data,
            mode="create"
        )
        
        # Remove placeholder record
        table.delete("id = 'placeholder'")
        
        self._tables[collection_name] = table

    async def list_collections(self) -> List[str]:
        """List all collections in the database"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.get_executor(),
                self._list_collections_sync
            )
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise

    def _list_collections_sync(self) -> List[str]:
        """Synchronous collection listing"""
        return self.db.table_names()

    async def insert_documents(
        self, 
        collection_name: str, 
        documents: List[VectorDocument]
    ) -> None:
        """
        Insert documents into a collection
        
        Args:
            collection_name: Name of the collection
            documents: List of documents to insert
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.get_executor(),
                self._insert_documents_sync,
                collection_name,
                documents
            )
            logger.info(f"Inserted {len(documents)} documents into '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to insert documents into '{collection_name}': {e}")
            raise

    def _insert_documents_sync(
        self, 
        collection_name: str, 
        documents: List[VectorDocument]
    ) -> None:
        """Synchronous document insertion"""
        if collection_name not in self._tables:
            table = self.db.open_table(collection_name)
            self._tables[collection_name] = table
        else:
            table = self._tables[collection_name]

        # Convert documents to dictionaries
        data = [doc.to_dict() for doc in documents]
        
        # Insert data
        table.add(data)

    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        filter_condition: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors
        
        Args:
            collection_name: Name of the collection
            query_vector: Query vector
            limit: Number of results to return
            filter_condition: Optional filter condition
            
        Returns:
            List of similar documents with scores
        """
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.get_executor(),
                self._search_vectors_sync,
                collection_name,
                query_vector,
                limit,
                filter_condition
            )
        except Exception as e:
            logger.error(f"Failed to search vectors in '{collection_name}': {e}")
            raise

    def _search_vectors_sync(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int,
        filter_condition: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Synchronous vector search"""
        if collection_name not in self._tables:
            table = self.db.open_table(collection_name)
            self._tables[collection_name] = table
        else:
            table = self._tables[collection_name]

        # Build search query
        search = table.search(query_vector).limit(limit)
        
        if filter_condition:
            search = search.where(filter_condition)
        
        # Execute search and convert to list
        results = search.to_list()
        
        # Add similarity scores (LanceDB returns _distance, we convert to similarity)
        for result in results:
            if '_distance' in result:
                # Convert distance to similarity score (0-1, higher is more similar)
                result['similarity_score'] = 1.0 / (1.0 + result['_distance'])
        
        return results

    async def get_document(
        self, 
        collection_name: str, 
        document_id: str
    ) -> Optional[VectorDocument]:
        """
        Get a specific document by ID
        
        Args:
            collection_name: Name of the collection
            document_id: ID of the document
            
        Returns:
            Document if found, None otherwise
        """
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.get_executor(),
                self._get_document_sync,
                collection_name,
                document_id
            )
            
            if result:
                return VectorDocument.from_dict(result)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document '{document_id}' from '{collection_name}': {e}")
            raise

    def _get_document_sync(
        self, 
        collection_name: str, 
        document_id: str
    ) -> Optional[Dict[str, Any]]:
        """Synchronous document retrieval"""
        if collection_name not in self._tables:
            table = self.db.open_table(collection_name)
            self._tables[collection_name] = table
        else:
            table = self._tables[collection_name]

        # Search for document by ID
        results = table.search().where(f"id = '{document_id}'").to_list()
        
        return results[0] if results else None

    async def delete_document(
        self, 
        collection_name: str, 
        document_id: str
    ) -> bool:
        """
        Delete a document by ID
        
        Args:
            collection_name: Name of the collection
            document_id: ID of the document to delete
            
        Returns:
            True if deleted, False if not found
        """
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.get_executor(),
                self._delete_document_sync,
                collection_name,
                document_id
            )
        except Exception as e:
            logger.error(f"Failed to delete document '{document_id}' from '{collection_name}': {e}")
            raise

    def _delete_document_sync(
        self, 
        collection_name: str, 
        document_id: str
    ) -> bool:
        """Synchronous document deletion"""
        if collection_name not in self._tables:
            table = self.db.open_table(collection_name)
            self._tables[collection_name] = table
        else:
            table = self._tables[collection_name]

        # Delete document
        table.delete(f"id = '{document_id}'")
        return True  # LanceDB doesn't return deletion count

    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics for a collection
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection statistics
        """
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.get_executor(),
                self._get_collection_stats_sync,
                collection_name
            )
        except Exception as e:
            logger.error(f"Failed to get stats for collection '{collection_name}': {e}")
            raise

    def _get_collection_stats_sync(self, collection_name: str) -> Dict[str, Any]:
        """Synchronous collection stats retrieval"""
        if collection_name not in self._tables:
            table = self.db.open_table(collection_name)
            self._tables[collection_name] = table
        else:
            table = self._tables[collection_name]

        # Get table info
        num_rows = table.count_rows()
        schema = table.schema
        
        return {
            "collection_name": collection_name,
            "num_documents": num_rows,
            "schema": str(schema)
        }

    async def close(self) -> None:
        """Close database connection and cleanup resources"""
        try:
            self._tables.clear()
            self.db = None
            logger.info("LanceDB connection closed")
        except Exception as e:
            logger.error(f"Error closing LanceDB: {e}")