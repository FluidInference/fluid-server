"""
Vector store management endpoints for LanceDB operations
"""

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from ...managers.embedding_manager import EmbeddingManager
from ...storage.lancedb_client import LanceDBClient, VectorDocument
from ...models.openai import (
    VectorStoreInsertRequest,
    VectorStoreInsertResponse,
    VectorStoreSearchRequest,
    VectorStoreSearchResponse,
    VectorStoreSearchResult,
    CollectionListResponse,
    CollectionInfo,
    CreateCollectionRequest
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/vector_store")


def get_embedding_manager(request: Request) -> EmbeddingManager:
    """Dependency to get embedding manager"""
    return request.app.embedding_manager


def get_lancedb_client(request: Request) -> LanceDBClient:
    """Dependency to get LanceDB client"""
    return request.app.lancedb_client


@router.post("/collections")
async def create_collection(
    request: CreateCollectionRequest,
    lancedb_client: Annotated[LanceDBClient, Depends(get_lancedb_client)]
) -> JSONResponse:
    """
    Create a new vector store collection
    """
    try:
        await lancedb_client.create_collection(
            collection_name=request.name,
            dimension=request.dimension,
            content_type=request.content_type,
            overwrite=request.overwrite
        )
        
        return JSONResponse(content={
            "collection_name": request.name,
            "dimension": request.dimension,
            "content_type": request.content_type,
            "created": True
        })
        
    except Exception as e:
        logger.error(f"Error creating collection '{request.name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections")
async def list_collections(
    lancedb_client: Annotated[LanceDBClient, Depends(get_lancedb_client)]
) -> CollectionListResponse:
    """
    List all vector store collections
    """
    try:
        collection_names = await lancedb_client.list_collections()
        collections = []
        
        for name in collection_names:
            try:
                stats = await lancedb_client.get_collection_stats(name)
                collections.append(
                    CollectionInfo(
                        name=name,
                        num_documents=stats["num_documents"],
                        content_types=["text"]  # Default, could be enhanced
                    )
                )
            except Exception as e:
                logger.warning(f"Could not get stats for collection '{name}': {e}")
                collections.append(
                    CollectionInfo(
                        name=name,
                        num_documents=0,
                        content_types=["text"]
                    )
                )
        
        return CollectionListResponse(
            collections=collections,
            total_collections=len(collections)
        )
        
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/insert")
async def insert_documents(
    request: VectorStoreInsertRequest,
    embedding_manager: Annotated[EmbeddingManager, Depends(get_embedding_manager)],
    lancedb_client: Annotated[LanceDBClient, Depends(get_lancedb_client)]
) -> VectorStoreInsertResponse:
    """
    Insert documents into a vector store collection
    """
    try:
        if not embedding_manager.config.enable_embeddings:
            raise HTTPException(
                status_code=503,
                detail="Embeddings functionality is disabled"
            )
        
        # Extract text content for embedding generation
        texts = [doc.content for doc in request.documents]
        
        # Generate embeddings using specified model or default
        model_name = request.model or embedding_manager.config.embedding_model
        embeddings = await embedding_manager.get_text_embeddings(
            texts=texts,
            model_name=model_name
        )
        
        # Create VectorDocument objects
        vector_documents = []
        inserted_ids = []
        
        for i, (doc, embedding) in enumerate(zip(request.documents, embeddings)):
            # Use provided ID or generate one
            doc_id = doc.id if doc.id else str(uuid.uuid4())
            inserted_ids.append(doc_id)
            
            vector_doc = VectorDocument(
                id=doc_id,
                content=doc.content,
                vector=embedding,
                metadata=doc.metadata or {},
                content_type=doc.content_type
            )
            vector_documents.append(vector_doc)
        
        # Insert into LanceDB
        await lancedb_client.insert_documents(
            collection_name=request.collection,
            documents=vector_documents
        )
        
        logger.info(
            f"Inserted {len(vector_documents)} documents into collection '{request.collection}'"
        )
        
        return VectorStoreInsertResponse(
            inserted_count=len(vector_documents),
            collection=request.collection,
            ids=inserted_ids
        )
        
    except Exception as e:
        logger.error(f"Error inserting documents: {e}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_vectors(
    request: VectorStoreSearchRequest,
    embedding_manager: Annotated[EmbeddingManager, Depends(get_embedding_manager)],
    lancedb_client: Annotated[LanceDBClient, Depends(get_lancedb_client)]
) -> VectorStoreSearchResponse:
    """
    Search for similar vectors in a collection
    """
    try:
        if not embedding_manager.config.enable_embeddings:
            raise HTTPException(
                status_code=503,
                detail="Embeddings functionality is disabled"
            )
        
        # Generate query embedding based on query type
        if request.query_type != "text":
            raise HTTPException(
                status_code=400,
                detail="Only text query_type is supported for embeddings"
            )

        if isinstance(request.query, bytes):
            raise HTTPException(
                status_code=400,
                detail="Text query must be a string"
            )

        model_name = request.model or embedding_manager.config.embedding_model
        query_embeddings = await embedding_manager.get_text_embeddings(
            texts=[request.query],
            model_name=model_name
        )
        query_vector = query_embeddings[0]
        
        # Perform vector search
        search_results = await lancedb_client.search_vectors(
            collection_name=request.collection,
            query_vector=query_vector,
            limit=request.limit,
            filter_condition=request.filter
        )
        
        # Convert results to response format
        results = []
        for result in search_results:
            results.append(
                VectorStoreSearchResult(
                    id=result["id"],
                    content=result["content"],
                    metadata=result.get("metadata", {}),
                    similarity_score=result.get("similarity_score", 0.0),
                    content_type=result.get("content_type", "text")
                )
            )
        
        logger.info(
            f"Found {len(results)} results for {request.query_type} query in collection '{request.collection}'"
        )
        
        return VectorStoreSearchResponse(
            results=results,
            collection=request.collection,
            query_type=request.query_type,
            total_results=len(results)
        )
        
    except Exception as e:
        logger.error(f"Error searching vectors: {e}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/{collection}/{document_id}")
async def get_document(
    collection: str,
    document_id: str,
    lancedb_client: Annotated[LanceDBClient, Depends(get_lancedb_client)]
) -> JSONResponse:
    """
    Get a specific document by ID
    """
    try:
        document = await lancedb_client.get_document(collection, document_id)
        
        if not document:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{document_id}' not found in collection '{collection}'"
            )
        
        return JSONResponse(content={
            "id": document.id,
            "content": document.content,
            "metadata": document.metadata,
            "content_type": document.content_type
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{collection}/{document_id}")
async def delete_document(
    collection: str,
    document_id: str,
    lancedb_client: Annotated[LanceDBClient, Depends(get_lancedb_client)]
) -> JSONResponse:
    """
    Delete a document by ID
    """
    try:
        success = await lancedb_client.delete_document(collection, document_id)
        
        if success:
            return JSONResponse(content={
                "deleted": True,
                "collection": collection,
                "document_id": document_id
            })
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{document_id}' not found in collection '{collection}'"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{collection}/stats")
async def get_collection_stats(
    collection: str,
    lancedb_client: Annotated[LanceDBClient, Depends(get_lancedb_client)]
) -> JSONResponse:
    """
    Get statistics for a collection
    """
    try:
        stats = await lancedb_client.get_collection_stats(collection)
        return JSONResponse(content=stats)
        
    except Exception as e:
        logger.error(f"Error getting collection stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

