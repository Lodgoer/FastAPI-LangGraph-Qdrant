from datetime import datetime, UTC
from typing import List, Dict, Any
import uuid

from app.domain.ports.vector_store import VectorStorePort
from app.domain.ports.state_store import StateStorePort
from app.domain.ports.embeddings import EmbeddingsPort

class DocumentService:

    def __init__(self, vector_store: VectorStorePort, state_store: StateStorePort, embeddings: EmbeddingsPort):
        self._vector_store = vector_store
        self._state_store = state_store
        self._embeddings = embeddings
    
    async def ingest_document(self, content: str, metadata: Dict[str, Any]) -> str:
        doc_id = str(uuid.uuid4())
        payload = {
            "content": content,
            "metadata": metadata or {},
            "ingested_at": datetime.now(UTC).isoformat(),
        }
        vector = self._embeddings.embed(content)
        self._vector_store.upsert_document(doc_id=doc_id, vector=vector, payload=payload)
        self._state_store.save(doc_id, payload)
        return doc_id

    async def list_documents(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self._vector_store.list_documents(limit=limit)

    async def delete_document(self, doc_id: str) -> None:
        self._vector_store.delete_document(doc_id)
        self._state_store.delete(doc_id)
