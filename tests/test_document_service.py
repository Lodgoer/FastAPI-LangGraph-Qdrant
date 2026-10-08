import pytest

from app.application.services.document_service import DocumentService
from tests.fakes.fake_vector_store import FakeVectorStore


class FailingVectorStore(FakeVectorStore):
    def upsert_document(self, doc_id, vector, payload):
        raise RuntimeError("qdrant down")


@pytest.mark.asyncio
async def test_ingest_and_list(fake_services):
    ds = fake_services["document_service"]

    doc_id = await ds.ingest_document("Hello world", {"tag": "test"})

    docs = await ds.list_documents()
    assert len(docs) == 1
    assert docs[0]["id"] == doc_id
    assert docs[0]["content"] == "Hello world"


@pytest.mark.asyncio
async def test_delete(fake_services):
    ds = fake_services["document_service"]

    doc_id = await ds.ingest_document("Doc", {})
    await ds.delete_document(doc_id)

    docs = await ds.list_documents()
    assert len(docs) == 0


@pytest.mark.asyncio
async def test_ingest_does_not_touch_state_store_when_vector_store_fails(fake_services):
    state_store = fake_services["state_store"]
    ds = DocumentService(FailingVectorStore(), state_store, fake_services["embeddings"])

    with pytest.raises(RuntimeError):
        await ds.ingest_document("Hello", {})

    assert state_store.count() == 0