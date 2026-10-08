import pytest

from tests.fakes.fake_embeddings import FakeEmbeddings
from tests.fakes.fake_vector_store import FakeVectorStore
from tests.fakes.fake_workflow_engine import FakeWorkflowEngine
from app.application.services.document_service import DocumentService
from app.application.services.query_service import QueryService
from app.infrastructure.state.in_memory_state_store import InMemoryStateStore


@pytest.fixture
def fake_services():
    embeddings = FakeEmbeddings()
    vector_store = FakeVectorStore()
    state_store = InMemoryStateStore()
    workflow_engine = FakeWorkflowEngine()

    return {
        "embeddings": embeddings,
        "vector_store": vector_store,
        "state_store": state_store,
        "document_service": DocumentService(vector_store, state_store, embeddings),
        "query_service": QueryService(workflow_engine),
    }
