import os
from dataclasses import dataclass
from qdrant_client import QdrantClient

from app.core.config_dev import DevSettings
from app.core.config_staging import StagingSettings
from app.core.config_prod import ProdSettings
from app.core.logging import setup_logging
from app.domain.ports.embeddings import EmbeddingsPort
from app.domain.ports.vector_store import VectorStorePort
from app.domain.ports.workflow import WorkflowEnginePort
from app.domain.ports.state_store import StateStorePort
from app.application.services.document_service import DocumentService
from app.application.services.query_service import QueryService
from app.application.services.counter_service import CounterService
from app.infrastructure.embeddings.random_embeddings import RandomEmbeddingsService
from app.infrastructure.vectorstores.qdrant_vector_store import QdrantVectorStore
from app.infrastructure.workflows.langgraph_workflow import LangGraphWorkflowEngine
from app.infrastructure.state.in_memory_state_store import InMemoryStateStore


def load_settings():
    env = os.getenv("ENVIRONMENT", "dev").lower()
    if env == "prod":
        return ProdSettings()
    if env == "staging":
        return StagingSettings()
    return DevSettings()


@dataclass
class AppContainer:
    settings: object
    qdrant_client: QdrantClient
    embeddings: EmbeddingsPort
    vector_store: VectorStorePort
    workflow_engine: WorkflowEnginePort
    state_store: StateStorePort
    document_service: DocumentService
    query_service: QueryService
    counter_service: CounterService


def build_container() -> AppContainer:
    settings = load_settings()
    setup_logging(settings)

    qdrant_client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )

    embeddings = RandomEmbeddingsService(dim=settings.embedding_dim)
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=settings.qdrant_collection_name,
        dim=settings.embedding_dim,
    )
    state_store = InMemoryStateStore()
    workflow_engine = LangGraphWorkflowEngine(
        embeddings=embeddings,
        vector_store=vector_store,
        state_store=state_store,
    )

    document_service = DocumentService(
        vector_store=vector_store,
        state_store=state_store,
        embeddings=embeddings,

    )
    query_service = QueryService(workflow_engine=workflow_engine)
    counter_service = CounterService()

    return AppContainer(
        settings=settings,
        qdrant_client=qdrant_client,
        embeddings=embeddings,
        vector_store=vector_store,
        workflow_engine=workflow_engine,
        state_store=state_store,
        document_service=document_service,
        query_service=query_service,
        counter_service=counter_service,
    )
