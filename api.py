"""
API FastAPI — StrikeMetrics Solutions RAG CS:GO
Expõe os endpoints para integração com o app.py (Gradio) e outros clientes.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from main import SUPPORTED_MODELS, build_vector_store, get_vector_store, query


# ==========================================
# SCHEMAS (Request / Response)
# ==========================================

class QueryRequest(BaseModel):
    model_name: str = Field(
        default="llama3.2",
        description="Nome do modelo Ollama a ser usado na geração.",
        examples=["llama3.2", "mistral"],
    )
    question: str = Field(
        ...,
        min_length=1,
        description="Pergunta em linguagem natural sobre estatísticas de CS:GO.",
        examples=["Qual arma causa mais dano médio na cabeça?"],
    )


class QueryResponse(BaseModel):
    answer: str = Field(description="Resposta gerada pelo LLM.")
    logs: str = Field(description="Logs internos do pipeline RAG.")


class RebuildResponse(BaseModel):
    message: str


class ModelsResponse(BaseModel):
    models: list[str]


# ==========================================
# LIFESPAN (warm-up do vector store)
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega (ou cria) o índice vetorial uma única vez na inicialização."""
    print("[API] Inicializando vector store...")
    get_vector_store()
    print("[API] Vector store pronto.")
    yield
    print("[API] Encerrando API.")


# ==========================================
# APLICAÇÃO
# ==========================================

app = FastAPI(
    title="StrikeMetrics RAG API",
    description=(
        "API para consultas RAG sobre estatísticas de combate e bomba do CS:GO. "
        "Utiliza Milvus como vector store e modelos Ollama locais para geração."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # ajuste para seus domínios em produção
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# ENDPOINTS
# ==========================================

@app.get("/", tags=["health"])
def root():
    """Health check básico."""
    return {"status": "ok", "service": "StrikeMetrics RAG API"}


@app.get("/models", response_model=ModelsResponse, tags=["config"])
def list_models():
    """Retorna a lista de modelos Ollama disponíveis."""
    return ModelsResponse(models=SUPPORTED_MODELS)


@app.post("/query", response_model=QueryResponse, tags=["rag"])
def run_query(body: QueryRequest):
    """
    Executa o pipeline RAG completo:
    1. Busca semântica no Milvus pelos chunks mais relevantes.
    2. Monta o prompt com o contexto recuperado.
    3. Gera a resposta com o modelo Ollama escolhido.
    """
    if body.model_name not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Modelo '{body.model_name}' não suportado. Use GET /models para ver as opções.",
        )

    answer, logs = query(body.model_name, body.question)
    return QueryResponse(answer=answer, logs=logs)


@app.post("/rebuild-index", response_model=RebuildResponse, tags=["admin"])
def rebuild_index():
    """
    Força a reconstrução completa do índice vetorial no Milvus.
    Use com cuidado: apaga e recria toda a coleção.
    """
    try:
        build_vector_store(force_rebuild=True)
        return RebuildResponse(message="Índice vetorial reconstruído com sucesso.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# EXECUÇÃO DIRETA
# ==========================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
