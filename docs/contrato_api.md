# Contrato da API — StrikeMetrics RAG CS:GO
**Versão:** 1.0.0  
**Data:** Junho 2026  
**Instituição:** Facens  
**Projeto:** Engenharia — Análise de Dados de eSports (CS:GO)

---

## 1. Visão Geral

Este documento formaliza o contrato técnico completo da StrikeMetrics RAG API. O sistema permite consultas em linguagem natural sobre estatísticas de combate do CS:GO utilizando arquitetura RAG (Retrieval-Augmented Generation) com dados estruturados na camada Gold (MinIO), indexação vetorial no Milvus e modelos de linguagem locais via Ollama.

### 1.1. Servidor

| Ambiente | URL Base |
|----------|----------|
| Desenvolvimento | `http://localhost:8000` |
| Documentação Interativa (Swagger) | `http://localhost:8000/docs` |
| Documentação Alternativa (ReDoc) | `http://localhost:8000/redoc` |

### 1.2. Formato de Dados

- **Content-Type:** `application/json`
- **Encoding:** UTF-8
- **Protocolo:** HTTP/1.1

### 1.3. Versionamento

A API utiliza versionamento semântico (SemVer). A versão atual é `1.0.0`.  
Versões futuras com breaking changes serão prefixadas na URL: `/v2/query`.  
A versão atual não exige prefixo de versão na URL.

---

## 2. Fluxo Completo de uma Requisição

O diagrama abaixo descreve o que acontece internamente quando o endpoint `/query` é chamado:

```
Cliente (Gradio / curl / Python)
        │
        │  POST /query
        │  { "model_name": "llama3.2", "question": "..." }
        ▼
┌─────────────────────────────────┐
│         FastAPI (porta 8000)    │
│  1. Validação Pydantic          │
│  2. Normalização da pergunta    │
│     (PT → EN para termos-chave) │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│    Ollama — nomic-embed-text    │
│  Gera embedding da pergunta     │
│  Vetor de 768 dimensões         │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│         Milvus (porta 19530)    │
│  Busca semântica COSINE         │
│  Retorna top-K chunks           │
│  (K=12 por padrão)              │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Montagem do Prompt             │
│  contexto + pergunta → prompt   │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│    Ollama — LLM (llama3.2 etc)  │
│  Geração da resposta            │
└────────────────┬────────────────┘
                 │
                 ▼
        Resposta ao cliente
        { "answer": "...", "logs": "..." }
```

---

## 3. Modelos Suportados

| Modelo | Tipo | Uso |
|--------|------|-----|
| `nomic-embed-text` | Embedding | Geração de vetores (interno, não selecionável) |
| `llama3.2` | LLM | Geração de resposta — **padrão recomendado** |
| `llama3` | LLM | Alternativa ao llama3.2 |
| `mistral` | LLM | Alternativa leve |
| `gemma2` | LLM | Alternativa Google |
| `phi3` | LLM | Alternativa Microsoft |

> **Nota:** Modelos LLM devem estar previamente baixados no Ollama. O modelo `nomic-embed-text` é usado internamente para embeddings e não pode ser selecionado pelo cliente.

---

## 4. Catálogo de Endpoints

### 4.1. Health Check — `GET /`

Verificação de integridade do serviço.

**Resposta 200 OK:**
```json
{
  "status": "ok",
  "service": "StrikeMetrics RAG API"
}
```

---

### 4.2. Listar Modelos — `GET /models`

Retorna todos os modelos LLM disponíveis para uso no endpoint `/query`.

**Resposta 200 OK:**
```json
{
  "models": [
    "llama3.2",
    "llama3",
    "mistral",
    "gemma2",
    "phi3"
  ]
}
```

---

### 4.3. Metadados do Sistema — `GET /metadata`

Retorna informações sobre a infraestrutura, versões e fontes de dados.

**Resposta 200 OK:**
```json
{
  "api_version": "1.0.0",
  "supported_models": ["llama3.2", "llama3", "mistral", "gemma2", "phi3"],
  "vector_store": "Milvus",
  "embedding_model": "nomic-embed-text (Ollama)",
  "data_sources": [
    "csgo_combat_stats (Kaggle → MinIO Gold)",
    "csgo_bomb_stats (Kaggle → MinIO Gold)"
  ],
  "endpoints": ["/", "/models", "/metadata", "/query", "/rebuild-index"],
  "timestamp": "2026-06-08T14:30:00.000000Z"
}
```

---

### 4.4. Consulta RAG — `POST /query`

Endpoint principal. Executa o pipeline RAG completo e retorna a resposta gerada pelo LLM.

**Request Body:**
```json
{
  "model_name": "llama3.2",
  "question": "Qual arma causa mais dano médio na cabeça?"
}
```

**Regras de Validação:**

| Campo | Tipo | Obrigatório | Regras |
|-------|------|-------------|--------|
| `model_name` | string | Não (default: `llama3.2`) | Deve estar na lista de modelos suportados |
| `question` | string | **Sim** | Mínimo 5, máximo 500 caracteres. Deve conter ao menos uma letra. Não pode ser só espaços. |

**Resposta 200 OK:**
```json
{
  "answer": "De acordo com os dados de combate, a AK-47 apresenta o maior índice de dano médio quando atinge a hitbox da cabeça, registrando valores superiores aos da M4A1-S.",
  "logs": "Iniciando busca vetorial no Milvus...\n12 documentos recuperados\nGerando resposta com o modelo 'llama3.2' via Ollama...\nResposta gerada com sucesso."
}
```

**Resposta 422 Unprocessable Entity — Validação falhou:**
```json
{
  "detail": [
    {
      "loc": ["body", "question"],
      "msg": "String should have at least 5 characters",
      "type": "string_too_short"
    }
  ]
}
```

**Resposta 400 Bad Request — Modelo inválido:**
```json
{
  "detail": [
    {
      "loc": ["body", "model_name"],
      "msg": "Modelo 'gpt-4' não suportado. Modelos disponíveis: ['llama3.2', 'llama3', 'mistral', 'gemma2', 'phi3']",
      "type": "value_error"
    }
  ]
}
```

---

### 4.5. Reconstruir Índice — `POST /rebuild-index`

Endpoint administrativo. Apaga e recria toda a coleção no Milvus.

> ⚠️ **Atenção:** Este endpoint apaga todos os vetores existentes. Use com cautela.

**Resposta 200 OK:**
```json
{
  "message": "Índice vetorial reconstruído com sucesso."
}
```

**Resposta 500 Internal Server Error:**
```json
{
  "detail": "Erro na pipeline RAG: falha ao conectar no host milvus:19530"
}
```

---

## 5. Tabela Completa de Códigos HTTP

| Código | Significado | Quando ocorre |
|--------|-------------|---------------|
| `200` | OK | Requisição processada com sucesso |
| `400` | Bad Request | Modelo inválido na requisição |
| `404` | Not Found | Endpoint inexistente |
| `422` | Unprocessable Entity | Validação de campos falhou (Pydantic) |
| `500` | Internal Server Error | Falha no Milvus, Ollama ou MinIO |

---

## 6. Exemplos Práticos de Uso

### 6.1. curl

**Health check:**
```bash
curl http://localhost:8000/
```

**Listar modelos:**
```bash
curl http://localhost:8000/models
```

**Consulta RAG:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"model_name": "llama3.2", "question": "Qual arma causa mais dano na cabeça?"}'
```

**Metadados:**
```bash
curl http://localhost:8000/metadata
```

### 6.2. Python

```python
import requests

BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/")
print(response.json())

# Consulta RAG
payload = {
    "model_name": "llama3.2",
    "question": "Qual arma causa mais dano médio na cabeça?"
}
response = requests.post(f"{BASE_URL}/query", json=payload, timeout=120)
data = response.json()
print(data["answer"])
```

### 6.3. JavaScript (fetch)

```javascript
const response = await fetch("http://localhost:8000/query", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model_name: "llama3.2",
    question: "Qual arma causa mais dano médio na cabeça?"
  })
});
const data = await response.json();
console.log(data.answer);
```

---

## 7. Infraestrutura e Segurança

### 7.1. CORS

| Configuração | Valor |
|-------------|-------|
| `allow_origins` | `*` (qualquer origem) |
| `allow_methods` | `*` (GET, POST, OPTIONS, etc.) |
| `allow_headers` | `*` |

> Em produção, substituir `*` pelos domínios específicos da aplicação.

### 7.2. SLA e Timeouts

| Operação | Tempo Esperado | Timeout Recomendado |
|----------|---------------|---------------------|
| Health check | < 100ms | 5s |
| Listar modelos | < 100ms | 5s |
| Metadados | < 100ms | 5s |
| Consulta RAG (CPU) | 30–120s | 180s |
| Rebuild de índice | 10–30min | 1800s |

> **Hardware de referência:** Intel i5, 16GB RAM, sem GPU dedicada.

### 7.3. Autenticação

A versão `1.0.0` não implementa autenticação. O sistema foi projetado para uso local em ambiente controlado. Versões futuras poderão implementar API Keys via header `X-API-Key`.

---

## 8. Limitações Conhecidas

| Limitação | Descrição |
|-----------|-----------|
| **CPU-only** | Sem GPU, cada consulta RAG leva 30–120s |
| **Modelos locais** | Requer download prévio dos modelos no Ollama |
| **Dados CS:GO** | Restrito ao dataset Kaggle CS:GO Matchmaking Damage |
| **Sem autenticação** | Endpoints abertos, uso local apenas |
| **Sem rate limiting** | Sem controle de requisições simultâneas |
| **Língua** | Respostas sempre em português, perguntas em PT ou EN |

---

## 9. Glossário

| Termo | Definição |
|-------|-----------|
| **RAG** | Retrieval-Augmented Generation — técnica que combina busca semântica com geração de texto |
| **Embedding** | Representação numérica (vetor) de um texto, capturando seu significado semântico |
| **Milvus** | Banco de dados vetorial usado para armazenar e buscar embeddings |
| **Ollama** | Ferramenta para rodar modelos de linguagem (LLMs) localmente |
| **MinIO** | Object storage compatível com S3, usado para armazenar os dados nas camadas Bronze/Silver/Gold |
| **Medallion** | Arquitetura de dados em camadas: Bronze (raw) → Silver (limpo) → Gold (agregado) |
| **LLM** | Large Language Model — modelo de linguagem de grande escala |
| **Vector Store** | Banco de dados especializado em busca por similaridade vetorial |
| **Chunk** | Trecho de texto extraído dos dados Gold para indexação no Milvus |
| **COSINE** | Métrica de similaridade usada para comparar vetores no Milvus |

---

## 10. Changelog

| Versão | Data | Alterações |
|--------|------|-----------|
| `1.0.0` | Junho 2026 | Versão inicial — endpoints `/query`, `/metadata`, `/models`, `/rebuild-index` |

---

*Organização: StrikeMetrics Solutions | Instituição: Facens*