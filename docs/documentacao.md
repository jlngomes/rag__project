# Como Rodar o Projeto do Zero

## Pré-requisitos

- Docker e Docker Compose instalados
- Git instalado
- Conta no Kaggle (para download do dataset)
- Mínimo 16GB RAM e 20GB de espaço em disco

## ⚠️ Atenção — Ollama local

Se você tiver o Ollama instalado diretamente na máquina, ele ocupa a porta 11434 e vai conflitar com o Docker. Pare ele antes de subir o projeto:

```bash
# Se instalado via sistema:
sudo systemctl stop ollama

# Se instalado via Snap:
sudo snap stop ollama
sudo snap disable ollama

# Confirma que a porta está livre:
sudo lsof -i :11434
```

Se não retornar nada, pode continuar.

---

## 1. Clonar o repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd rag__project
```

## 2. Configurar credenciais do Kaggle

```bash
cp .env.example .env
```

Edite o `.env` e adicione seu token do Kaggle:

```
KAGGLE_API_TOKEN={"username":"SEU_USUARIO","key":"SUA_CHAVE"}
```

Para obter o token: kaggle.com → Account → API → Create New Token.

## 3. Subir toda a infraestrutura

```bash
make up
```

Este comando sobe todos os serviços e executa o pipeline automaticamente:

| Serviço | Função | Porta |
|---------|--------|-------|
| MinIO | Object storage (Bronze/Silver/Gold) | 9000, 9001 |
| PostgreSQL | Metadados e MLflow backend | 5433 |
| Milvus + etcd | Vector database | 19530 |
| Ollama | LLMs locais | 11434 |
| MLflow | Rastreamento de experimentos | 5000 |
| API (FastAPI) | Endpoints RAG | 8000 |
| Frontend (Gradio) | Interface de usuário | 7860 |

## 4. Acompanhar o pipeline completo

O pipeline executa automaticamente em ordem. Acompanhe cada etapa:

```bash
# 1. Ingestão do dataset (~7 min — download de 600MB do Kaggle)
docker logs -f rag__project-ingest-1

# 2. Transformação Silver
docker logs -f rag__project-transform_silver-1

# 3. Transformação Gold
docker logs -f rag__project-transform_gold-1

# 4. Treinamento dos modelos ML
docker logs -f rag__project-model_training-1

# 5. Download dos modelos Ollama (~2GB na primeira vez)
docker logs -f ollama-setup

# 6. Geração de embeddings dos dados (~10 min em CPU)
docker logs -f embedding-pipeline

# 7. API e Frontend sobem automaticamente após o pipeline
docker logs -f rag-api
```

> **Na primeira execução:** o pipeline completo leva entre 30-60 minutos dependendo da internet e do hardware. Nas execuções seguintes, os dados e modelos já estão nos volumes e o sistema sobe em 2-3 minutos.

## 5. Verificar saúde dos serviços

```bash
make health
```

Todos os serviços devem retornar OK:

```
=== MinIO ===      OK
=== PostgreSQL === OK
=== Milvus ===     OK
=== Ollama ===     OK
=== MLflow ===     OK
=== API ===        OK
=== Frontend ===   OK
```

## 6. Executar testes automáticos

```bash
make test
```

Valida automaticamente:
- Buckets Bronze/Silver/Gold no MinIO
- Coleções `csgo_rag` e `csgo_metadata` no Milvus
- Endpoints da API (`/`, `/models`, `/metadata`)
- Runs do MLflow no PostgreSQL

## 7. Acessar a interface

Abra no navegador: **http://localhost:7860**

Exemplos de perguntas sobre CS:GO:
- `Qual arma causa mais dano médio na cabeça?`
- `Quais armas causam mais dano pós plantação no site A?`
- `Qual lado causa mais dano com AK47?`

Exemplos de perguntas sobre o sistema:
- `Quais modelos foram utilizados no projeto?`
- `Qual a arquitetura do sistema?`
- `Como funciona o pipeline de dados?`

## 8. Acessar os serviços de monitoramento

**MinIO Console** — http://localhost:9001
- Usuário: `minioadmin` | Senha: `minioadmin`
- Bucket `csgodatalake` deve conter `bronze/`, `silver/` e `gold/`

**MLflow UI** — http://localhost:5000
- Experimento `csgo_damage_prediction`: XGBoost (R²=0.97) e RandomForest (R²=0.69)
- Experimento `rag-queries`: queries logadas em tempo real com métricas de latência e relevância

**API Swagger** — http://localhost:8000/docs
- Documentação interativa de todos os endpoints

## 9. Testar o RAG via terminal (alternativo ao Gradio)

```bash
pip install pymilvus==2.4.9 requests marshmallow==3.20.1 --break-system-packages
python3 scripts/rag_query.py
```

Digite `sair` para encerrar.

## 10. Comandos úteis do Makefile

```bash
make up       # Sobe todos os serviços
make down     # Para todos os serviços (mantém dados)
make reset    # Para e apaga todos os volumes ⚠️
make health   # Verifica saúde de todos os serviços
make test     # Executa testes automáticos
make ps       # Lista status dos containers
make logs s=api  # Logs de um serviço específico
make query    # RAG interativo no terminal
make rebuild  # Reconstrói índice vetorial no Milvus
make metadata # Reindexar metadados de governança
```

> ⚠️ **NUNCA rode `make reset` antes de uma apresentação.** Apaga todos os volumes e você precisará baixar tudo novamente.

## 11. Resetar tudo (opcional)

```bash
make reset
```

Apaga todos os volumes e dados. Após isso, volte ao passo 3.

---

## Arquitetura dos Dados

```
Kaggle CSV
    ↓
MinIO Bronze (raw)
    ↓
MinIO Silver (parquet limpo)
    ↓
MinIO Gold (agregado)
    ↓
Milvus csgo_rag (embeddings dos dados)
Milvus csgo_metadata (embeddings de governança)
    ↓
FastAPI (orquestra busca + geração)
    ↓
Ollama (LLM gera resposta)
    ↓
Gradio (interface do usuário)
    ↓
MLflow (rastreabilidade salva no PostgreSQL)
```

---

*Organização: StrikeMetrics Solutions | Instituição: Facens*