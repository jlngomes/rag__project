# Como Rodar o Projeto do Zero

## Pré-requisitos

- Docker e Docker Compose instalados
- Git instalado
- Conta no Kaggle (para download do dataset)
- Mínimo 16GB RAM e 20GB de espaço em disco

## 1. Clonar o repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd rag__project
```

## 2. Configurar credenciais do Kaggle

Crie o arquivo `.env` na raiz do projeto:

```bash
cp .env.example .env
```

Edite o `.env` e adicione seu token do Kaggle. Para obter o token: kaggle.com → Account → API → Create New Token.

## 3. Subir a infraestrutura

```bash
make up
```

Este comando inicia todos os serviços em ordem:
- MinIO (object storage)
- PostgreSQL (metadados e MLflow backend)
- Milvus + etcd (vector database)
- Ollama (LLM local)

## 4. Acompanhar o pipeline completo

O pipeline executa automaticamente na seguinte ordem:

```bash
# Acompanhar ingestão do dataset (demora ~7 minutos — download de 600MB)
docker logs -f rag__project-ingest-1

# Acompanhar transformação Silver
docker logs -f rag__project-transform_silver-1

# Acompanhar transformação Gold
docker logs -f rag__project-transform_gold-1

# Acompanhar treinamento dos modelos
docker logs -f rag__project-model_training-1

# Acompanhar geração de embeddings (demora ~10 minutos em CPU)
docker logs -f embedding-pipeline
```

## 5. Verificar saúde dos serviços

```bash
make health
```

Todos os serviços devem retornar OK.

## 6. Verificar dados no MinIO

Acesse: http://localhost:9001
Usuário: minioadmin
Senha: minioadmin

Bucket `csgodatalake` deve conter pastas `bronze/`, `silver/` e `gold/`.

## 7. Verificar experimentos no MLflow

Acesse: http://localhost:5000

Deve conter o experimento `csgo_damage_prediction` com 2 runs:
- XGBoost (R² ~0.97)
- RandomForest (R² ~0.69)

Os metadados são persistidos no PostgreSQL automaticamente.

## 8. Verificar arquivos na camada Gold (MinIO)

```bash
docker run --rm --network rag__project_default \
  --entrypoint sh minio/mc:latest -c \
  "mc alias set local http://minio:9000 minioadmin minioadmin --quiet && \
  mc ls local/csgodatalake/gold/"
```

## 9. Testar o RAG via terminal

Com todos os serviços rodando e o pipeline concluído, instale as dependências caso rode fora do Docker:

```bash
pip install pymilvus==2.4.9 requests --break-system-packages
```

Execute o script interativo:

```bash
python3 scripts/rag_query.py
```

Digite perguntas em português sobre estatísticas de CS:GO. Exemplos:
- `Qual arma causa mais dano médio na cabeça?`
- `Qual arma do lado TR causa mais dano?`
- `Quais armas causam mais dano pós plantação no site A?`

Digite `sair` para encerrar.

## 10. Resetar tudo (opcional)

```bash
make reset
```

Este comando apaga todos os volumes e dados. Após isso, volte ao passo 3.

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
Milvus (embeddings vetoriais)
    ↓
Ollama (respostas em linguagem natural)
    ↓
MLflow (rastreabilidade — salvo no PostgreSQL)
```