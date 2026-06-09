.PHONY: up down reset logs health ps test query rebuild metadata

# ─── Infraestrutura ───────────────────────────────────────────────────────────

up:
	docker-compose up -d

down:
	docker-compose down

reset:
	docker-compose down -v

ps:
	docker-compose ps

logs:
	docker-compose logs -f $(s)

# ─── Health Check ─────────────────────────────────────────────────────────────

health:
	@echo "=== MinIO ===" && curl -sf http://localhost:9000/minio/health/live && echo " OK" || echo " FAIL"
	@echo "=== PostgreSQL ===" && docker exec postgres pg_isready -U postgres && echo " OK" || echo " FAIL"
	@echo "=== Milvus ===" && curl -sf http://localhost:9091/healthz && echo " OK" || echo " FAIL"
	@echo "=== Ollama ===" && curl -sf http://localhost:11434/api/tags > /dev/null && echo " OK" || echo " FAIL"
	@echo "=== MLflow ===" && curl -sf http://localhost:5000/health && echo " OK" || echo " FAIL"
	@echo "=== API ===" && curl -sf http://localhost:8000/ > /dev/null && echo " OK" || echo " FAIL"
	@echo "=== Frontend ===" && curl -sf http://localhost:7860/ > /dev/null && echo " OK" || echo " FAIL"

# ─── Testes ───────────────────────────────────────────────────────────────────

test:
	@echo "=== Teste 1: MinIO buckets ==="
	@docker run --rm --network rag__project_default \
		--entrypoint sh minio/mc:latest -c \
		"mc alias set local http://minio:9000 minioadmin minioadmin --quiet && \
		mc ls local/csgodatalake/bronze/ > /dev/null && echo 'bronze OK' || echo 'bronze FAIL' && \
		mc ls local/csgodatalake/silver/ > /dev/null && echo 'silver OK' || echo 'silver FAIL' && \
		mc ls local/csgodatalake/gold/ > /dev/null && echo 'gold OK' || echo 'gold FAIL'"
	@echo ""
	@echo "=== Teste 2: Milvus collections ==="
	@docker exec rag-api python3 -c "\
from pymilvus import MilvusClient; \
c = MilvusClient(uri='http://milvus:19530'); \
cols = c.list_collections(); \
print('csgo_rag OK' if 'csgo_rag' in cols else 'csgo_rag FAIL'); \
print('csgo_metadata OK' if 'csgo_metadata' in cols else 'csgo_metadata FAIL')"
	@echo ""
	@echo "=== Teste 3: API endpoints ==="
	@curl -sf http://localhost:8000/ > /dev/null && echo "GET / OK" || echo "GET / FAIL"
	@curl -sf http://localhost:8000/models > /dev/null && echo "GET /models OK" || echo "GET /models FAIL"
	@curl -sf http://localhost:8000/metadata > /dev/null && echo "GET /metadata OK" || echo "GET /metadata FAIL"
	@echo ""
	@echo "=== Teste 4: PostgreSQL tabelas MLflow ==="
	@docker exec postgres psql -U postgres -d db -c \
		"SELECT COUNT(*) as total_runs FROM runs;" 2>/dev/null | grep -E "[0-9]+" | head -1 | \
		xargs -I{} echo "MLflow runs no PostgreSQL: {}"
	@echo ""
	@echo "=== Todos os testes concluídos ==="

# ─── RAG ──────────────────────────────────────────────────────────────────────

query:
	@python3 scripts/rag_query.py

rebuild:
	@echo "Reconstruindo índice vetorial..."
	@curl -sf -X POST http://localhost:8000/rebuild-index \
		-H "Content-Type: application/json" | python3 -m json.tool

metadata:
	@echo "Reindexando metadados de governança..."
	@docker run --rm --network rag__project_default \
		-v "$(PWD)/scripts:/app/scripts" \
		-e MILVUS_URI=http://milvus:19530 \
		-e OLLAMA_BASE_URL=http://ollama:11434 \
		rag__project-embedding_pipeline \
		python scripts/generate_metadata_embeddings.py
