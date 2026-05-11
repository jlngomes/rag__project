.PHONY: up down reset logs health ps

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

health:
	@echo "=== MinIO ===" && curl -sf http://localhost:9000/minio/health/live && echo " OK" || echo " FAIL"
	@echo "=== PostgreSQL ===" && docker exec postgres pg_isready -U postgres && echo " OK" || echo " FAIL"
	@echo "=== Milvus ===" && curl -sf http://localhost:9091/healthz && echo " OK" || echo " FAIL"
	@echo "=== Ollama ===" && curl -sf http://localhost:11434/api/tags > /dev/null && echo " OK" || echo " FAIL"
	@echo "=== MLflow ===" && curl -sf http://localhost:5000/health && echo " OK" || echo " FAIL"