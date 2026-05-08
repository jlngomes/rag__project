# StrikeMetrics Solutions — Plataforma RAG Enterprise para CS:GO

Plataforma de análise de performance de CS:GO baseada em RAG (Retrieval-Augmented Generation) com governança de dados local, desenvolvida como projeto acadêmico na Facens.

## Objetivo

Permitir que analistas façam perguntas em linguagem natural sobre estatísticas de combate do CS:GO, recebendo respostas geradas por IA com rastreabilidade completa dos dados utilizados.

## Tecnologias

| Camada | Tecnologia |
|--------|-----------|
| Storage | MinIO (Bronze/Silver/Gold) |
| Banco Relacional | PostgreSQL |
| Vector Database | Milvus |
| LLM Local | Ollama (nomic-embed-text) |
| MLOps | MLflow |
| Orquestração | Docker Compose |
| Metodologia | Scrum (12 sprints) |

## Arquitetura Medallion

- **Bronze**: CSVs originais do Kaggle (dados brutos de partidas CS:GO)
- **Silver**: Parquet limpos e padronizados
- **Gold**: Agregações multidimensionais prontas para RAG

## Como subir o projeto

```bash
make up
```

## Verificar saúde dos serviços

```bash
make health
```

## Grupo

- Gustavo Valadares Fukui - 234719
- Jean Luca Novaes Gomes - 236999
- Samuel Augusto Magalhães Alexandre - 237334 (PO)
- Sebastião Gonçalves da Cunha Neto - 236106
- Gustavo Aguiar Brandão - 210349
- Gabriel Henrique Cuchera - 211614
- Eduardo de Souza Monteiro - 212123
- Carlos Kainã de Oliveira Quirino Ramalho - 190378
- André Vinícius Siqueira - 236811

**Instituição:** Facens  
**Disciplina:** Engenharia — Análise de Dados de eSports