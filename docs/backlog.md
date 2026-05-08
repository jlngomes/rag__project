# Backlog — StrikeMetrics Solutions

## Definition of Done (DoD)
- Código funciona dentro do Docker sem erro
- Dados persistem entre reinicializações
- Funcionalidade demonstrável via terminal ou interface

---

## ÉPICO 1: Infraestrutura

**US01** — Como DevOps, quero subir todos os serviços com um único comando para agilizar o desenvolvimento.
- Critério: `make up` sobe todos os containers sem erro

**US02** — Como DevOps, quero que os dados persistam entre reinicializações para não perder o trabalho.
- Critério: `make down && make up` não apaga dados do MinIO, PostgreSQL e Milvus

**US03** — Como DevOps, quero verificar a saúde de todos os serviços com um comando.
- Critério: `make health` retorna OK para todos os serviços

---

## ÉPICO 2: Governança e Ingestão de Dados

**US04** — Como engenheiro de dados, quero ingerir CSVs brutos do Kaggle na camada Bronze para preservar os dados originais.
- Critério: Arquivos CSV disponíveis em `csgodatalake/bronze/` no MinIO

**US05** — Como engenheiro de dados, quero transformar os dados Bronze em Silver para garantir qualidade e padronização.
- Critério: Parquet limpos em `csgodatalake/silver/` sem valores nulos

**US06** — Como engenheiro de dados, quero agregar os dados Silver em Gold para otimizar as consultas RAG.
- Critério: `combat_context_stats.parquet` e `bomb_context_stats.parquet` em `csgodatalake/gold/`

---

## ÉPICO 3: Machine Learning e MLOps

**US07** — Como cientista de dados, quero treinar um modelo preditivo de dano para validar a qualidade dos dados Gold.
- Critério: Modelo XGBoost com R² > 0.90 registrado no MLflow

**US08** — Como gestor, quero visualizar métricas e experimentos no MLflow para acompanhar a evolução dos modelos.
- Critério: MLflow UI acessível em `http://localhost:5000` com experimentos logados

---

## ÉPICO 4: Pipeline RAG

**US09** — Como engenheiro de IA, quero gerar embeddings dos dados Gold para permitir busca semântica.
- Critério: 1165+ chunks indexados na coleção `csgo_rag` do Milvus

**US10** — Como analista, quero fazer perguntas em linguagem natural sobre CS:GO e receber respostas baseadas nos dados.
- Critério: Sistema retorna resposta relevante com chunks recuperados do Milvus

---

## Priorização (MoSCoW)

| US | Descrição | Prioridade |
|----|-----------|-----------|
| US01 | Subir serviços com um comando | Must Have |
| US04 | Ingestão Bronze | Must Have |
| US05 | Transformação Silver | Must Have |
| US06 | Agregação Gold | Must Have |
| US09 | Embeddings Milvus | Must Have |
| US10 | Query RAG | Must Have |
| US07 | Modelo XGBoost | Should Have |
| US08 | MLflow UI | Should Have |
| US02 | Persistência de dados | Should Have |
| US03 | Health check | Could Have |