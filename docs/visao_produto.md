# Visão do Produto — StrikeMetrics Solutions

## Problema

Times e analistas de CS:GO precisam consultar manualmente planilhas e relatórios para entender performance de jogadores e armas. Esse processo é lento, não escala e exige conhecimento técnico para interpretar os dados.

## Solução

Plataforma RAG local que permite fazer perguntas em linguagem natural sobre estatísticas de combate, retornando respostas geradas por IA com rastreabilidade completa até os dados originais.

## Usuários

- Analistas de performance que fazem perguntas sobre armas e jogadores
- Engenheiros de dados que gerenciam o pipeline de ingestão
- Gestores que acompanham métricas no MLflow

## Restrições

- 100% local — sem cloud, sem envio de dados para terceiros
- Dataset: CS:GO Matchmaking Damage (Kaggle)
- Infraestrutura: Docker Compose, máquina local

## Métricas de Sucesso

- Pipeline Bronze→Silver→Gold executa sem erro
- Milvus indexa 1000+ chunks com embeddings válidos
- Sistema responde perguntas sobre CS:GO em menos de 60 segundos
