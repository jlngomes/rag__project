////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////							Governança de Dados e Auditoria - StrikeMetrics Solutions (Sprint 3)
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Este documento estabelece as diretrizes de governança, rastreabilidade e integridade para o pipeline de análise de performance de eSports (CS:GO) da StrikeMetrics Solutions, seguindo a metodologia Scrum da Facens.

1. Arquitetura de Dados 

Para garantir a precisão nas métricas de eficiência de combate, os dados de dano seguem o fluxo estruturado no MinIO:Camada Bronze (Raw): Armazenamento dos arquivos brutos do Kaggle (ex: csgo_damage.csv) contendo registros de cada hit, arma utilizada e localização do dano.Camada Silver (Trusted): Dados limpos e padronizados. Nesta etapa, normalizamos os nomes das armas, removemos registros inconsistentes e adicionamos metadados de tempo de ingestão.Camada Gold (Refined): Dados agregados para análise de performance (ex: média de dano por round, eficiência de custo por arma e ADR - Average Damage per Round), prontos para busca semântica no Milvus.

2. Rastreabilidade e Auditoria 

O sistema da StrikeMetrics Solutions foi projetado para garantir total transparência sobre os insights gerados pela IA:

	2.1. Origem e Ciclo de VidaLinhagem de Dano (Lineage): Cada métrica de performance exibida ao usuário final pode ser rastreada até o arquivo original na camada Bronze, garantindo que 	estatísticas de "clutch" ou "entry damage" sejam verídicas.Versionamento de Dataset: O MinIO mantém versões históricas dos datasets, permitindo comparar a performance de jogadores em 	diferentes versões (patches) do jogo.

	2.2. Governança de Inteligência ArtificialMLFlow: Registra quais versões dos dados de dano foram utilizadas para calibrar os modelos de busca semântica.Rastreio de Evidências: Ao 	responder sobre o melhor jogador ou arma, o sistema RAG identifica as instâncias reais (rounds específicos) que serviram de base para a conclusão da IA.

3. Conformidade Técnica 
A infraestrutura utilizada para sustentar a governança da StrikeMetrics Solutions:

Ferramenta		Função na Governança StrikeMetrics

MinIO			Versionamento dos registros de dano e armazenamento das camadas
Medallion.PostgreSQL	Armazenamento de metadados das partidas e logs de auditoria de acesso.
MLFlow			Rastreabilidade do ciclo de vida dos modelos de análise de performance.
Milvus			Recuperação íntegra de contextos históricos de combate para o LLM.


4. Perguntas de Auditoria Atendidas

O sistema da StrikeMetrics Solutions responde prontamente:

- Qual versão do dataset de dano foi usada para gerar este relatório de performance?
- Quais rounds específicos comprovam que a arma X foi mais eficiente que a Y neste mapa?
- Por que a IA classificou este jogador como um talento de alto impacto em rounds econômicos?

Organização: StrikeMetrics Solutions
Instituição: Facens
Projeto: Engenharia - Análise de Dados de eSports (CS:GO)