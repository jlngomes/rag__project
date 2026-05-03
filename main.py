from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_core.documents import Document

# ==========================================
# REQUISITO 1: INTEGRAÇÃO COM OLLAMA
# ==========================================
print("1. Conectando ao Ollama...")
# Vamos usar o modelo nomic-embed-text, que é especialista em gerar embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434" # Porta padrão onde o Ollama roda no seu PC
)

# ==========================================
# PREPARAÇÃO DOS DADOS (CHUNKING)
# ==========================================
# Aqui é onde você puxaria os dados do seu MinIO (Camada Gold). 
# Vou usar uma lista de exemplo para podermos testar agora:
textos_exemplo = [
    "O RAG (Retrieval-Augmented Generation) é uma técnica que melhora a geração de respostas da IA.",
    "Milvus é um banco de dados vetorial de alto desempenho focado em IA.",
    "Ollama é uma ferramenta fantástica que permite rodar modelos de IA localmente na sua máquina.",
    "O Palmeiras é um time de futebol do Brasil."
]

# Transformando os textos no formato que o LangChain entende ("Documents")
documentos = [Document(page_content=t) for t in textos_exemplo]

# Nós precisamos quebrar textos muito grandes em pedaços menores (chunks).
# Isso melhora a qualidade da busca vetorial.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documentos)

# ==========================================
# REQUISITO 2 e 3: GERAÇÃO DE EMBEDDINGS E INDEXAÇÃO VETORIAL
# ==========================================
print("2. Gerando Embeddings e Indexando no Milvus (Isso pode levar alguns segundos)...")

# A mágica acontece aqui: O Langchain pega os chunks, manda pro Ollama virar Embedding, 
# e salva direto no Milvus rodando no seu Docker na porta 19530!
vetor_db = Milvus.from_documents(
    documents=chunks,
    embedding=embeddings,
    connection_args={"uri": "http://localhost:19530"},
    collection_name="meu_projeto_rag",
    drop_old=True # Apaga a coleção antiga se você rodar o script de novo (ótimo para testes)
)
print("-> Indexação concluída com sucesso no Milvus!")

# ==========================================
# BÔNUS: TESTANDO SE DEU TUDO CERTO (BUSCA VETORIAL)
# ==========================================
pergunta = "Qual ferramenta eu uso para banco de dados de IA?"
print(f"\nFazendo uma busca vetorial (por significado) para a pergunta: '{pergunta}'")

# Vamos buscar os 2 resultados que têm o significado mais próximo da pergunta
resultados = vetor_db.similarity_search(pergunta, k=2)

for i, res in enumerate(resultados):
    print(f"\nResultado {i+1}: {res.page_content}")