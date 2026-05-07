import os
import pandas as pd
from minio import Minio
from io import BytesIO
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

print("Iniciando pipeline de Machine Learning - Bomb Context")

# ---------------------------------------------------------
# 1. CONFIGURAÇÕES DE AMBIENTE E CONEXÕES
# ---------------------------------------------------------
minio_endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
minio_user = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
minio_pass = os.getenv("MINIO_SECRET_KEY", "minioadmin")
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Conexão MinIO
client = Minio(
    minio_endpoint,
    access_key=minio_user,
    secret_key=minio_pass,
    secure=False
)
bucket = "csgodatalake"

# Conexão MLflow
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("csgo_damage_prediction")

# ---------------------------------------------------------
# 2. CARREGAMENTO DOS DADOS (Camada Gold - Bomb Context)
# ---------------------------------------------------------
print("Baixando dados de bomb context da camada Gold...")
response = client.get_object(bucket, "gold/bomb_context_stats.parquet")
df = pd.read_parquet(BytesIO(response.read()))

# ---------------------------------------------------------
# 3. PRÉ-PROCESSAMENTO (Preparação para o Modelo)
# ---------------------------------------------------------
print("Pré-processando os dados...")

# Label Encoding para colunas categóricas
categorical_cols = ['bomb_site', 'att_side', 'wp']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features (X): dimensões do contexto de bomba + volume de acertos
# Alvo (y): dano médio causado após plant do C4
X = df[['bomb_site', 'att_side', 'wp', 'hits_post_plant']]
y = df['avg_damage_post_plant']

# Separando 80% dos dados para treinar e 20% para testar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 4. TREINAMENTO E RASTREAMENTO (MLflow)
# ---------------------------------------------------------
print("Iniciando treinamento com RandomForestRegressor...")

# Parâmetros de configuração do modelo
rf_params = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42
}

# Inicia a gravação do experimento no MLflow
with mlflow.start_run():
    # 4.1 Registra os parâmetros usados
    mlflow.log_params(rf_params)

    # 4.2 Cria e treina o modelo de fato
    model = RandomForestRegressor(**rf_params)
    model.fit(X_train, y_train)

    # 4.3 Faz as predições usando os 20% de dados de teste
    predictions = model.predict(X_test)

    # 4.4 Calcula o quão bom o modelo ficou
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"Resultados do Treino -> RMSE: {rmse:.4f} | R2 Score: {r2:.4f}")

    # 4.5 Salva as métricas de acerto no MLflow
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    # 4.6 Salva o próprio modelo treinado no MLflow
    mlflow.sklearn.log_model(model, "sklearn-randomforest-model")

print("Sucesso! Experimento e modelo de bomb context salvos no MLflow.")
