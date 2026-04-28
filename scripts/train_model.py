import os
import pandas as pd
from minio import Minio
from io import BytesIO
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

print("Iniciando pipeline de Machine Learning (Sprint 4)...")

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
# 2. CARREGAMENTO DOS DADOS (Camada Gold)
# ---------------------------------------------------------
print("Baixando dados multidimensionais da camada Gold...")
response = client.get_object(bucket, "gold/combat_context_stats.parquet")
df = pd.read_parquet(BytesIO(response.read()))

# ---------------------------------------------------------
# 3. PRÉ-PROCESSAMENTO (Preparação para o Modelo)
# ---------------------------------------------------------
print("Pré-processando os dados...")

# O modelo de ML só entende números. Vamos converter os textos em números (Label Encoding)
categorical_cols = ['wp', 'hitbox', 'att_side']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Definindo quem são as Features (X) e quem é o Alvo a ser previsto (y)
X = df[['wp', 'att_rank', 'hitbox', 'att_side']]
y = df['avg_hp_damage']

# Separando 80% dos dados para treinar e 20% para testar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 4. TREINAMENTO E RASTREAMENTO (MLflow)
# ---------------------------------------------------------
print("Iniciando treinamento com XGBoost...")

# Parâmetros de configuração do nosso modelo
xgb_params = {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "random_state": 42
}

# Inicia a gravação do experimento no MLflow
with mlflow.start_run():
    # 4.1 Registra os parâmetros usados
    mlflow.log_params(xgb_params)
    
    # 4.2 Cria e treina o modelo de fato
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)
    
    # 4.3 Faz as predições usando os 20% de dados de teste
    predictions = model.predict(X_test)
    
    # 4.4 Calcula o quão bom o modelo ficou
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5  # Eleva a meio para extrair a raiz quadrada
    r2 = r2_score(y_test, predictions)
    
    print(f"Resultados do Treino -> RMSE: {rmse:.4f} | R2 Score: {r2:.4f}")
    
    # 4.5 Salva as métricas de acerto no MLflow
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    
    # 4.6 Salva o próprio modelo treinado (o arquivo .pkl/binário) no MLflow
    mlflow.xgboost.log_model(model, "xgboost-model")
    
    print("Sucesso! Experimento e modelo salvos no MLflow.")