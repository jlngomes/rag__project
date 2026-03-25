import pandas as pd
from minio import Minio
from io import BytesIO

client = Minio(
    "minio:9000",
    access_key="minio",
    secret_key="minio123",
    secure=False
)

bucket = "csgo.datalake"

# baixar bronze
response = client.get_object(
    bucket,
    "bronze/damage.csv"
)

df = pd.read_csv(response)

# tratamento simples
df = df.dropna()

df.columns = df.columns.str.lower()

# salvar silver
buffer = BytesIO()
df.to_parquet(buffer, index=False)
buffer.seek(0)

client.put_object(
    bucket,
    "silver/damage_clean.parquet",
    buffer,
    length=buffer.getbuffer().nbytes
)

print("Silver dataset created")
