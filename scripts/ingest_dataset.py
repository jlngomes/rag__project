import kagglehub
import pandas as pd
from minio import Minio

# baixar dataset
dataset_path = kagglehub.dataset_download(
    "skihikingkevin/csgo-matchmaking-damage"
)

print("Dataset downloaded to:", dataset_path)

# conectar no MinIO
client = Minio(
    "minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

bucket = "csgo.datalake"

if not client.bucket_exists(bucket):
    client.make_bucket(bucket)

# enviar arquivos
import os

for file in os.listdir(dataset_path):
    path = os.path.join(dataset_path, file)

    if file.endswith(".csv"):
        client.fput_object(
            bucket,
            f"bronze/{file}",
            path
        )

print("Upload concluído")
