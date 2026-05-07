from minio import Minio

client = Minio(
    "minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

buckets = ["csgodatalake", "mlflow-artifacts"]

for bucket in buckets:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        print(f"Bucket criado: {bucket}")
    else:
        print(f"Bucket já existe: {bucket}")
