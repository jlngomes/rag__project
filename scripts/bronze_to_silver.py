import pandas as pd
from minio import Minio
from io import BytesIO

client = Minio(
    "minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

bucket = "csgodatalake"

# listar arquivos bronze
objects = client.list_objects(bucket, prefix="bronze/", recursive=True)

for obj in objects:

    if not obj.object_name.endswith(".csv"):
        continue

    print("Processing:", obj.object_name)

    response = client.get_object(bucket, obj.object_name)

    part = 0

    for chunk in pd.read_csv(BytesIO(response.read()), chunksize=100000):

        # limpeza simples
        chunk = chunk.dropna()
        chunk.columns = chunk.columns.str.lower()

        # salvar parquet
        buffer = BytesIO()
        chunk.to_parquet(buffer, index=False)
        buffer.seek(0)

        silver_name = obj.object_name.replace(
            "bronze/", "silver/"
        ).replace(".csv", f"_part{part}.parquet")

        client.put_object(
            bucket,
            silver_name,
            buffer,
            length=buffer.getbuffer().nbytes
        )

        part += 1

print("Silver datasets created")
