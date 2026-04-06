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

objects = client.list_objects(bucket, prefix="silver/", recursive=True)

for obj in objects:
    
    if not obj.object_name.endswith(".parquet"):
        continue

    print("Processing:", obj.object_name)

    response = client.get_object(bucket, obj.object_name)

    df = pd.read_parquet(BytesIO(response.read()))

    print(df.columns)

    df.head()
