import os
import pandas as pd
from minio import Minio
from io import BytesIO

minio_endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")

client = Minio(
    minio_endpoint, 
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

bucket = "csgodatalake"

print("Lendo dados da camada Silver...")
objects = client.list_objects(bucket, prefix="silver/", recursive=True)

dfs = []

for obj in objects:
    if obj.object_name.endswith(".parquet"):
        print(f"Lendo: {obj.object_name}")
        response = client.get_object(bucket, obj.object_name)
        dfs.append(pd.read_parquet(BytesIO(response.read())))

# Junta todas as partes em um único DataFrame
df_silver = pd.concat(dfs, ignore_index=True)

print("Gerando tabelas unificadas para a camada Gold...")

# --- REGRAS DE NEGÓCIO (Camada Gold - Foco em RAG) ---

# TABELA 1: Estatísticas de Combate Multidimensional
# Agrupa as principais dimensões: Arma, Patente do Atacante, Local do Acerto e Lado (TR/CT)
df_gold_combat = df_silver.groupby(['wp', 'att_rank', 'hitbox', 'att_side']).agg(
    total_hp_damage=('hp_dmg', 'sum'),
    total_armor_damage=('arm_dmg', 'sum'),
    avg_hp_damage=('hp_dmg', 'mean'),
    total_hits=('hp_dmg', 'count')
).reset_index()

# Filtro de Relevância (Ruído): 
# Remove combinações que aconteceram menos de 50 vezes para evitar que a IA 
# aprenda com pontos fora da curva (outliers estatísticos)
df_gold_combat = df_gold_combat[df_gold_combat['total_hits'] >= 50]
df_gold_combat = df_gold_combat.sort_values(by='total_hits', ascending=False)


# TABELA 2: Análise de Pressão com Bomba Plantada Multidimensional
# Filtra apenas momentos pós-plant e agrupa por bomb site, lado e arma
df_gold_bomb = df_silver[df_silver['is_bomb_planted'] == True].groupby(['bomb_site', 'att_side', 'wp']).agg(
    damage_post_plant=('hp_dmg', 'sum'),
    avg_damage_post_plant=('hp_dmg', 'mean'),
    hits_post_plant=('hp_dmg', 'count')
).reset_index()

# Filtro de Relevância para a bomba 
df_gold_bomb = df_gold_bomb[df_gold_bomb['hits_post_plant'] >= 10]
df_gold_bomb = df_gold_bomb.sort_values(by='hits_post_plant', ascending=False)


# --- SALVAMENTO NO MINIO ---
def save_gold_table(df, filename):
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    
    client.put_object(
        bucket,
        f"gold/{filename}",
        buffer,
        length=buffer.getbuffer().nbytes
    )
    print(f"Tabela salva com sucesso: gold/{filename}")

print("Salvando arquivos no MinIO...")
save_gold_table(df_gold_combat, "combat_context_stats.parquet")
save_gold_table(df_gold_bomb, "bomb_context_stats.parquet")

print("Pipeline concluído! Camada Gold multidimensional gerada.")