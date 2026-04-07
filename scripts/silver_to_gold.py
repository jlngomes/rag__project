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

print("Gerando tabelas para a camada Gold...")


# REGRAS DE NEGÓCIO (Camada Gold)
# TABELA 1: Estatísticas de Dano por Arma
# Agrupa e soma os danos de vida e colete
df_gold_weapons = df_silver.groupby('wp').agg(
    total_hp_damage=('hp_dmg', 'sum'),
    total_armor_damage=('arm_dmg', 'sum'),
    total_shots_hit=('hp_dmg', 'count')
).reset_index()

# Ordena pelas armas mais destrutivas
df_gold_weapons = df_gold_weapons.sort_values(by='total_hp_damage', ascending=False)


# TABELA 2: Letalidade por Região do Corpo (Hitbox)
# Agrupa para ver onde os jogadores mais acertam tiros
df_gold_hitbox = df_silver.groupby('hitbox').agg(
    total_hp_damage=('hp_dmg', 'sum'),
    avg_damage_per_hit=('hp_dmg', 'mean'),
    total_hits=('hp_dmg', 'count')
).reset_index()

# Ordena pelos locais mais atingidos
df_gold_hitbox = df_gold_hitbox.sort_values(by='total_hits', ascending=False)

# --- TABELA 3: Desempenho CT vs T ---
df_gold_side = df_silver.groupby('att_side').agg(
    total_hp_damage=('hp_dmg', 'sum'),
    avg_hp_damage=('hp_dmg', 'mean'),
    count_hits=('hp_dmg', 'count')
).reset_index()

# --- TABELA 4: Letalidade por Rank (Patente) ---
df_gold_rank = df_silver.groupby('att_rank').agg(
    avg_damage=('hp_dmg', 'mean'),
    total_hits=('hp_dmg', 'count')
).reset_index().sort_values(by='att_rank')

# --- TABELA 5: Análise de Bomb Site ---
# Apenas quando a bomba plantada
df_gold_bomb = df_silver[df_silver['is_bomb_planted'] == True].groupby('bomb_site').agg(
    damage_post_plant=('hp_dmg', 'sum'),
    hits_post_plant=('hp_dmg', 'count')
).reset_index()


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
save_gold_table(df_gold_weapons, "weapon_stats.parquet")
save_gold_table(df_gold_hitbox, "hitbox_stats.parquet")
save_gold_table(df_gold_side, "side_performance.parquet")
save_gold_table(df_gold_rank, "rank_lethality.parquet")
save_gold_table(df_gold_bomb, "bomb_site_pressure.parquet")

print("Pipeline concluído! Camada Gold gerada.")