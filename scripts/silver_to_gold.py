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

print("Lendo dados da camada Silver (agregação incremental)...")

# Em vez de carregar tudo na memória de uma vez, processamos cada chunk
# incrementalmente mantendo apenas agregações parciais (menor footprint de RAM)
partial_combat = []
partial_bomb = []

objects = list(client.list_objects(bucket, prefix="silver/", recursive=True))

for obj in objects:
    if not obj.object_name.endswith(".parquet"):
        continue

    print(f"Processando: {obj.object_name}")
    response = client.get_object(bucket, obj.object_name)
    chunk = pd.read_parquet(BytesIO(response.read()))

    # Arquivos de granadas usam 'nade' ao invés de 'wp' — renomeia para unificar
    if 'nade' in chunk.columns and 'wp' not in chunk.columns:
        chunk = chunk.rename(columns={'nade': 'wp'})

    # Colunas obrigatórias para as agregações de dano — pula arquivos incompatíveis (ex: kills)
    required_cols = {'wp', 'att_rank', 'hitbox', 'att_side', 'hp_dmg', 'arm_dmg'}
    if not required_cols.issubset(chunk.columns):
        print(f"Pulando (colunas faltando): {obj.object_name}")
        del chunk
        continue

    # Agregação parcial para TABELA 1 (Combat)
    combat_chunk = chunk.groupby(['wp', 'att_rank', 'hitbox', 'att_side']).agg(
        total_hp_damage=('hp_dmg', 'sum'),
        total_armor_damage=('arm_dmg', 'sum'),
        avg_hp_damage_sum=('hp_dmg', 'sum'),
        total_hits=('hp_dmg', 'count')
    ).reset_index()
    partial_combat.append(combat_chunk)

    # Agregação parcial para TABELA 2 (Bomb)
    bomb_chunk = chunk[chunk['is_bomb_planted'] == True].groupby(['bomb_site', 'att_side', 'wp']).agg(
        damage_post_plant=('hp_dmg', 'sum'),
        avg_damage_post_plant_sum=('hp_dmg', 'sum'),
        hits_post_plant=('hp_dmg', 'count')
    ).reset_index()
    partial_bomb.append(bomb_chunk)

    # Libera memória do chunk logo após processar
    del chunk, combat_chunk, bomb_chunk

print("Consolidando agregações parciais...")

# --- REGRAS DE NEGÓCIO (Camada Gold - Foco em RAG) ---

# TABELA 1: Estatísticas de Combate Multidimensional
df_combat = pd.concat(partial_combat, ignore_index=True)
del partial_combat

df_gold_combat = df_combat.groupby(['wp', 'att_rank', 'hitbox', 'att_side']).agg(
    total_hp_damage=('total_hp_damage', 'sum'),
    total_armor_damage=('total_armor_damage', 'sum'),
    avg_hp_damage=('avg_hp_damage_sum', 'sum'),
    total_hits=('total_hits', 'sum')
).reset_index()

# Recalcula a média corretamente: soma_dos_danos / total_de_acertos
df_gold_combat['avg_hp_damage'] = df_gold_combat['avg_hp_damage'] / df_gold_combat['total_hits']

# Filtro de Relevância (Ruído):
# Remove combinações que aconteceram menos de 50 vezes para evitar que a IA
# aprenda com pontos fora da curva (outliers estatísticos)
df_gold_combat = df_gold_combat[df_gold_combat['total_hits'] >= 50]
df_gold_combat = df_gold_combat.sort_values(by='total_hits', ascending=False)

# TABELA 2: Análise de Pressão com Bomba Plantada Multidimensional
df_bomb = pd.concat(partial_bomb, ignore_index=True)
del partial_bomb

df_gold_bomb = df_bomb.groupby(['bomb_site', 'att_side', 'wp']).agg(
    damage_post_plant=('damage_post_plant', 'sum'),
    avg_damage_post_plant=('avg_damage_post_plant_sum', 'sum'),
    hits_post_plant=('hits_post_plant', 'sum')
).reset_index()

# Recalcula a média corretamente: soma_dos_danos / total_de_acertos
df_gold_bomb['avg_damage_post_plant'] = df_gold_bomb['avg_damage_post_plant'] / df_gold_bomb['hits_post_plant']

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
