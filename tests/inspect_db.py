import sqlite3
import pandas as pd

db_path = r'c:\Users\USER\New folder\output\temp_preds.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 1. Total count
cursor.execute('SELECT COUNT(*) FROM preds')
total_in_db = cursor.fetchone()[0]

cursor.execute('SELECT COUNT(*) FROM preds WHERE matches IS NOT NULL AND matches != ""')
matches_count = cursor.fetchone()[0]

# 2. Sample records
cursor.execute('SELECT s1_id, length(candidates), length(matches) FROM preds LIMIT 5')
first_5 = cursor.fetchall()

cursor.execute('SELECT s1_id, length(candidates), length(matches) FROM preds ORDER BY rowid DESC LIMIT 5')
last_5 = cursor.fetchall()

print(f"Total entities in SQLite database: {total_in_db:,} / 1,732,544 ({total_in_db / 1732544 * 100:.2f}%)")
print(f"Entities with positive matches: {matches_count:,}")
print("First 5 records:", first_5)
print("Last 5 records:", last_5)

# 3. Check country coverage
data_dir = r"c:\Users\USER\Downloads\ML Challenge" if os.path.exists(r"c:\Users\USER\Downloads\ML Challenge") else r"G:\My Drive\ML Challenge"
s1_path = os.path.join(data_dir, 'test_source1.tsv')
df_s1 = pd.read_csv(s1_path, sep='\t', usecols=['entity_id', 'country'])

cursor.execute('SELECT s1_id FROM preds')
saved_ids = set(r[0] for r in cursor.fetchall())

for country, group in df_s1.groupby('country'):
    group_ids = set(group['entity_id'])
    covered = len(group_ids.intersection(saved_ids))
    print(f"Country {country:<8}: {covered:,} / {len(group_ids):,} ({covered / len(group_ids) * 100:.2f}%)")

conn.close()
