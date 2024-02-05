import pandas as pd

# Excel dosyasını oku
df = pd.read_excel('veri.xlsx')

# data_v kolonuna göre duplikatları kaldır
unique_values = set()
duplicates_indices = []

for index, row in df.iterrows():
    data_v_value = row['date_v']
    if data_v_value in unique_values:
        duplicates_indices.append(index)
    else:
        unique_values.add(data_v_value)

# Duplikat satırları sil
df = df.drop(index=duplicates_indices)

# Sonucu yeni bir Excel dosyasına yaz
df.to_excel('yeni_data.xlsx', index=False)
