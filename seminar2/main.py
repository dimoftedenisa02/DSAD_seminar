import pandas as pd
import os

from functii_online import nan_replace_df, salvare_matrice
from scipy.spatial.distance import cdist


## Cerința 1 — Încărcare
# Încărcați `winequality-red.csv` cu separator corect.
# Afișați `shape`, primele 5 rânduri, tipurile coloanelor.

df=pd.read_csv("winequality-red.csv",sep=";")

print("--- SHAPE ---")
print(df.shape)

print("\n--- PRIMELE 5 RÂNDURI ---")
print(df.head())

print("\n--- TIPURILE COLOANELOR ---")
print(df.dtypes)


#CERINTA 2
df=nan_replace_df(df)

total_nan=df.isna().sum().sum()
print("--- CERINTA 2 ---")
print(f"Număr total de valori NaN rămase în setul de date: {total_nan}")


## Cerința 3 — Filtrare
# Salvați în `data_out/vinuri_calitate_min7.csv`
# toate vinurile cu **`quality >= 7`**. Păstrați toate
# coloanele.

df_min7=df[df['quality']>=7]

os.makedirs('data_out',exist_ok=True)

df_min7.to_csv('data_out/vinuri_calitate_min7.csv',index=False)

print('--- CERINȚA 3 ---')
print(f'Număr de vinuri de calitate >= 7 salvate: {df_min7.shape[0]}')

## Cerința 4 — Sortare (ca „Prezenta_sort”)
# Salvați în `data_out/top50_alcool.csv` cele **50** de
# vinuri cu cel mai mare **`alcohol`**, sortate descrescător.

top50_alcool=df.sort_values(by='alcohol',ascending=False).head(50)

top50_alcool.to_csv('data_out/top50_alcool.csv',index=False)

## Cerința 5 — Agregare (ca „Regiuni.csv”)
# Salvați în `data_out/medii_pe_calitate.csv`
# media coloanelor **`alcohol`** și **`pH`** pentru
# fiecare valoare a coloanei **`quality`** (un rând per
# notă de calitate).

medii_calitate=(df.groupby('quality')[['alcohol','pH']].mean().reset_index())

medii_calitate.to_csv('data_out/medii_pe_calitate.csv',index=False)

## Cerința 6 — Coloană derivată (ca procente)
# Adăugați coloana **`aciditate_totala`** =
# `fixed acidity` + `volatile acidity` + `citric acid`.
# Salvați în `data_out/vinuri_cu_aciditate.csv` doar
# coloanele:
# `quality`, `alcohol`, `aciditate_totala`, `pH`.

df['aciditate_totala']=(
    df['fixed acidity']+df['volatile acidity']+df['citric acid']
)

df_acid=df[['quality','alcohol','aciditate_totala','pH']]

df_acid.to_csv('data_out/vinuri_cu_aciditate.csv',index=False)


## Cerința 7 — Interpretare numerică

# Determinați **nota de calitate** (`quality`) pentru
# care media **`alcohol`** este maximă.
# Salvați un fișier `data_out/calitate_max_alcool.csv`
# cu o singură coloană `quality` și o singură valoare.

medie_alc_per_cal=df.groupby('quality')['alcohol'].mean()

max_quality=medie_alc_per_cal.idxmax()

df_max_q=pd.DataFrame({'quality':[max_quality]})

df_max_q.to_csv('data_out/calitate_max_alcool.csv', index=False)

## Cerința 8 — Corelații (pregătire ACP / preliminară)
# Pentru coloanele:
# `fixed acidity`, `volatile acidity`, `citric acid`,
# `alcohol`, `quality`
# calculați matricea de corelație și
# salvați **`data_out/R.csv`** (cu nume de rând/coloană =
# numele variabilelor).


cols_acp=[
'fixed acidity',
    'volatile acidity',
    'citric acid',
    'alcohol',
    'quality',
]

R=df[cols_acp].corr()

R.to_csv('data_out/R.csv',index=True)

## Cerința 9 — Standardizare (pregătire ACP)

# Creați **`alcohol_std`** = (alcohol − media) /
# deviația standard (populație sau eșantion —
# **documentați** în comentariu ce folosiți).
# Salvați `data_out/alcohol_standardizat.csv`
# cu coloanele `alcohol`, `alcohol_std`.


# Folosim deviatia standard de esantion (ddof=1, impartire la n-1),
# implicita în pandas. Pentru populatie ar fi ddof=0.

media_alc=df['alcohol'].mean()
std_alc=df['alcohol'].std(ddof=1)

df['alcohol_std']=(df['alcohol']-media_alc)/std_alc

df_std=df[['alcohol','alcohol_std']]

df_std.to_csv('data_out/alcohol_standardizat.csv', index=False)

## Cerința 10 — Mini matrice de distanțe

# Pe **primele 20 de vinuri** și pe coloanele standardizate
# `fixed acidity`, `volatile acidity`, `alcohol`
# (standardizați fiecare coloană),
# calculați matricea distanțelor **euclidiene**
# 20×20 (`scipy.spatial.distance.cdist` sau formula voastră).
# Salvați **`data_out/d_euclid_20.csv`**.

cols_dist=['fixed acidity','volatile acidity','alcohol']
sub20=df[cols_dist].head(20).copy()

for col in cols_dist:
    sub20[col]=(sub20[col]-sub20[col].mean())/sub20[col].std(ddof=1)

dist_matrix=cdist(sub20,sub20,metric='euclidean')

indici_vinuri=[f'Vin_{i}' for i in range (1,21)]
salvare_matrice(dist_matrix, indici_vinuri, indici_vinuri, 'data_out/d_euclid_20.csv')
