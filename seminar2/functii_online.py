import numpy as np
import pandas as pd

## Cerința 2 — Valori lipsă
# Implementați `nan_replace_df(tabel)` în `functii_online.py`
# (media pe coloană pentru numerice). Aplicați pe tabel;
# confirmați că nu rămân NaN (setul UCI nu are de obicei
# lipsuri — codul trebuie să fie **general**, ca la examen).

def nan_replace_df(tabel):
    for col in tabel.select_dtypes(include=[np.number]).columns:
        if tabel[col].isna().sum()>0:
            tabel[col]=tabel[col].fillna(tabel[col].mean())
    return tabel

def salvare_matrice(x, index_linie, index_coloana, fisier):
    df = pd.DataFrame(x, index=index_linie, columns=index_coloana)
    df.to_csv(fisier, index=True)
    return df