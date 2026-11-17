import os
import numpy as np
import pandas as pd


def medie_ponderata(grup: pd.DataFrame, coloane, coloana_pondere):
    # media ponderata: suma(x_i * p_i) / suma(p_i)
    return pd.Series(
        np.average(grup[coloane], axis=0, weights=grup[coloana_pondere]),
        index=coloane
    )


os.makedirs("data_out", exist_ok=True)

tabel_ind = pd.read_csv("data_in/IndustriaAlimentara.csv", index_col=0)
tabel_pop = pd.read_csv("data_in/PopulatieLocalitati.csv", index_col=0)
tabel_judete = pd.read_csv("data_in/Coduri_Judete.csv", index_col=0)

activitati = list(tabel_ind.columns[1:])

# 1. Să se salveze în fișierul Cerinta1.csv datele privind numărul de angajați
# pe activitățile din industria alimentară, pentru localitățile în care numărul de angajați
# este mai mare decât 0. Fișierul Cerinta1.csv va avea aceleași coloane ca fișierul
# IndustriaAlimentara.csv.
total_angajati = tabel_ind[activitati].sum(axis=1)
c1 = tabel_ind[total_angajati > 0]
c1.to_csv("data_out/Cerinta1.csv")

# 2. Să se calculeze și să se salveze în fișierul Cerinta2.csv, procentul de angajați
# pe fiecare activitate la nivel de localitate.
c2 = c1[activitati].div(c1[activitati].sum(axis=1), axis=0) * 100
c2.insert(0, "Localitate", c1["Localitate"])
c2.round(2).to_csv("data_out/Cerinta2.csv")

# 3. Să se salveze în fișierul Cerinta3.csv județele sortate descrescător după
# ponderea angajaților din industria alimentară în populația totală.
tabel = tabel_ind[activitati].join(tabel_pop[["Judet", "Populatie"]])
tabel_judet = tabel.groupby("Judet").sum()

c3 = pd.DataFrame({
    "Pondere": tabel_judet[activitati].sum(axis=1) / tabel_judet["Populatie"]
})
c3 = c3.sort_values(by="Pondere", ascending=False)
c3.to_csv("data_out/Cerinta3.csv")

# 4. Să se salveze în fișierul Cerinta4.csv, activitatea cu cei mai
# mulți angajați la nivel de județ. Se va afișa indicativul de județ și activitatea dominantă.
c4 = pd.DataFrame({
    "Activitate": tabel_judet[activitati].idxmax(axis=1)
})
c4.to_csv("data_out/Cerinta4.csv")

# 5. Să se salveze în fișierul Cerinta5.csv numărul mediu de angajați pe
# fiecare activitate la nivel de regiune.
# Se va calcula media ponderată, folosind ca pondere populația localităților.
tabel_regiune = tabel.merge(tabel_judete[["Regiune"]], left_on="Judet", right_index=True)
c5 = tabel_regiune.groupby("Regiune")[activitati + ["Populatie"]].apply(
    medie_ponderata, coloane=activitati, coloana_pondere="Populatie"
)
c5.round(3).to_csv("data_out/Cerinta5.csv")

# 6. Să se calculeze și să se afișeze în fișierul Cerinta6.csv indicii de localizare
# la nivel de județ pentru fiecare activitate. Un indice de localizare se calculează
# ca raport între ponderea județului i în activitatea j la nivel național și ponderea
# populației din județul i în totalul populației.
pondere_activitate = tabel_judet[activitati] / tabel_judet[activitati].sum()
pondere_populatie = tabel_judet["Populatie"] / tabel_judet["Populatie"].sum()
c6 = pondere_activitate.div(pondere_populatie, axis=0)
c6.round(3).to_csv("data_out/Cerinta6.csv")