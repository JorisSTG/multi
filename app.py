import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- STYLE sombre pour se fondre avec le thème Streamlit ----
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "axes.edgecolor": "#FFFFFF",
    "axes.labelcolor": "#FFFFFF",
    "xtick.color": "#DDDDDD",
    "ytick.color": "#DDDDDD",
    "text.color": "#FFFFFF",
})

st.title("Comparaison multisource")
st.markdown("""
L’objectif de cette application est d’évaluer la précision et la cohérence entre **N jeux de données** (température uniquement) à des fins de simulations STD.
""")

# -------- Paramètres généraux --------
heures_par_mois = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
mois_noms = ["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Août","Septembre","Octobre","Novembre","Décembre"]
couleurs = ["goldenrod", "lightgray", "navy", "green", "darkmagenta", "peru", "silver", "orange"]

# -------- Upload --------
uploaded_files = st.file_uploader("Déposer les fichiers CSV (colonne unique T°C) :", type=["csv"], accept_multiple_files=True)

if len(uploaded_files) >= 2:

    # -------- Lecture --------
    data, file_names = {}, {}
    for i, file in enumerate(uploaded_files):
        key = f"source_{i+1}"
        file_names[key] = file.name.replace(".csv", "")
        data[key] = pd.read_csv(file).iloc[:,0].values

    # -------- PARAMÈTRES BARRES --------
    n_sources = len(data)
    group_width = 0.8
    group_margin = 0.1
    bar_width = group_width / n_sources

    def bar_positions(x, i):
        return x - group_width/2 + bar_width/2 + i*bar_width

    # -------- Fonctions --------
    def daily_stats_from_hourly(hourly):
        n = len(hourly)//24
        arr = np.array(hourly[:n*24]).reshape(n,24)
        return arr.min(1), arr.mean(1), arr.max(1)

    def nombre_jours_vague(T):
        T = np.array(T)
        jours = np.zeros(len(T), bool)
        jours[T >= 25.3] = True
        i = 0
        while i < len(T):
            if i+2 < len(T) and np.all(T[i:i+3] >= 23.4):
                j = i+3
                while j < len(T) and T[j] >= 23.4: j+=1
                jours[i:j] = True
                i = j
            else: i+=1
        return int(jours.sum()), jours

    # -------- Tn/Tm/Tx mensuels --------
    tstats = {k:[] for k in data}
    for m in range(12):
        idx0 = sum(heures_par_mois[:m])
        idx1 = sum(heures_par_mois[:m+1])
        for k in data:
            x = data[k][idx0:idx1]
            tstats[k].append([x.min(), (x.min()+x.max())/2, x.max()])

    st.subheader("Tn / Tm / Tx mensuels")
    fig, ax = plt.subplots(figsize=(14,6))
    for i,k in enumerate(data):
        arr = np.array(tstats[k])
        ax.plot(mois_noms, arr[:,2], "-", label=f"{file_names[k]} Tx", color=couleurs[i])
        ax.plot(mois_noms, arr[:,1], "--", label=f"{file_names[k]} Tm", color=couleurs[i])
        ax.plot(mois_noms, arr[:,0], ":", label=f"{file_names[k]} Tn", color=couleurs[i])
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig); plt.close()

    # -------- Tn/Tm/Tx journaliers --------
    Tn_jour_all, Tm_jour_all, Tx_jour_all = {k:[] for k in data}, {k:[] for k in data}, {k:[] for k in data}

    for m in range(12):
        idx0 = sum(heures_par_mois[:m])
        idx1 = sum(heures_par_mois[:m+1])
        for k in data:
            tn, tm, tx = daily_stats_from_hourly(data[k][idx0:idx1])
            Tn_jour_all[k].append(tn)
            Tm_jour_all[k].append(tm)
            Tx_jour_all[k].append(tx)

    # -------- Vagues de chaleur --------
    jours_par_mois = [h//24 for h in heures_par_mois]
    df_vagues = {}

    for k in data:
        Tm_all = np.concatenate(Tm_jour_all[k])
        _, mask = nombre_jours_vague(Tm_all)
        idx = 0
        df_vagues[k] = []
        for L in jours_par_mois:
            df_vagues[k].append(int(mask[idx:idx+L].sum()))
            idx += L

    df_vagues = pd.DataFrame(df_vagues, index=mois_noms)
    st.subheader("Vagues de chaleur")
    st.dataframe(df_vagues)

    fig, ax = plt.subplots(figsize=(12,5))
    x = np.arange(12)
    for i,k in enumerate(data):
        ax.bar(bar_positions(x,i), df_vagues[k], width=bar_width, label=file_names[k], color=couleurs[i])
    ax.set_xticks(x)
    ax.set_xticklabels(mois_noms, rotation=45)
    ax.legend()
    st.pyplot(fig); plt.close()

    # -------- Jours chauds / nuits tropicales --------
    st.subheader("Jours chauds et nuits tropicales")
    tx_seuil = st.number_input("Tx jour >",25)
    tn_seuil = st.number_input("Tn nuit >",20)

    jc, nt = {}, {}
    for k in data:
        jc[k] = [int(np.sum(Tx_jour_all[k][m] > tx_seuil)) for m in range(12)]
        nt[k] = [int(np.sum(Tn_jour_all[k][m] > tn_seuil)) for m in range(12)]

    fig, ax = plt.subplots(2,1,figsize=(14,8))
    for i,k in enumerate(data):
        ax[0].bar(bar_positions(x,i), jc[k], width=bar_width, label=file_names[k], color=couleurs[i])
        ax[1].bar(bar_positions(x,i), nt[k], width=bar_width, label=file_names[k], color=couleurs[i])

    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(mois_noms, rotation=45)
        a.legend()

    ax[0].set_title("Jours chauds")
    ax[1].set_title("Nuits tropicales")
    st.pyplot(fig); plt.close()

    # -------- DJC / DJF --------
    st.subheader("DJC / DJF")
    base_c = st.number_input("Base DJC",19)
    base_f = st.number_input("Base DJF",23)

    DJC, DJF = {}, {}
    for k in data:
        DJC[k], DJF[k] = [], []
        for m in range(12):
            Tm = (Tx_jour_all[k][m] + Tn_jour_all[k][m]) / 2
            DJC[k].append(np.sum(np.maximum(0, base_c - Tm)))
            DJF[k].append(np.sum(np.maximum(0, Tm - base_f)))

    fig, ax = plt.subplots(2,1,figsize=(14,8))
    for i,k in enumerate(data):
        ax[0].bar(bar_positions(x,i), DJC[k], width=bar_width, label=file_names[k], color=couleurs[i])
        ax[1].bar(bar_positions(x,i), DJF[k], width=bar_width, label=file_names[k], color=couleurs[i])

    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(mois_noms, rotation=45)
        a.legend()

    ax[0].set_title("DJC mensuel")
    ax[1].set_title("DJF mensuel")
    st.pyplot(fig); plt.close()

    st.subheader("Sommes annuelles")
    for k in data:
        st.write(f"{file_names[k]} → DJC={int(sum(DJC[k]))} | DJF={int(sum(DJF[k]))}")

