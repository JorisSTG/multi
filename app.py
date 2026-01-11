import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- STYLE sombre ----
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
st.markdown(
    "L’objectif de cette application est d’évaluer la précision et la cohérence entre **N jeux de données** (température uniquement).",
    unsafe_allow_html=True
)

# -------- Paramètres --------
heures_par_mois = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
couleurs = ["goldenrod", "lightgray", "navy", "green", "darkmagenta", "peru", "silver", "orange"]

mois_noms = {
    1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril", 5: "Mai", 6: "Juin",
    7: "Juillet", 8: "Août", 9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
}

# -------- Upload CSV --------
uploaded_files = st.file_uploader("Déposer les fichiers CSV :", type=["csv"], accept_multiple_files=True)

if len(uploaded_files) >= 2:

    data, file_names = {}, {}
    for i, file in enumerate(uploaded_files):
        key = f"source_{i+1}"
        data[key] = pd.read_csv(file).iloc[:, 0].values
        file_names[key] = file.name.replace(".csv", "")

    n_files = len(data)
    group_width = 1 - 0.2
    bar_width = group_width / n_files
    offsets = (np.arange(n_files) - (n_files-1)/2) * bar_width

    # -------- Fonctions --------
    def daily_stats_from_hourly(hourly):
        if len(hourly) < 24:
            return np.array([]), np.array([]), np.array([])
        n = len(hourly) // 24
        arr = np.array(hourly[:n*24]).reshape((n, 24))
        return arr.min(axis=1), arr.mean(axis=1), arr.max(axis=1)

    def nombre_jours_vague(T):
        T = np.array(T)
        n = len(T)
        jours_vague = np.zeros(n, dtype=bool)
        jours_vague[T >= 25.3] = True
        i = 0
        while i < n:
            if i + 2 < n and np.all(T[i:i+3] >= 23.4):
                debut = i
                fin = i + 2
                j = fin + 1
                while j < n and T[j] >= 23.4:
                    fin = j
                    j += 1
                prolong = fin + 1
                compteur = 0
                while prolong < n and compteur < 2:
                    if T[prolong] < 22.4:
                        break
                    fin = prolong
                    compteur += 1
                    prolong += 1
                jours_vague[debut:fin+1] = True
                i = fin + 1
            else:
                i += 1
        return int(jours_vague.sum()), jours_vague

    # -------- Stats journalières --------
    Tn_jour_all = {k: [] for k in data}
    Tm_jour_all = {k: [] for k in data}
    Tx_jour_all = {k: [] for k in data}

    for mois in range(1, 13):
        i0 = sum(heures_par_mois[:mois-1])
        i1 = sum(heures_par_mois[:mois])
        for k in data:
            tn, tm, tx = daily_stats_from_hourly(data[k][i0:i1])
            Tn_jour_all[k].append(tn)
            Tm_jour_all[k].append(tm)
            Tx_jour_all[k].append(tx)

    # =========================================================
    # ================== HISTOGRAMMES ========================
    # =========================================================

    st.subheader("Histogrammes mensuels (1°C)")
    bin_edges = np.arange(-10, 46, 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for m in range(12):
        fig, ax = plt.subplots(figsize=(12,4))
        for i, k in enumerate(data):
            counts, _ = np.histogram(data[k][sum(heures_par_mois[:m]):sum(heures_par_mois[:m+1])],
                                      bins=bin_edges)
            ax.bar(bin_centers + offsets[i], counts, width=bar_width,
                   color=couleurs[i], label=file_names[k])
        ax.set_title(f"{mois_noms[m+1]} – Histogramme températures")
        ax.set_xlabel("Température (°C)")
        ax.set_ylabel("Nombre d'heures")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Histogramme annuel (1°C)")
    fig, ax = plt.subplots(figsize=(13,4))
    for i, k in enumerate(data):
        counts, _ = np.histogram(data[k], bins=bin_edges)
        ax.bar(bin_centers + offsets[i], counts, width=bar_width,
               color=couleurs[i], label=file_names[k])
    ax.set_title("Histogramme annuel températures")
    ax.set_xlabel("Température (°C)")
    ax.set_ylabel("Nombre d'heures")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # =========================================================
    # ====================== CDF ==============================
    # =========================================================
    st.subheader("CDF mensuelles – Tm journalière")
    for m in range(12):
        fig, ax = plt.subplots(figsize=(7,4))
        for i, k in enumerate(data):
            X = np.sort(Tm_jour_all[k][m])
            F = np.arange(1, len(X)+1) / len(X)
            ax.plot(X, F, color=couleurs[i], label=file_names[k])
        ax.set_title(f"CDF Tm – {mois_noms[m+1]}")
        ax.set_xlabel("Température (°C)")
        ax.set_ylabel("Probabilité cumulée")
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("CDF annuelle – Tm journalière")
    fig, ax = plt.subplots(figsize=(8,5))
    for i, k in enumerate(data):
        X = np.sort(np.concatenate(Tm_jour_all[k]))
        F = np.arange(1, len(X)+1) / len(X)
        ax.plot(X, F, color=couleurs[i], label=file_names[k])
    ax.set_title("CDF annuelle Tm")
    ax.set_xlabel("Température (°C)")
    ax.set_ylabel("Probabilité cumulée")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # -------- Vagues de chaleur --------
    st.subheader("Vagues de chaleur")
    Tm_all = {key: np.concatenate(Tm_jour_all[key]) if len(Tm_jour_all[key])>0 else np.array([]) for key in data}
    jours_par_mois = [h // 24 for h in heures_par_mois]
    jours_vagues = {key: [] for key in data}

    for key in data:
        if len(Tm_all[key])>0:
            _, jours_vague_all = nombre_jours_vague(Tm_all[key])
        else:
            jours_vague_all = np.zeros(sum(jours_par_mois), dtype=bool)
        idx = 0
        for L in jours_par_mois:
            nb = int(jours_vague_all[idx:idx+L].sum())
            jours_vagues[key].append(nb)
            idx += L

    df_vagues = pd.DataFrame(jours_vagues)
    df_vagues["Mois"] = list(mois_noms.values())
    st.dataframe(df_vagues.set_index("Mois"), use_container_width=True)

    fig, ax = plt.subplots(figsize=(12,5))
    x = np.arange(12)
    width = 1.0 / len(data)
    for i, key in enumerate(data):
        ax.bar(x + i*width, df_vagues[key], width=width, label=file_names[key], color=couleurs[i])
    ax.set_xticks(x + width*(len(data)-1)/2)
    ax.set_xticklabels(list(mois_noms.values()), rotation=45)
    ax.set_xlabel("Mois")
    ax.set_ylabel("Nombre de jours de vague de chaleur")
    ax.set_title("Nombre de jours de vague de chaleur par mois")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # -------- Jours chauds et nuits tropicales --------
    st.subheader("Jours chauds et nuits tropicales")
    tx_seuil = st.number_input("Seuil Tx_jour (°C) pour jours chauds :", value=25, step=1)
    tn_seuil = st.number_input("Seuil Tn_jour (°C) pour nuits tropicales :", value=20, step=1)
    jours_chauds, nuits_tropicales = {k: [] for k in data}, {k: [] for k in data}

    for mois_num in range(1, 13):
        for k in data:
            jours_chauds[k].append(np.sum(Tx_jour_all[k][mois_num-1] > tx_seuil) if len(Tx_jour_all[k][mois_num-1])>0 else 0)
            nuits_tropicales[k].append(np.sum(Tn_jour_all[k][mois_num-1] > tn_seuil) if len(Tn_jour_all[k][mois_num-1])>0 else 0)

    df_jours_chauds = pd.DataFrame(jours_chauds)
    df_jours_chauds["Mois"] = list(mois_noms.values())
    df_nuits_trop = pd.DataFrame(nuits_tropicales)
    df_nuits_trop["Mois"] = list(mois_noms.values())
    st.dataframe(df_jours_chauds.set_index("Mois"), use_container_width=True)
    st.dataframe(df_nuits_trop.set_index("Mois"), use_container_width=True)

    fig, ax = plt.subplots(2,1, figsize=(14,8))
    for i,k in enumerate(data):
        ax[0].bar(x + i*width, df_jours_chauds[k], width=width, label=file_names[k], color=couleurs[i])
        ax[1].bar(x + i*width, df_nuits_trop[k], width=width, label=file_names[k], color=couleurs[i])
    for a in ax:
        a.set_xticks(x + width*(len(data)-1)/2)
        a.set_xticklabels(list(mois_noms.values()), rotation=45)
    ax[0].set_ylabel(f"Nombre de jours Tx_jour > {tx_seuil}°C")
    ax[0].set_title("Jours chauds par mois")
    ax[0].legend()
    ax[1].set_ylabel(f"Nombre de nuits Tn_jour > {tn_seuil}°C")
    ax[1].set_title("Nuits tropicales par mois")
    ax[1].legend()
    st.pyplot(fig)
    plt.close(fig)

    # -------- DJC et DJF --------
    st.subheader("DJC (chauffage) et DJF (froid)")
    T_base_chauffage = float(st.text_input("Base DJC (°C) — chauffage", "19"))
    T_base_froid = float(st.text_input("Base DJF (°C) — refroidissement", "23"))
    results_djc, results_djf = {k: [] for k in data}, {k: [] for k in data}

    for mois_num in range(1, 13):
        for k in data:
            Tx, Tn = Tx_jour_all[k][mois_num-1], Tn_jour_all[k][mois_num-1]
            DJC_jours = [max(0, (T_base_chauffage - (Tx[j]+Tn[j])/2)) for j in range(len(Tx))]
            DJF_jours = [max(0, ((Tx[j]+Tn[j])/2 - T_base_froid)) for j in range(len(Tx))]
            results_djc[k].append(float(np.nansum(DJC_jours)))
            results_djf[k].append(float(np.nansum(DJF_jours)))

    df_DJC = pd.DataFrame(results_djc)
    df_DJC["Mois"] = list(mois_noms.values())
    df_DJF = pd.DataFrame(results_djf)
    df_DJF["Mois"] = list(mois_noms.values())
    st.dataframe(df_DJC.set_index("Mois"), use_container_width=True)
    st.dataframe(df_DJF.set_index("Mois"), use_container_width=True)

    fig, ax = plt.subplots(2,1, figsize=(14,8))
    for i,k in enumerate(data):
        ax[0].bar(x + i*width, df_DJC[k], width=width, label=file_names[k], color=couleurs[i])
        ax[1].bar(x + i*width, df_DJF[k], width=width, label=file_names[k], color=couleurs[i])
    for a in ax:
        a.set_xticks(x + width*(len(data)-1)/2)
        a.set_xticklabels(list(mois_noms.values()), rotation=45)
    ax[0].set_ylabel("DJC (°C·jour)")
    ax[0].set_title("DJC mensuel")
    ax[0].legend()
    ax[1].set_ylabel("DJF (°C·jour)")
    ax[1].set_title("DJF mensuel")
    ax[1].legend()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Sommes annuelles")
    total_DJC = {k: df_DJC[k].sum() for k in data}
    total_DJF = {k: df_DJF[k].sum() for k in data}
    for k in data:
        st.write(f"{file_names[k]} — DJC annuel : {total_DJC[k]:.0f}, DJF annuel : {total_DJF[k]:.0f}")





