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

# -------- Mois --------
mois_noms = {
    1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril", 5: "Mai", 6: "Juin",
    7: "Juillet", 8: "Août", 9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
}

# -------- Upload --------
uploaded_files = st.file_uploader("Déposer les fichiers CSV :", type=["csv"], accept_multiple_files=True)

if len(uploaded_files) >= 2:

    data, file_names = {}, {}
    for i, file in enumerate(uploaded_files):
        key = f"source_{i+1}"
        data[key] = pd.read_csv(file).iloc[:, 0].values
        file_names[key] = file.name.replace(".csv", "")

    n_files = len(data)

    # -------- Param barres --------
    group_width = 1 - 0.2
    bar_width = group_width / n_files
    offsets = (np.arange(n_files) - (n_files-1)/2) * bar_width

    # -------- Fonctions --------
    def daily_stats_from_hourly(hourly):
        n = len(hourly) // 24
        arr = np.array(hourly[:n*24]).reshape((n, 24))
        return arr.min(axis=1), arr.mean(axis=1), arr.max(axis=1)

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

    # -------- Histogramme annuel --------
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

    # -------- CDF mensuelles --------
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

    # -------- CDF annuelle --------
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
