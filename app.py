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
st.markdown(
    """
    L’objectif de cette application est d’évaluer la précision et la cohérence entre **N jeux de données** (température uniquement) à des fins de simulations STD.
    """,
    unsafe_allow_html=True
)

# -------- Paramètres --------
heures_par_mois = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
percentiles_list = [10, 25, 50, 75, 90]
couleurs = ["goldenrod", "lightgray", "lightblue", "lightgreen", "salmon", "cyan", "magenta", "orange"]

# -------- Noms des mois --------
mois_noms = {
    1: "Janvier", 2: "Février", 3: "Mars",
    4: "Avril", 5: "Mai", 6: "Juin",
    7: "Juillet", 8: "Août", 9: "Septembre",
    10: "Octobre", 11: "Novembre", 12: "Décembre"
}

# -------- Upload des fichiers CSV --------
uploaded_files = st.file_uploader(
    "Déposer les fichiers CSV (colonne unique T°C) :",
    type=["csv"],
    accept_multiple_files=True
)

if len(uploaded_files) >= 2:
    st.markdown("---")

    # -------- Lecture des fichiers CSV --------
    data = {}
    for i, file in enumerate(uploaded_files):
        data[f"source_{i+1}"] = pd.read_csv(file, header=0).iloc[:, 0].values

    # -------- Création de df_ref (référence) --------
    ref_key = "source_1"
    ref_values = data[ref_key]
    df_ref = pd.DataFrame({"T2m": ref_values})
    df_ref["month_num"] = pd.concat([pd.Series([m] * h) for m, h in enumerate(heures_par_mois, start=1)], ignore_index=True)
    df_ref["month"] = df_ref["month_num"].map(mois_noms)

    # -------- Fonctions utilitaires --------
    def rmse(a, b):
        min_len = min(len(a), len(b))
        a_sorted = np.sort(a[:min_len])
        b_sorted = np.sort(b[:min_len])
        return np.sqrt(np.nanmean((a_sorted - b_sorted) ** 2))

    def precision_overlap(a, b, bin_width=1.0):
        if len(a) == 0 or len(b) == 0:
            return np.nan
        min_val = min(np.min(a), np.min(b))
        max_val = max(np.max(a), np.max(b))
        bins = np.arange(min_val, max_val + bin_width, bin_width)
        hist_a, _ = np.histogram(a, bins=bins, density=True)
        hist_b, _ = np.histogram(b, bins=bins, density=True)
        overlap = np.sum(np.minimum(hist_a, hist_b) * bin_width)
        return round(overlap * 100, 2)

    # -------- Comparaison de toutes les sources --------
    results_all = {key: [] for key in data if key != ref_key}
    tstats_all = {key: [] for key in data}

    # -------- Boucle sur les mois --------
    for mois_num, nb_heures in enumerate(heures_par_mois, start=1):
        mois = mois_noms[mois_num]
        ref_mois = df_ref[df_ref["month_num"] == mois_num]["T2m"].values

        for key in data:
            model_values = data[key]
            start_idx = sum(heures_par_mois[:mois_num-1])
            mod_mois = model_values[start_idx:start_idx + nb_heures]

            if key != ref_key:
                val_rmse = rmse(mod_mois, ref_mois)
                pct_precision = precision_overlap(mod_mois, ref_mois)
                results_all[key].append({
                    "Mois": mois,
                    "RMSE (°C)": round(val_rmse, 2),
                    "Précision (%)": pct_precision
                })

            # Calcul des Tn/Tm/Tx pour chaque source
            mod_tn = np.min(mod_mois)
            mod_tm = np.mean(mod_mois)
            mod_tx = np.max(mod_mois)

            tstats_all[key].append({
                "Mois": mois,
                "Tn": mod_tn,
                "Tm": mod_tm,
                "Tx": mod_tx
            })

    # -------- DataFrame consolidé pour les RMSE/Précision --------
    df_results = pd.DataFrame()
    for key in results_all:
        df_source = pd.DataFrame(results_all[key])
        df_source["Source"] = key
        df_results = pd.concat([df_results, df_source], ignore_index=True)

    # -------- Affichage du tableau des RMSE/Précision --------
    st.subheader("Précision par mois pour toutes les sources (par rapport à la source 1)")
    df_results_styled = (
        df_results.style
        .background_gradient(subset=["Précision (%)"], cmap="RdYlGn", vmin=50, vmax=100, axis=None)
        .format({"Précision (%)": "{:.2f}", "RMSE (°C)": "{:.2f}"})
    )
    st.dataframe(df_results_styled, hide_index=True)

    # -------- Graphiques superposés Tn/Tm/Tx --------
    st.subheader("Comparaison des températures mensuelles (Tn/Tm/Tx)")

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, key in enumerate(data):
        df_tstats = pd.DataFrame(tstats_all[key])
        ax.plot(df_tstats["Mois"], df_tstats["Tx"], label=f"{key} Tx", color=couleurs[i], linestyle="-")
        ax.plot(df_tstats["Mois"], df_tstats["Tm"], label=f"{key} Tm", color=couleurs[i], linestyle="--")
        ax.plot(df_tstats["Mois"], df_tstats["Tn"], label=f"{key} Tn", color=couleurs[i], linestyle=":")

    ax.set_title("Tn/Tm/Tx mensuels pour toutes les sources")
    ax.set_ylabel("Température (°C)")
    ax.tick_params(axis='x', rotation=45)
    ax.legend(facecolor="black", bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)
    plt.close(fig)

    # -------- Histogrammes mensuels --------
    st.subheader("Histogrammes mensuels (tranches de 1 °C)")

    bin_edges = np.arange(-10, 46, 1)
    bin_labels = bin_edges[:-1].astype(int)

    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]
        fig, ax = plt.subplots(figsize=(12, 4))

        for i, key in enumerate(data):
            model_values = data[key]
            start_idx = sum(heures_par_mois[:mois_num-1])
            mod_mois = model_values[start_idx:start_idx + heures_par_mois[mois_num-1]]
            mod_counts, _ = np.histogram(mod_mois, bins=bin_edges)
            ax.bar(bin_labels + i * 0.2, mod_counts, width=0.2, label=f"{key}", color=couleurs[i])

        ax.set_title(f"{mois} - Histogramme des températures (1 °C)")
        ax.set_xlabel("Température (°C)")
        ax.set_ylabel("Nombre d'heures")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    # -------- Histogramme annuel --------
    st.subheader("Histogramme annuel (tranches de 1 °C)")

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, key in enumerate(data):
        model_values = data[key]
        mod_counts, _ = np.histogram(model_values, bins=bin_edges)
        ax.bar(bin_labels + i * 0.2, mod_counts, width=0.2, label=f"{key}", color=couleurs[i])

    ax.set_title("Histogramme annuel des températures (1 °C)")
    ax.set_xlabel("Température (°C)")
    ax.set_ylabel("Nombre d'heures")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # -------- Courbes CDF mensuelles --------
    st.subheader("Courbes CDF mensuelles")

    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]
        fig, ax = plt.subplots(figsize=(12, 4))

        for i, key in enumerate(data):
            model_values = data[key]
            start_idx = sum(heures_par_mois[:mois_num-1])
            mod_mois = model_values[start_idx:start_idx + heures_par_mois[mois_num-1]]
            mod_mois_sorted = np.sort(mod_mois)
            cdf = np.arange(1, len(mod_mois_sorted) + 1) / len(mod_mois_sorted)
            ax.plot(mod_mois_sorted, cdf, label=f"{key}", color=couleurs[i])

        ax.set_title(f"{mois} - Courbe CDF")
        ax.set_xlabel("Température (°C)")
        ax.set_ylabel("CDF")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    # -------- Courbe CDF annuelle --------
    st.subheader("Courbe CDF annuelle")

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, key in enumerate(data):
        model_values = data[key]
        model_values_sorted = np.sort(model_values)
        cdf = np.arange(1, len(model_values_sorted) + 1) / len(model_values_sorted)
        ax.plot(model_values_sorted, cdf, label=f"{key}", color=couleurs[i])

    ax.set_title("Courbe CDF annuelle")
    ax.set_xlabel("Température (°C)")
    ax.set_ylabel("CDF")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
