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
    1: "01 - Janvier", 2: "02 - Février", 3: "03 - Mars",
    4: "04 - Avril", 5: "05 - Mai", 6: "06 - Juin",
    7: "07 - Juillet", 8: "08 - Août", 9: "09 - Septembre",
    10: "10 - Octobre", 11: "11 - Novembre", 12: "12 - Décembre"
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
    df_ref["year"] = 2023
    df_ref["month_num"] = pd.concat([pd.Series([m] * h) for m, h in enumerate(heures_par_mois, start=1)], ignore_index=True)
    df_ref["month"] = df_ref["month_num"].map(mois_noms)
    df_ref["day"] = pd.concat([pd.Series(range(1, h // 24 + 2)) for h in heures_par_mois], ignore_index=True)[:len(ref_values)]

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
    results_all = {key: [] for key in data}
    tstats_all = {key: [] for key in data}

    # -------- Boucle sur les mois --------
    for mois_num, nb_heures in enumerate(heures_par_mois, start=1):
        mois = mois_noms[mois_num]
        ref_mois = df_ref[df_ref["month_num"] == mois_num]["T2m"].values

        for key in data:
            model_values = data[key]
            start_idx = sum(heures_par_mois[:mois_num-1])
            mod_mois = model_values[start_idx:start_idx + nb_heures]

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
    st.subheader("Précision par mois pour toutes les sources")
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

    # -------- Tableau des différences par rapport à la référence --------
    st.subheader("Différences par rapport à la référence (source 1)")
    df_diff = pd.DataFrame()
    ref_tstats = pd.DataFrame(tstats_all[ref_key])

    for key in data:
        if key == ref_key:
            continue
        df_source = pd.DataFrame(tstats_all[key])
        df_source["Source"] = key
        df_source["Diff_Tn"] = df_source["Tn"] - ref_tstats["Tn"]
        df_source["Diff_Tm"] = df_source["Tm"] - ref_tstats["Tm"]
        df_source["Diff_Tx"] = df_source["Tx"] - ref_tstats["Tx"]
        df_diff = pd.concat([df_diff, df_source], ignore_index=True)

    df_diff_styled = (
        df_diff.style
        .background_gradient(subset=["Diff_Tn", "Diff_Tm", "Diff_Tx"], cmap="bwr", vmin=-5, vmax=5)
        .format("{:.2f}")
    )
    st.dataframe(df_diff_styled, hide_index=True)
