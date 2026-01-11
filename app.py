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
couleurs = ["goldenrod", "lightgray", "navy", "green", "darkmagenta", "peru", "silver", "orange"]
vmaxH = 100
vminH = -100
vmaxJ = 150
vminJ = -150

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
    file_names = {}
    for i, file in enumerate(uploaded_files):
        file_name = file.name.replace(".csv", "")
        file_names[f"source_{i+1}"] = file_name
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

    def count_days_in_bins(daily_values, bin_edges):
        return np.histogram(daily_values, bins=bin_edges)[0]

    def daily_stats_from_hourly(hourly):
        if len(hourly) < 24:
            return np.array([]), np.array([]), np.array([])
        n_full_days = len(hourly) // 24
        arr = np.array(hourly[: n_full_days * 24]).reshape((n_full_days, 24))
        daily_min = arr.min(axis=1)
        daily_mean = arr.mean(axis=1)
        daily_max = arr.max(axis=1)
        return daily_min, daily_mean, daily_max

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
        df_source["Source"] = file_names[key]
        df_results = pd.concat([df_results, df_source], ignore_index=True)

    # -------- Affichage du tableau des RMSE/Précision --------
    st.subheader("Précision par mois pour toutes les sources (par rapport à la référence)")
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
        ax.plot(df_tstats["Mois"], df_tstats["Tx"], label=f"{file_names[key]} Tx", color=couleurs[i], linestyle="-")
        ax.plot(df_tstats["Mois"], df_tstats["Tm"], label=f"{file_names[key]} Tm", color=couleurs[i], linestyle="--")
        ax.plot(df_tstats["Mois"], df_tstats["Tn"], label=f"{file_names[key]} Tn", color=couleurs[i], linestyle=":")

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
    n_files = len(uploaded_files)
    bar_width = 1.0 / n_files

    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]
        fig, ax = plt.subplots(figsize=(12, 4))

        for i, key in enumerate(data):
            model_values = data[key]
            start_idx = sum(heures_par_mois[:mois_num-1])
            mod_mois = model_values[start_idx:start_idx + heures_par_mois[mois_num-1]]
            mod_counts, _ = np.histogram(mod_mois, bins=bin_edges)
            ax.bar(bin_labels + i * bar_width, mod_counts, width=bar_width, label=f"{file_names[key]}", color=couleurs[i])

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
        ax.bar(bin_labels + i * bar_width, mod_counts, width=bar_width, label=f"{file_names[key]}", color=couleurs[i])

    ax.set_title("Histogramme annuel des températures (1 °C)")
    ax.set_xlabel("Température (°C)")
    ax.set_ylabel("Nombre d'heures")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # -------- Seuils --------
    st.subheader("Seuils de température")
    t_sup_thresholds = st.text_input("Seuils supérieurs (°C, séparer les seuils par des / )", "25/30")
    t_inf_thresholds = st.text_input("Seuils inférieurs (°C, séparer les seuils par des / )", "0/5")

    t_sup_thresholds_list = [int(float(x.strip())) for x in t_sup_thresholds.split("/")]
    t_inf_thresholds_list = [int(float(x.strip())) for x in t_inf_thresholds.split("/")]

    stats_sup = []
    stats_inf = []

    # -------- Calculs mensuels --------
    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]
        idx0 = sum(heures_par_mois[:mois_num-1])
        idx1 = sum(heures_par_mois[:mois_num])

        ref_mois = df_ref[df_ref["month_num"] == mois_num]["T2m"].values

        for key in data:
            mod_mois = data[key][idx0:idx1]

            # Seuils supérieurs
            for seuil in t_sup_thresholds_list:
                heures_ref = np.sum(ref_mois > seuil)
                heures_mod = np.sum(mod_mois > seuil)
                ecart = heures_mod - heures_ref

                stats_sup.append({
                    "Mois": mois,
                    "Source": file_names[key],
                    "Seuil (°C)": seuil,
                    "Heures": heures_mod,
                    "Heures Référence": heures_ref,
                    "Ecart (Source - Référence)": ecart
                })

            # Seuils inférieurs
            for seuil in t_inf_thresholds_list:
                heures_ref = np.sum(ref_mois < seuil)
                heures_mod = np.sum(mod_mois < seuil)
                ecart = heures_mod - heures_ref

                stats_inf.append({
                    "Mois": mois,
                    "Source": file_names[key],
                    "Seuil (°C)": seuil,
                    "Heures": heures_mod,
                    "Heures Référence": heures_ref,
                    "Ecart (Source - Référence)": ecart
                })

    # -------- DataFrames --------
    df_sup = pd.DataFrame(stats_sup)
    df_inf = pd.DataFrame(stats_inf)

    for df in [df_sup, df_inf]:
        df["Heures"] = df["Heures"].astype(int)
        df["Heures Référence"] = df["Heures Référence"].astype(int)
        df["Ecart (Source - Référence)"] = df["Ecart (Source - Référence)"].astype(int)

    # -------- Affichage mensuel --------
    st.subheader("Nombre d'heures supérieur au(x) seuil(s)")
    df_sup_styled = (
        df_sup.style
        .background_gradient(subset=["Ecart (Source - Référence)"], cmap="bwr", vmin=vminH, vmax=vmaxH, axis=None)
    )
    st.dataframe(df_sup_styled, hide_index=True)

    st.subheader("Nombre d'heures inférieur au(x) seuil(s)")
    df_inf_styled = (
        df_inf.style
        .background_gradient(subset=["Ecart (Source - Référence)"], cmap="bwr_r", vmin=vminH, vmax=vmaxH, axis=None)
    )
    st.dataframe(df_inf_styled, hide_index=True)

    # -------- Sommes annuelles --------
    ref_all = df_ref["T2m"].values
    annual_sup = []
    annual_inf = []

    # Seuils supérieurs annuels
    for key in data:
        mod_all = data[key]
        for seuil in t_sup_thresholds_list:
            heures_ref = np.sum(ref_all > seuil)
            heures_mod = np.sum(mod_all > seuil)
            ecart = heures_mod - heures_ref

            annual_sup.append({
                "Période": "Année",
                "Source": file_names[key],
                "Seuil (°C)": seuil,
                "Heures": int(heures_mod),
                "Heures Référence": int(heures_ref),
                "Ecart (Source - Référence)": int(ecart)
            })

    # Seuils inférieurs annuels
    for key in data:
        mod_all = data[key]
        for seuil in t_inf_thresholds_list:
            heures_ref = np.sum(ref_all < seuil)
            heures_mod = np.sum(mod_all < seuil)
            ecart = heures_mod - heures_ref

            annual_inf.append({
                "Période": "Année",
                "Source": file_names[key],
                "Seuil (°C)": seuil,
                "Heures": int(heures_mod),
                "Heures Référence": int(heures_ref),
                "Ecart (Source - Référence)": int(ecart)
            })

    df_sup_year = pd.DataFrame(annual_sup)
    df_inf_year = pd.DataFrame(annual_inf)

    # -------- Affichage annuel --------
    st.subheader("Somme annuelle — Nombre d'heures supérieur au(x) seuil(s)")
    df_sup_year_styled = (
        df_sup_year.style
        .background_gradient(subset=["Ecart (Source - Référence)"], cmap="bwr", vmin=vminH*12, vmax=vmaxH*12, axis=None)
    )
    st.dataframe(df_sup_year_styled, hide_index=True)

    st.subheader("Somme annuelle — Nombre d'heures inférieur au(x) seuil(s)")
    df_inf_year_styled = (
        df_inf_year.style
        .background_gradient(subset=["Ecart (Source - Référence)"], cmap="bwr_r", vmin=vminH*12, vmax=vmaxH*12, axis=None)
    )
    st.dataframe(df_inf_year_styled, hide_index=True)

    # -------- Calcul des Tn_jour, Tm_jour, Tx_jour --------
    Tn_jour_all = []
    Tm_jour_all = []
    Tx_jour_all = []
    Tn_jour_mod_all = []
    Tm_jour_mod_all = []
    Tx_jour_mod_all = []

    pct_for_cdf = np.linspace(0, 100, 100)

    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]

        ref_hourly = df_ref[df_ref["month_num"] == mois_num]["T2m"].values
        ref_tn, ref_tm, ref_tx = daily_stats_from_hourly(ref_hourly)

        idx0 = sum(heures_par_mois[:mois_num-1])
        idx1 = sum(heures_par_mois[:mois_num])

        for key in data:
            model_hourly = data[key][idx0:idx1]
            mod_tn, mod_tm, mod_tx = daily_stats_from_hourly(model_hourly)

            if key == ref_key:
                Tn_jour_all.append(ref_tn)
                Tm_jour_all.append(ref_tm)
                Tx_jour_all.append(ref_tx)
            else:
                Tn_jour_mod_all.append(mod_tn)
                Tm_jour_mod_all.append(mod_tm)
                Tx_jour_mod_all.append(mod_tx)

    # -------- CDF annuelle Tn / Tx --------
    st.subheader("CDF annuelle Tn / Tm / Tx")

    obs_tn_year = np.concatenate(Tn_jour_all)
    obs_tm_year = np.concatenate(Tm_jour_all)
    obs_tx_year = np.concatenate(Tx_jour_all)

    mod_tn_year = np.concatenate(Tn_jour_mod_all)
    mod_tm_year = np.concatenate(Tm_jour_mod_all)
    mod_tx_year = np.concatenate(Tx_jour_mod_all)

    obs_tn_cdf_year = np.percentile(obs_tn_year, pct_for_cdf)
    mod_tn_cdf_year = np.percentile(mod_tn_year, pct_for_cdf)

    obs_tm_cdf_year = np.percentile(obs_tm_year, pct_for_cdf)
    mod_tm_cdf_year = np.percentile(mod_tm_year, pct_for_cdf)

    obs_tx_cdf_year = np.percentile(obs_tx_year, pct_for_cdf)
    mod_tx_cdf_year = np.percentile(mod_tx_year, pct_for_cdf)

    fig, ax = plt.subplots(figsize=(12, 5))

    colors = {
        "Tn": "cyan",
        "Tm": "white",
        "Tx": "red"
    }

    ax.plot(pct_for_cdf, mod_tx_cdf_year, "-", lw=2, label=f"{file_names[ref_key]} Tx", color=colors["Tx"])
    ax.plot(pct_for_cdf, mod_tm_cdf_year, "-", lw=2, label=f"{file_names[ref_key]} Tmoy", color=colors["Tm"])
    ax.plot(pct_for_cdf, mod_tn_cdf_year, "-", lw=2, label=f"{file_names[ref_key]} Tn", color=colors["Tn"])

    for i, key in enumerate(data):
        if key != ref_key:
            ax.plot(pct_for_cdf, np.percentile(np.concatenate(Tx_jour_mod_all), pct_for_cdf), "--", lw=1.7, label=f"{file_names[key]} Tx", color=colors["Tx"])
            ax.plot(pct_for_cdf, np.percentile(np.concatenate(Tm_jour_mod_all), pct_for_cdf), "--", lw=1.7, label=f"{file_names[key]} Tmoy", color=colors["Tm"])
            ax.plot(pct_for_cdf, np.percentile(np.concatenate(Tn_jour_mod_all), pct_for_cdf), "--", lw=1.7, label=f"{file_names[key]} Tn", color=colors["Tn"])

    ax.set_title("Année complète — CDF Tn_jour / Tmoy_jour / Tx_jour")
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Température (°C)")
    ax.legend(facecolor="black")
    st.pyplot(fig)
    plt.close(fig)

    # -------- Histogramme annuel Tn / Tx --------
    st.subheader("Histogramme annuel Tn / Tx")

    bin_edges = np.arange(-10, 45, 1)
    bin_labels = bin_edges[:-1].astype(int)

    obs_counts_Tn = count_days_in_bins(obs_tn_year, bin_edges)
    mod_counts_Tn = count_days_in_bins(mod_tn_year, bin_edges)

    obs_counts_Tx = count_days_in_bins(obs_tx_year, bin_edges)
    mod_counts_Tx = count_days_in_bins(mod_tx_year, bin_edges)

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(bin_labels - 0.25, obs_counts_Tn, width=0.4, label=f"{file_names[ref_key]} Tn", color="cyan")
    ax.bar(bin_labels + 0.25, mod_counts_Tn, width=0.4, label=f"Autres Tn", color="lightblue")
    ax.set_title("Histogramme annuel – Nombre de jours par classe de Tn")
    ax.set_xlabel("Température (°C)")
    ax.set_ylabel("Nombre de jours")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(bin_labels - 0.25, obs_counts_Tx, width=0.4, label=f"{file_names[ref_key]} Tx", color="red")
    ax.bar(bin_labels + 0.25, mod_counts_Tx, width=0.4, label=f"Autres Tx", color="salmon")
    ax.set_title("Histogramme annuel – Nombre de jours par classe de Tx")
    ax.set_xlabel("Température (°C)")
    ax.set_ylabel("Nombre de jours")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # -------- Calcul des vagues de chaleur --------
    st.subheader("Vagues de chaleur")

    jours_par_mois = [len(Tx_jour_all[m]) for m in range(12)]

    Tm_obs_all = np.concatenate([
        (np.array(Tx_jour_all[m]) + np.array(Tn_jour_all[m])) / 2 for m in range(12)
    ])

    Tm_mod_all = np.concatenate([
        (np.array(Tx_jour_mod_all[m]) + np.array(Tn_jour_mod_all[m])) / 2 for m in range(12)
    ])

    _, jours_vague_obs_all = nombre_jours_vague(Tm_obs_all)
    _, jours_vague_mod_all = nombre_jours_vague(Tm_mod_all)

    jours_vague_obs = []
    jours_vague_mod = []

    idx = 0
    for L in jours_par_mois:
        jours_vague_obs.append(int(jours_vague_obs_all[idx:idx+L].sum()))
        jours_vague_mod.append(int(jours_vague_mod_all[idx:idx+L].sum()))
        idx += L

    df_vagues = pd.DataFrame({
        "Mois": list(mois_noms.values()),
        file_names[ref_key]: jours_vague_obs,
        "Autres": jours_vague_mod
    })

    st.subheader("Nombre de jours de vague de chaleur par mois")
    st.dataframe(df_vagues, hide_index=True, use_container_width=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(1, 13)
    ax.bar(x - 0.2, jours_vague_obs, width=0.4, label=file_names[ref_key], color="cyan")
    ax.bar(x + 0.2, jours_vague_mod, width=0.4, label="Autres", color="lightblue")
    ax.set_xlabel("Mois")
    ax.set_ylabel("Nombre de jours de vague de chaleur")
    ax.set_title("Nombre de jours de vague de chaleur par mois")
    ax.set_xticks(x)
    ax.set_xticklabels(list(mois_noms.values()), rotation=45)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # -------- Jours chauds et nuits tropicales --------
    st.subheader("Jours chauds et nuits tropicales")

    tx_seuil = st.number_input("Seuil Tx_jour (°C) pour jours chauds :", value=25, step=1)
    tn_seuil = st.number_input("Seuil Tn_jour (°C) pour nuits tropicales :", value=20, step=1)

    jours_chauds_obs = []
    jours_chauds_mod = []
    nuits_tropicales_obs = []
    nuits_tropicales_mod = []

    for mois_num in range(1, 13):
        obs_tx_jour = Tx_jour_all[mois_num - 1]
        obs_tn_jour = Tn_jour_all[mois_num - 1]
        jours_chauds_obs.append(np.sum(obs_tx_jour > tx_seuil))
        nuits_tropicales_obs.append(np.sum(obs_tn_jour > tn_seuil))

        mod_tx_jour = Tx_jour_mod_all[mois_num - 1]
        mod_tn_jour = Tn_jour_mod_all[mois_num - 1]
        jours_chauds_mod.append(np.sum(mod_tx_jour > tx_seuil))
        nuits_tropicales_mod.append(np.sum(mod_tn_jour > tn_seuil))

    mois_labels = list(mois_noms.values())
    x = np.arange(len(mois_labels))

    df_jours_chauds = pd.DataFrame({
        "Mois": mois_labels,
        file_names[ref_key]: jours_chauds_obs,
        "Autres": jours_chauds_mod,
    })
    df_jours_chauds["Différence"] = df_jours_chauds["Autres"] - df_jours_chauds[file_names[ref_key]]

    st.markdown("Jours chauds par mois")
    st.dataframe(df_jours_chauds, hide_index=True, use_container_width=True)

    df_nuits_trop = pd.DataFrame({
        "Mois": mois_labels,
        file_names[ref_key]: nuits_tropicales_obs,
        "Autres": nuits_tropicales_mod,
    })
    df_nuits_trop["Différence"] = df_nuits_trop["Autres"] - df_nuits_trop[file_names[ref_key]]

    st.markdown("Nuits tropicales par mois")
    st.dataframe(df_nuits_trop, hide_index=True, use_container_width=True)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(x - 0.2, jours_chauds_obs, width=0.4, label=file_names[ref_key], color="cyan")
    ax.bar(x + 0.2, jours_chauds_mod, width=0.4, label="Autres", color="lightblue")
    ax.set_xticks(x)
    ax.set_xticklabels(mois_labels, rotation=45)
    ax.set_ylabel(f"Nombre de jours Tx_jour > {tx_seuil}°C")
    ax.set_title("Jours chauds par mois")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(x - 0.2, nuits_tropicales_obs, width=0.4, label=file_names[ref_key], color="cyan")
    ax.bar(x + 0.2, nuits_tropicales_mod, width=0.4, label="Autres", color="lightblue")
    ax.set_xticks(x)
    ax.set_xticklabels(mois_labels, rotation=45)
    ax.set_ylabel(f"Nombre de nuits Tn_jour > {tn_seuil}°C")
    ax.set_title("Nuits tropicales par mois")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # -------- Calcul DJC (chauffage) et DJF (froid) --------
    st.subheader("DJC (chauffage) et DJF (froid) journaliers")

    T_base_chauffage = float(st.text_input("Base DJC (°C) — chauffage", "19"))
    T_base_froid = float(st.text_input("Base DJF (°C) — refroidissement", "23"))

    results_djc = []
    results_djf = []

    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]

        Tx_obs = Tx_jour_all[mois_num-1]
        Tn_obs = Tn_jour_all[mois_num-1]

        idx0 = sum(heures_par_mois[:mois_num-1])
        idx1 = sum(heures_par_mois[:mois_num])
        model_hourly = data[ref_key][idx0:idx1]
        Tx_mod, Tm_mod, Tn_mod = daily_stats_from_hourly(model_hourly)

        DJC_obs_jours, DJF_obs_jours = [], []
        DJC_mod_jours, DJF_mod_jours = [], []

        n_jours = len(Tx_obs)
        for j in range(n_jours):
            Tm_obs = (Tx_obs[j] + Tn_obs[j]) / 2
            DJC_obs_jours.append(max(0, T_base_chauffage - Tm_obs))
            DJF_obs_jours.append(max(0, Tm_obs - T_base_froid))

            if j < len(Tx_mod):
                Tm_mod = (Tx_mod[j] + Tn_mod[j]) / 2
                DJC_mod_jours.append(max(0, T_base_chauffage - Tm_mod))
                DJF_mod_jours.append(max(0, Tm_mod - T_base_froid))
            else:
                DJC_mod_jours.append(0)
                DJF_mod_jours.append(0)

        DJC_obs_sum = float(np.nansum(DJC_obs_jours))
        DJC_mod_sum = float(np.nansum(DJC_mod_jours))
        DJF_obs_sum = float(np.nansum(DJF_obs_jours))
        DJF_mod_sum = float(np.nansum(DJF_mod_jours))

        results_djc.append({
            "Mois": mois,
            file_names[ref_key]: DJC_obs_sum,
            "Autres": DJC_mod_sum,
            "Différence": DJC_mod_sum - DJC_obs_sum
        })
        results_djf.append({
            "Mois": mois,
            file_names[ref_key]: DJF_obs_sum,
            "Autres": DJF_mod_sum,
            "Différence": DJF_mod_sum - DJF_obs_sum
        })

    df_DJC = pd.DataFrame(results_djc)
    df_DJF = pd.DataFrame(results_djf)

    st.subheader("DJC – Chauffage (somme journalière par mois)")
    st.dataframe(df_DJC, hide_index=True, use_container_width=True)

    st.subheader("DJF – Refroidissement (somme journalière par mois)")
    st.dataframe(df_DJF, hide_index=True, use_container_width=True)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(df_DJC.index - 0.2, df_DJC[file_names[ref_key]], width=0.4, label=file_names[ref_key], color="cyan")
    ax.bar(df_DJC.index + 0.2, df_DJC["Autres"], width=0.4, label="Autres", color="lightblue")
    ax.set_xticks(df_DJC.index)
    ax.set_xticklabels(df_DJC["Mois"], rotation=45)
    ax.set_ylabel("DJC (°C·jour)")
    ax.set_title("DJC mensuel")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(df_DJF.index - 0.2, df_DJF[file_names[ref_key]], width=0.4, label=file_names[ref_key], color="cyan")
    ax.bar(df_DJF.index + 0.2, df_DJF["Autres"], width=0.4, label="Autres", color="lightblue")
    ax.set_xticks(df_DJF.index)
    ax.set_xticklabels(df_DJF["Mois"], rotation=45)
    ax.set_ylabel("DJF (°C·jour)")
    ax.set_title("DJF mensuel")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    total_DJC_obs = df_DJC[file_names[ref_key]].sum()
    total_DJC_mod = df_DJC["Autres"].sum()
    total_DJF_obs = df_DJF[file_names[ref_key]].sum()
    total_DJF_mod = df_DJF["Autres"].sum()

    st.subheader("Sommes annuelles")
    st.write(f"DJC annuel : {file_names[ref_key]} = {total_DJC_obs:.0f} / Autres = {total_DJC_mod:.0f}")
    st.write(f"DJF annuel : {file_names[ref_key]} = {total_DJF_obs:.0f} / Autres = {total_DJF_mod:.0f}")


