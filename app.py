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
    results_all = {key: [] for key in data}
    tstats_all = {key: [] for key in data}

    # -------- Boucle sur les mois --------
    for mois_num, nb_heures in enumerate(heures_par_mois, start=1):
        mois = mois_noms[mois_num]

        for key in data:
            model_values = data[key]
            start_idx = sum(heures_par_mois[:mois_num-1])
            mod_mois = model_values[start_idx:start_idx + nb_heures]

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

        for key in data:
            mod_mois = data[key][idx0:idx1]

            # Seuils supérieurs
            for seuil in t_sup_thresholds_list:
                heures_mod = np.sum(mod_mois > seuil)

                stats_sup.append({
                    "Mois": mois,
                    "Source": file_names[key],
                    "Seuil (°C)": seuil,
                    "Heures": heures_mod
                })

            # Seuils inférieurs
            for seuil in t_inf_thresholds_list:
                heures_mod = np.sum(mod_mois < seuil)

                stats_inf.append({
                    "Mois": mois,
                    "Source": file_names[key],
                    "Seuil (°C)": seuil,
                    "Heures": heures_mod
                })

    # -------- DataFrames --------
    df_sup = pd.DataFrame(stats_sup)
    df_inf = pd.DataFrame(stats_inf)

    # -------- Affichage mensuel --------
    st.subheader("Nombre d'heures supérieur au(x) seuil(s)")
    df_sup_pivot = df_sup.pivot_table(index=["Mois", "Seuil (°C)"], columns="Source", values="Heures", fill_value=0)
    st.dataframe(df_sup_pivot, use_container_width=True)

    st.subheader("Nombre d'heures inférieur au(x) seuil(s)")
    df_inf_pivot = df_inf.pivot_table(index=["Mois", "Seuil (°C)"], columns="Source", values="Heures", fill_value=0)
    st.dataframe(df_inf_pivot, use_container_width=True)

    # -------- Sommes annuelles --------
    annual_sup = []
    annual_inf = []

    # Seuils supérieurs annuels
    for key in data:
        mod_all = data[key]
        for seuil in t_sup_thresholds_list:
            heures_mod = np.sum(mod_all > seuil)

            annual_sup.append({
                "Période": "Année",
                "Source": file_names[key],
                "Seuil (°C)": seuil,
                "Heures": int(heures_mod)
            })

    # Seuils inférieurs annuels
    for key in data:
        mod_all = data[key]
        for seuil in t_inf_thresholds_list:
            heures_mod = np.sum(mod_all < seuil)

            annual_inf.append({
                "Période": "Année",
                "Source": file_names[key],
                "Seuil (°C)": seuil,
                "Heures": int(heures_mod)
            })

    df_sup_year = pd.DataFrame(annual_sup)
    df_inf_year = pd.DataFrame(annual_inf)

    # -------- Affichage annuel --------
    st.subheader("Somme annuelle — Nombre d'heures supérieur au(x) seuil(s)")
    df_sup_year_pivot = df_sup_year.pivot_table(index=["Période", "Seuil (°C)"], columns="Source", values="Heures", fill_value=0)
    st.dataframe(df_sup_year_pivot, use_container_width=True)

    st.subheader("Somme annuelle — Nombre d'heures inférieur au(x) seuil(s)")
    df_inf_year_pivot = df_inf_year.pivot_table(index=["Période", "Seuil (°C)"], columns="Source", values="Heures", fill_value=0)
    st.dataframe(df_inf_year_pivot, use_container_width=True)

    # -------- Calcul des Tn_jour, Tm_jour, Tx_jour --------
    Tn_jour_all = {key: [] for key in data}
    Tm_jour_all = {key: [] for key in data}
    Tx_jour_all = {key: [] for key in data}

    pct_for_cdf = np.linspace(0, 100, 100)

    for mois_num in range(1, 13):
        idx0 = sum(heures_par_mois[:mois_num-1])
        idx1 = sum(heures_par_mois[:mois_num])

        for key in data:
            model_hourly = data[key][idx0:idx1]
            mod_tn, mod_tm, mod_tx = daily_stats_from_hourly(model_hourly)

            Tn_jour_all[key].append(mod_tn)
            Tm_jour_all[key].append(mod_tm)
            Tx_jour_all[key].append(mod_tx)

    # -------- CDF annuelle Tn / Tx --------
    st.subheader("CDF annuelle Tn / Tm / Tx")

    fig, ax = plt.subplots(figsize=(12, 5))

    colors = {
        "Tn": "cyan",
        "Tm": "white",
        "Tx": "red"
    }

    for key in data:
        Tn_year = np.concatenate(Tn_jour_all[key])
        Tm_year = np.concatenate(Tm_jour_all[key])
        Tx_year = np.concatenate(Tx_jour_all[key])

        Tn_cdf_year = np.percentile(Tn_year, pct_for_cdf)
        Tm_cdf_year = np.percentile(Tm_year, pct_for_cdf)
        Tx_cdf_year = np.percentile(Tx_year, pct_for_cdf)

        ax.plot(pct_for_cdf, Tx_cdf_year, "-", lw=1.5, label=f"{file_names[key]} Tx", color=colors["Tx"])
        ax.plot(pct_for_cdf, Tm_cdf_year, "-", lw=1.5, label=f"{file_names[key]} Tm", color=colors["Tm"])
        ax.plot(pct_for_cdf, Tn_cdf_year, "-", lw=1.5, label=f"{file_names[key]} Tn", color=colors["Tn"])

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

    fig, ax = plt.subplots(2, 1, figsize=(15, 10))

    for key in data:
        Tn_year = np.concatenate(Tn_jour_all[key])
        Tx_year = np.concatenate(Tx_jour_all[key])

        Tn_counts = count_days_in_bins(Tn_year, bin_edges)
        Tx_counts = count_days_in_bins(Tx_year, bin_edges)

        ax[0].bar(bin_labels + list(data.keys()).index(key) * bar_width, Tn_counts, width=bar_width, label=f"{file_names[key]} Tn", color=couleurs[list(data.keys()).index(key)])
        ax[1].bar(bin_labels + list(data.keys()).index(key) * bar_width, Tx_counts, width=bar_width, label=f"{file_names[key]} Tx", color=couleurs[list(data.keys()).index(key)])

    ax[0].set_title("Histogramme annuel – Nombre de jours par classe de Tn")
    ax[0].set_xlabel("Température (°C)")
    ax[0].set_ylabel("Nombre de jours")
    ax[0].legend()

    ax[1].set_title("Histogramme annuel – Nombre de jours par classe de Tx")
    ax[1].set_xlabel("Température (°C)")
    ax[1].set_ylabel("Nombre de jours")
    ax[1].legend()

    st.pyplot(fig)
    plt.close(fig)

    # -------- Calcul des vagues de chaleur --------
    st.subheader("Vagues de chaleur")

    # Calcul des Tm (température moyenne journalière) pour chaque source
    Tm_jour_all = {key: [] for key in data}
    for mois_num in range(1, 13):
        for key in data:
            idx0 = sum(heures_par_mois[:mois_num-1])
            idx1 = sum(heures_par_mois[:mois_num])
            model_hourly = data[key][idx0:idx1]
            mod_tn, mod_tm, mod_tx = daily_stats_from_hourly(model_hourly)
            Tm_jour_all[key].append(mod_tm)

    # Concaténation des Tm pour toute l'année
    Tm_all = {key: np.concatenate(Tm_jour_all[key]) for key in data}

    # Calcul des jours de vague de chaleur pour chaque source
    jours_vagues = {key: [] for key in data}
    jours_par_mois = [len(Tm_jour_all[key][0]) for key in data][:12]  # Longueur des mois

    for key in data:
        _, jours_vague_all = nombre_jours_vague(Tm_all[key])
        idx = 0
        for L in jours_par_mois:
            jours_vagues[key].append(jours_vague_all[idx:idx+L].sum())
            idx += L

    # Création du DataFrame
    df_vagues = pd.DataFrame(jours_vagues)
    df_vagues["Mois"] = list(mois_noms.values())

    # Affichage du tableau
    st.subheader("Nombre de jours de vague de chaleur par mois")
    st.dataframe(df_vagues.set_index("Mois"), use_container_width=True)

    # Affichage du graphique
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(1, 13)
    n_sources = len(data)
    bar_width = 1.0 / n_sources  # Largeur des barres en fonction du nombre de sources

    for i, key in enumerate(data):
        ax.bar(x + i * bar_width, df_vagues[key], width=bar_width, label=file_names[key], color=couleurs[i])

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

    jours_chauds = {key: [] for key in data}
    nuits_tropicales = {key: [] for key in data}

    for mois_num in range(1, 13):
        for key in data:
            jours_chauds[key].append(np.sum(Tx_jour_all[key][mois_num-1] > tx_seuil))
            nuits_tropicales[key].append(np.sum(Tn_jour_all[key][mois_num-1] > tn_seuil))

    df_jours_chauds = pd.DataFrame(jours_chauds)
    df_jours_chauds["Mois"] = list(mois_noms.values())

    df_nuits_trop = pd.DataFrame(nuits_tropicales)
    df_nuits_trop["Mois"] = list(mois_noms.values())

    st.markdown("Jours chauds par mois")
    st.dataframe(df_jours_chauds.set_index("Mois"), use_container_width=True)

    st.markdown("Nuits tropicales par mois")
    st.dataframe(df_nuits_trop.set_index("Mois"), use_container_width=True)

    fig, ax = plt.subplots(2, 1, figsize=(14, 8))

    x = np.arange(1, 13)
    width = 1.0 / len(data)

    for i, key in enumerate(data):
        ax[0].bar(x + i * width, df_jours_chauds[key], width=width, label=file_names[key], color=couleurs[i])
        ax[1].bar(x + i * width, df_nuits_trop[key], width=width, label=file_names[key], color=couleurs[i])

    ax[0].set_xticks(x)
    ax[0].set_xticklabels(list(mois_noms.values()), rotation=45)
    ax[0].set_ylabel(f"Nombre de jours Tx_jour > {tx_seuil}°C")
    ax[0].set_title("Jours chauds par mois")
    ax[0].legend()

    ax[1].set_xticks(x)
    ax[1].set_xticklabels(list(mois_noms.values()), rotation=45)
    ax[1].set_ylabel(f"Nombre de nuits Tn_jour > {tn_seuil}°C")
    ax[1].set_title("Nuits tropicales par mois")
    ax[1].legend()

    st.pyplot(fig)
    plt.close(fig)

    # -------- Calcul DJC (chauffage) et DJF (froid) --------
    st.subheader("DJC (chauffage) et DJF (froid) journaliers")

    T_base_chauffage = float(st.text_input("Base DJC (°C) — chauffage", "19"))
    T_base_froid = float(st.text_input("Base DJF (°C) — refroidissement", "23"))

    results_djc = {key: [] for key in data}
    results_djf = {key: [] for key in data}

    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]

        for key in data:
            Tx = Tx_jour_all[key][mois_num-1]
            Tn = Tn_jour_all[key][mois_num-1]

            DJC_jours = []
            DJF_jours = []

            n_jours = len(Tx)
            for j in range(n_jours):
                Tm = (Tx[j] + Tn[j]) / 2
                DJC_jours.append(max(0, T_base_chauffage - Tm))
                DJF_jours.append(max(0, Tm - T_base_froid))

            DJC_sum = float(np.nansum(DJC_jours))
            DJF_sum = float(np.nansum(DJF_jours))

            results_djc[key].append(DJC_sum)
            results_djf[key].append(DJF_sum)

    df_DJC = pd.DataFrame(results_djc)
    df_DJC["Mois"] = list(mois_noms.values())

    df_DJF = pd.DataFrame(results_djf)
    df_DJF["Mois"] = list(mois_noms.values())

    st.subheader("DJC – Chauffage (somme journalière par mois)")
    st.dataframe(df_DJC.set_index("Mois"), use_container_width=True)

    st.subheader("DJF – Refroidissement (somme journalière par mois)")
    st.dataframe(df_DJF.set_index("Mois"), use_container_width=True)

    fig, ax = plt.subplots(2, 1, figsize=(14, 8))

    x = np.arange(1, 13)
    width = 1.0 / len(data)

    for i, key in enumerate(data):
        ax[0].bar(x + i * width, df_DJC[key], width=width, label=file_names[key], color=couleurs[i])
        ax[1].bar(x + i * width, df_DJF[key], width=width, label=file_names[key], color=couleurs[i])

    ax[0].set_xticks(x)
    ax[0].set_xticklabels(list(mois_noms.values()), rotation=45)
    ax[0].set_ylabel("DJC (°C·jour)")
    ax[0].set_title("DJC mensuel")
    ax[0].legend()

    ax[1].set_xticks(x)
    ax[1].set_xticklabels(list(mois_noms.values()), rotation=45)
    ax[1].set_ylabel("DJF (°C·jour)")
    ax[1].set_title("DJF mensuel")
    ax[1].legend()

    st.pyplot(fig)
    plt.close(fig)

    total_DJC = {key: df_DJC[key].sum() for key in data}
    total_DJF = {key: df_DJF[key].sum() for key in data}

    st.subheader("Sommes annuelles")
    st.write("DJC annuel :")
    for key in data:
        st.write(f"{file_names[key]} = {total_DJC[key]:.0f}")

    st.write("DJF annuel :")
    for key in data:
        st.write(f"{file_names[key]} = {total_DJF[key]:.0f}")
