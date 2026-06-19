"""
  01 — Distribuzione del target (squilibrio di classe)
  02 — Valori mancanti per colonna (dopo DataCleaning)
  03 — Matrice di correlazione (feature numeriche + binarie)
  04 — Feature continue vs target (boxplot per classe)
  05 — Feature categoriche vs target (stacked bar normalizzato)
  06 — Distribuzione geografica del danno (geo_level_1_id)
  07 — Distribuzione di age (istogramma con evidenza della skewness)
"""

import os
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_PALETTE_CLASSI = ["#4878CF", "#6ACC65", "#D65F5F"]
_FIGSIZE_MEDIUM  = (10, 6)
_FIGSIZE_LARGE   = (14, 8)

class Plotter:
    def __init__(self, dataframe: pd.DataFrame, output_dir: str = None):
        self.df = dataframe.copy()
        self.target = "damage_grade"
        self.output_dir = output_dir

        self.target_names = [
            "Danno Lieve (1)",
            "Danno Medio (2)",
            "Distruzione (3)",
        ]

        self.colonne_continue = [
            "age", "area_percentage", "height_percentage",
            "count_floors_pre_eq", "count_families",
        ]

        self.colonne_binarie = [c for c in self.df.columns if c.startswith("has_")]

        self.colonne_categoriche = [
            "land_surface_condition", "foundation_type", "roof_type",
            "ground_floor_type", "other_floor_type", "position",
            "plan_configuration", "legal_ownership_status",
        ]

        self.colonne_geo = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]

    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 1 — Distribuzione del target
    # ─────────────────────────────────────────────────────────────────────────

    def plot_distribuzione_target(self):
        """
        Countplot della variabile target `damage_grade`.

        Evidenzia lo squilibrio di classe caratteristico del dataset:
          Classe 1 (Danno Lieve)  ~  9.6%
          Classe 2 (Danno Medio)  ~ 56.9%
          Classe 3 (Distruzione)  ~ 33.5%
        """
        if self.target not in self.df.columns:
            print(f"  [Skip] Colonna target '{self.target}' non presente.")
            return

        conteggi   = self.df[self.target].value_counts().sort_index()
        percentuali = (conteggi / conteggi.sum() * 100).round(1)

        fig, ax = plt.subplots(figsize=_FIGSIZE_MEDIUM)
        barre = ax.bar(
            self.target_names,
            conteggi.values,
            color=_PALETTE_CLASSI,
            edgecolor="white",
            linewidth=1.5,
        )

        for barra, count, perc in zip(barre, conteggi.values, percentuali.values):
            ax.text(
                barra.get_x() + barra.get_width() / 2,
                barra.get_height() + conteggi.max() * 0.01,
                f"{count:,}\n({perc}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

        ax.set_title("Distribuzione delle classi di danno", fontsize=14, pad=14)
        ax.set_ylabel("Numero di edifici", fontsize=12)
        ax.set_xlabel("Classe di danno", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, conteggi.max() * 1.15)  # margine per le annotazioni

        plt.tight_layout()
        self._salva_o_mostra(fig, "01_distribuzione_target.png")

    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 2 — Valori mancanti per colonna
    # ─────────────────────────────────────────────────────────────────────────

    def plot_valori_mancanti(self):
        """
        Barplot orizzontale della percentuale di NaN per colonna.
        """
        pct_nulli = (self.df.isnull().sum() / len(self.df) * 100)
        pct_nulli = pct_nulli[pct_nulli > 0].sort_values(ascending=True)

        if pct_nulli.empty:
            print("  [Info] Nessun valore mancante nel dataset: plot saltato.")
            return

        fig, ax = plt.subplots(figsize=(10, max(4, len(pct_nulli) * 0.35)))

        colori = ["#D65F5F" if v > 40 else "#4878CF" for v in pct_nulli.values]
        barre  = ax.barh(pct_nulli.index, pct_nulli.values,
                         color=colori, edgecolor="white")

        for barra, val in zip(barre, pct_nulli.values):
            ax.text(
                val + 0.3, barra.get_y() + barra.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9,
            )

        ax.axvline(40, color="red", linestyle="--", alpha=0.6,
                   label="Soglia eliminazione (40%)")

        ax.set_title("Valori mancanti per colonna", fontsize=14, pad=14)
        ax.set_xlabel("Percentuale di NaN (%)", fontsize=12)
        ax.grid(axis="x", alpha=0.3)
        ax.legend(loc="lower right")

        plt.tight_layout()
        self._salva_o_mostra(fig, "02_valori_mancanti.png")

    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 3 — Matrice di correlazione
    # ─────────────────────────────────────────────────────────────────────────

    def plot_correlazioni(self, soglia_annotazione: float = 0.3):
        """
        Heatmap di correlazione di Pearson tra feature numeriche e target.
        Le feature geografiche ad alta cardinalità (geo_level_*_id) vengono
        temporaneamente Target-Encoded rispetto al target prima di calcolare
        la correlazione, in modo da rendere il calcolo di Pearson statisticamente valido.
        """
        df_corr = self.df.copy()

        geo_cols = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
        geo_cols_presenti = [c for c in geo_cols if c in df_corr.columns]

        if geo_cols_presenti and self.target in df_corr.columns:
            # Controlla se le colonne geografiche sono già state codificate
            da_codificare = [
                c for c in geo_cols_presenti
                if not pd.api.types.is_float_dtype(df_corr[c])
            ]
            if da_codificare:
                from sklearn.preprocessing import TargetEncoder
                print(f"  [Plotter] Target Encoding temporaneo per le correlazioni geografiche: {da_codificare}")
                
                # Convertiamo le colonne in float per evitare il LossySetitemError di pandas
                for c in da_codificare:
                    df_corr[c] = df_corr[c].astype(float)
                
                encoder = TargetEncoder(
                    categories='auto',
                    target_type='continuous',
                    random_state=42
                )
                mask_notna = df_corr[self.target].notna()
                if mask_notna.any():
                    encoded_vals = encoder.fit_transform(
                        df_corr.loc[mask_notna, da_codificare],
                        df_corr.loc[mask_notna, self.target]
                    )
                    df_corr.loc[mask_notna, da_codificare] = encoded_vals

        colonne_da_correlare = [
            c for c in (self.colonne_continue + geo_cols_presenti + self.colonne_binarie + [self.target])
            if c in df_corr.columns
        ]

        if len(colonne_da_correlare) < 2:
            print("  [Skip] Meno di 2 colonne numeriche disponibili.")
            return

        matrice_corr = df_corr[colonne_da_correlare].corr()
        mask = np.triu(np.ones_like(matrice_corr, dtype=bool))

        annot_matrix = matrice_corr.round(2).astype(str)
        annot_matrix = annot_matrix.where(matrice_corr.abs() >= soglia_annotazione, "")

        n_cols = len(colonne_da_correlare)
        tick_fontsize = max(6, 10 - n_cols // 5)

        fig, ax = plt.subplots(figsize=(14, 11))
        sns.heatmap(
            matrice_corr,
            mask=mask,
            annot=annot_matrix,
            fmt="",
            cmap="RdBu_r",
            center=0,
            vmin=-1, vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.7, "label": "Coefficiente di correlazione"},
            annot_kws={"size": 8},
            ax=ax,
        )
        ax.set_title(
            f"Matrice di correlazione di Pearson  (annotazioni per |r| ≥ {soglia_annotazione})",
            fontsize=13, pad=14,
        )
        plt.xticks(rotation=45, ha="right", fontsize=tick_fontsize)
        plt.yticks(rotation=0, fontsize=tick_fontsize)

        plt.tight_layout()
        self._salva_o_mostra(fig, "03_correlazioni.png")

        if self.target in matrice_corr.columns:
            corr_target = (matrice_corr[self.target]
                           .drop(self.target)
                           .abs()
                           .sort_values(ascending=False))
            print(f"\n  Top 10 feature più correlate con '{self.target}' (|r| Pearson):")
            for nome, val in corr_target.head(10).items():
                print(f"    {nome:<45} {val:.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 4 — Feature continue vs target (boxplot)
    # ─────────────────────────────────────────────────────────────────────────

    def plot_numeriche_vs_target(self):
        """
        Griglia di boxplot: una feature continua per pannello, per classe di danno.
        """
        if self.target not in self.df.columns:
            print(f"  [Skip] Colonna target '{self.target}' non presente.")
            return

        continue_presenti = [c for c in self.colonne_continue if c in self.df.columns]
        if not continue_presenti:
            print("  [Skip] Nessuna feature continua presente.")
            return

        n = len(continue_presenti)
        ncols = 2
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
        axes = axes.flatten() if n > 1 else [axes]

        for i, colonna in enumerate(continue_presenti):
            ax = axes[i]
            sns.boxplot(
                data=self.df,
                x=self.target,
                y=colonna,
                hue=self.target,
                palette=_PALETTE_CLASSI,
                ax=ax,
                legend=False,
                showfliers=False,   
            )
            ax.set_title(f"{colonna} per classe di danno", fontsize=11)
            ax.set_xlabel("Classe di danno")
            ax.set_ylabel(colonna)
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(["1 Lieve", "2 Medio", "3 Distr."])
            ax.grid(axis="y", alpha=0.3)

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            "Distribuzione delle feature continue per classe di danno  (outlier nascosti)",
            fontsize=14, y=1.01,
        )
        plt.tight_layout()
        self._salva_o_mostra(fig, "04_numeriche_vs_target.png")

    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 5 — Feature categoriche vs target (stacked bar) [NUOVO]
    # ─────────────────────────────────────────────────────────────────────────

    def plot_categoriche_vs_target(self):
        """
        Griglia di stacked bar chart normalizzati: una feature categorica per pannello.
        """
        cat_presenti = [c for c in self.colonne_categoriche if c in self.df.columns]
        if not cat_presenti or self.target not in self.df.columns:
            print("  [Skip] Colonne categoriche o target non presenti "
                  "(usare il DataFrame grezzo, prima di DataEncoding).")
            return

        n = len(cat_presenti)
        ncols = 2
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
        axes = axes.flatten() if n > 1 else [axes]

        for i, colonna in enumerate(cat_presenti):
            ax = axes[i]

            tabella = pd.crosstab(
                self.df[colonna],
                self.df[self.target],
                normalize="index",
            ) * 100

            tabella.plot(
                kind="bar",
                stacked=True,
                color=_PALETTE_CLASSI,
                ax=ax,
                width=0.8,
                edgecolor="white",
                legend=False,
            )
            ax.set_title(colonna, fontsize=11)
            ax.set_xlabel("")
            ax.set_ylabel("% edifici")
            ax.set_ylim(0, 100)
            ax.tick_params(axis="x", rotation=0)
            ax.grid(axis="y", alpha=0.3)

        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in _PALETTE_CLASSI]
        fig.legend(handles, self.target_names, title="Classe di danno",
                   loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            "Distribuzione delle classi di danno per feature categorica  (% normalizzata per riga)",
            fontsize=14, y=1.01,
        )
        plt.tight_layout()
        self._salva_o_mostra(fig, "05_categoriche_vs_target.png")

    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 6 — Distribuzione geografica del danno
    # ─────────────────────────────────────────────────────────────────────────

    def plot_geografia_danno(self):
        """
        Analisi del danno per geo_level_1_id: la divisione geografica più ampia.
        """
        if "geo_level_1_id" not in self.df.columns or self.target not in self.df.columns:
            print("  [Skip] Colonne geografiche o target non presenti.")
            return

        fig, axes = plt.subplots(2, 1, figsize=_FIGSIZE_LARGE)

        conteggi_regione = self.df["geo_level_1_id"].value_counts().sort_index()
        axes[0].bar(
            conteggi_regione.index.astype(str),
            conteggi_regione.values,
            color="#5B9BD5", edgecolor="white",
        )
        axes[0].set_title("Numero di edifici per regione (geo_level_1_id)", fontsize=12)
        axes[0].set_xlabel("geo_level_1_id")
        axes[0].set_ylabel("Numero di edifici")
        axes[0].grid(axis="y", alpha=0.3)

        tabella = pd.crosstab(
            self.df["geo_level_1_id"],
            self.df[self.target],
            normalize="index",
        ) * 100

        tabella.plot(
            kind="bar",
            stacked=True,
            color=_PALETTE_CLASSI,
            ax=axes[1],
            width=0.9,
            edgecolor="white",
        )
        axes[1].set_title("Distribuzione % delle classi di danno per regione", fontsize=12)
        axes[1].set_xlabel("geo_level_1_id")
        axes[1].set_ylabel("Percentuale (%)")
        axes[1].legend(self.target_names, title="Classe",
                       loc="center left", bbox_to_anchor=(1.01, 0.5))
        axes[1].set_ylim(0, 100)

        # Con più di 15 regioni le etichette si sovrappongono senza rotazione
        if len(tabella) > 15:
            axes[1].tick_params(axis="x", rotation=45)
            axes[0].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        self._salva_o_mostra(fig, "06_geografia_danno.png")

    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 7 — Distribuzione di age (istogramma + KDE) [NUOVO]
    # ─────────────────────────────────────────────────────────────────────────

    def plot_distribuzione_age(self):
        """
        Istogramma con curva KDE della variabile `age`, separato per classe di danno.
        """
        if "age" not in self.df.columns:
            print("  [Skip] Colonna 'age' non presente.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        soglia_visualizzazione = 200
        df_clip = self.df[self.df["age"] <= soglia_visualizzazione]
        pct_esclusi = (len(self.df) - len(df_clip)) / len(self.df) * 100

        for classe, colore, nome in zip([1, 2, 3], _PALETTE_CLASSI, self.target_names):
            subset = df_clip[df_clip[self.target] == classe]["age"] if self.target in df_clip.columns else df_clip["age"]
            axes[0].hist(
                subset,
                bins=40,
                alpha=0.5,
                color=colore,
                label=nome if self.target in df_clip.columns else "age",
                density=True,   # normalizza a densità per confronto tra classi sbilanciate
            )

        axes[0].set_title(
            f"Distribuzione di age per classe  (range 0-{soglia_visualizzazione}, "
            f"escluso {pct_esclusi:.1f}% coda)",
            fontsize=11,
        )
        axes[0].set_xlabel("Età edificio (anni)")
        axes[0].set_ylabel("Densità")
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.3)

        axes[1].boxplot(
            [self.df[self.df[self.target] == k]["age"].values
             if self.target in self.df.columns else self.df["age"].values
             for k in [1, 2, 3]],
            labels=["1 Lieve", "2 Medio", "3 Distr."],
            patch_artist=True,
            boxprops=dict(facecolor="#DDEEFF"),
            medianprops=dict(color="navy", linewidth=2),
            showfliers=False,   # nasconde outlier estremi (age=995)
        )
        skew_val = self.df["age"].skew()
        axes[1].set_title(
            f"Boxplot age per classe  (outlier nascosti)\nskewness totale = {skew_val:.2f}",
            fontsize=11,
        )
        axes[1].set_xlabel("Classe di danno")
        axes[1].set_ylabel("Età edificio (anni)")
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        self._salva_o_mostra(fig, "07_distribuzione_age.png")

    def esegui_tutto(self):
        """
        Esegue tutte le analisi esplorative in sequenza.

        Chiamare con il DataFrame GREZZO (prima di DataEncoding), poiché
        plot_categoriche_vs_target() richiede le colonne stringa originali.
        """
        sep = "=" * 62
        print(f"\n{sep}")
        print("  ANALISI ESPLORATIVA DEI DATI")
        print(sep)
        print(f"  {'Righe:':<40} {self.df.shape[0]:>8,}")
        print(f"  {'Colonne:':<40} {self.df.shape[1]:>8}")
        print(f"  {'Output dir:':<40} {self.output_dir or 'interattivo':>8}")
        print(sep)

        steps = [
            ("[1/7] Distribuzione del target...",          self.plot_distribuzione_target),
            ("[2/7] Valori mancanti per colonna...",       self.plot_valori_mancanti),
            ("[3/7] Matrice di correlazione...",           self.plot_correlazioni),
            ("[4/7] Feature continue vs target...",        self.plot_numeriche_vs_target),
            ("[5/7] Feature categoriche vs target...",     self.plot_categoriche_vs_target),
            ("[6/7] Distribuzione geografica del danno...",self.plot_geografia_danno),
            ("[7/7] Distribuzione di age...",              self.plot_distribuzione_age),
        ]

        for label, metodo in steps:
            print(f"\n  {label}")
            metodo()

        print(f"\n{sep}")
        print("  Analisi completata.")
        if self.output_dir:
            print(f"  Grafici salvati in: {self.output_dir}")
        print(sep)

    def _salva_o_mostra(self, fig: plt.Figure, nome_file: str):
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            percorso = os.path.join(self.output_dir, nome_file)
            fig.savefig(percorso, dpi=150, bbox_inches="tight")
            print(f"    -> Salvato: {percorso}")
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)
