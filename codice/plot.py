import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Palette coerente con evaluation.py: blu (classe 1), verde (classe 2), rosso (classe 3)
_PALETTE_CLASSI = ["#4878CF", "#6ACC65", "#D65F5F"]
_FIGSIZE_MEDIUM = (10, 6)
_FIGSIZE_LARGE = (14, 8)


class Plotter:
    """
    Classe per l'analisi esplorativa del dataset Richter's Predictor.

    Riceve in input il dataframe di train (con `damage_grade` incluso) e produce
    una serie di grafici che aiutano a comprendere i dati prima del modellamento.
    """

    def __init__(self, dataframe: pd.DataFrame, output_dir: str = None):
        # Copia difensiva per evitare modifiche involontarie al dataframe originale
        self.df = dataframe.copy()
        self.target = "damage_grade"
        self.output_dir = output_dir

        # Nomi leggibili per le tre classi (usati nelle legende e negli assi)
        self.target_names = [
            "Danno Lieve (1)",
            "Danno Medio (2)",
            "Distruzione (3)",
        ]

        # Classificazione delle colonne in gruppi semantici.
        # Questo permette di applicare plot diversi a seconda del tipo di feature.
        self.colonne_continue = [
            "age",
            "area_percentage",
            "height_percentage",
            "count_floors_pre_eq",
            "count_families",
        ]

        self.colonne_binarie = [c for c in self.df.columns if c.startswith("has_")]

        self.colonne_categoriche = [
            "land_surface_condition", "foundation_type", "roof_type",
            "ground_floor_type", "other_floor_type", "position",
            "plan_configuration", "legal_ownership_status",
        ]

        self.colonne_geo = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]

    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 1: Distribuzione del target
    # ─────────────────────────────────────────────────────────────────────────

    def plot_distribuzione_target(self):
        """
        Countplot del target `damage_grade`.

        Mostra lo squilibrio di classe caratteristico del dataset:
        tipicamente classe 1 ~10%, classe 2 ~57%, classe 3 ~33%.
        Giustifica le scelte di metrica (micro-F1) e di gestione dello sbilanciamento.
        """
        if self.target not in self.df.columns:
            print(f"  [Skip] Colonna target '{self.target}' non presente.")
            return

        # Conteggi assoluti e percentuali per ciascuna classe
        conteggi = self.df[self.target].value_counts().sort_index()
        percentuali = (conteggi / conteggi.sum() * 100).round(1)

        fig, ax = plt.subplots(figsize=_FIGSIZE_MEDIUM)
        barre = ax.bar(
            self.target_names,
            conteggi.values,
            color=_PALETTE_CLASSI,
            edgecolor="white",
            linewidth=1.5,
        )

        # Annotazione sopra ogni barra: conteggio assoluto + percentuale
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
        # Margine superiore per far spazio alle annotazioni
        ax.set_ylim(0, conteggi.max() * 1.15)

        plt.tight_layout()
        self._salva_o_mostra(fig, "01_distribuzione_target.png")

    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 2: Valori mancanti per colonna
    # ─────────────────────────────────────────────────────────────────────────

    def plot_valori_mancanti(self):
        """
        Barplot orizzontale della percentuale di valori mancanti per colonna.

        Nota: sul dataset grezzo di Richter non ci sono NaN, ma dopo
        `pulisci_variabili()` del modulo DataCleaning gli outlier vengono
        convertiti in NaN. Questo plot è quindi più informativo se eseguito
        dopo quel passaggio.
        """
        # Percentuale di NaN per ogni colonna, ordinata decrescente
        pct_nulli = (self.df.isnull().sum() / len(self.df) * 100)
        pct_nulli = pct_nulli[pct_nulli > 0].sort_values(ascending=True)

        if pct_nulli.empty:
            print("  [Info] Nessun valore mancante nel dataset: plot saltato.")
            return

        fig, ax = plt.subplots(figsize=(10, max(4, len(pct_nulli) * 0.35)))

        # Colora in rosso le colonne oltre soglia 40% (candidate alla rimozione)
        colori = ["#D65F5F" if v > 40 else "#4878CF" for v in pct_nulli.values]

        barre = ax.barh(pct_nulli.index, pct_nulli.values, color=colori, edgecolor="white")

        # Etichetta con il valore percentuale a fianco di ogni barra
        for barra, val in zip(barre, pct_nulli.values):
            ax.text(
                val + 0.3, barra.get_y() + barra.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9,
            )

        # Linea verticale di riferimento alla soglia di eliminazione (40%)
        ax.axvline(40, color="red", linestyle="--", alpha=0.6,
                   label="Soglia eliminazione (40%)")

        ax.set_title("Valori mancanti per colonna", fontsize=14, pad=14)
        ax.set_xlabel("Percentuale di NaN (%)", fontsize=12)
        ax.grid(axis="x", alpha=0.3)
        ax.legend(loc="lower right")

        plt.tight_layout()
        self._salva_o_mostra(fig, "02_valori_mancanti.png")

    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 3: Matrice di correlazione
    # ─────────────────────────────────────────────────────────────────────────

    def plot_correlazioni(self, soglia_annotazione: float = 0.3):
        """
        Heatmap di correlazione tra le feature numeriche (continue + binarie).

        Parameters
        ----------
        soglia_annotazione : float
            Valore sotto il quale le correlazioni NON vengono annotate nella cella,
            per evitare di sovraccaricare visivamente la heatmap.

        Obiettivo: individuare feature ridondanti (correlazione alta tra loro)
        e feature poco correlate col target (candidate all'eliminazione).
        """
        # Seleziona solo le colonne numeriche, escludendo gli ID geografici
        # (hanno valori categorici codificati come interi, la correlazione
        # lineare non è semanticamente sensata)
        colonne_da_correlare = [
            c for c in (self.colonne_continue + self.colonne_binarie + [self.target])
            if c in self.df.columns
        ]

        if len(colonne_da_correlare) < 2:
            print("  [Skip] Meno di 2 colonne numeriche disponibili per la correlazione.")
            return

        matrice_corr = self.df[colonne_da_correlare].corr()

        # Maschera triangolare superiore: mostriamo solo la metà inferiore
        # per evitare la ridondanza di una matrice simmetrica
        mask = np.triu(np.ones_like(matrice_corr, dtype=bool))

        # Costruisce gli annot solo per le correlazioni oltre soglia (più leggibile)
        annot_matrix = matrice_corr.round(2).astype(str)
        annot_matrix = annot_matrix.where(matrice_corr.abs() >= soglia_annotazione, "")

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
            ax=ax,
        )
        ax.set_title(
            f"Matrice di correlazione (annotazioni per |r| >= {soglia_annotazione})",
            fontsize=14, pad=14,
        )
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.tight_layout()
        self._salva_o_mostra(fig, "03_correlazioni.png")

        # Stampa testuale delle correlazioni più forti col target
        if self.target in matrice_corr.columns:
            corr_target = matrice_corr[self.target].drop(self.target).abs().sort_values(ascending=False)
            print(f"\n  Top 10 feature più correlate (in valore assoluto) con '{self.target}':")
            for nome, val in corr_target.head(10).items():
                print(f"    {nome:<40} {val:>6.3f}")

    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 4: Feature numeriche vs target (boxplot)
    # ─────────────────────────────────────────────────────────────────────────

    def plot_numeriche_vs_target(self):
        """
        Griglia di boxplot: una feature continua per pannello, suddivisa per classe di danno.

        Permette di vedere visivamente se una feature ha potere discriminante:
        se le distribuzioni delle tre classi sono simili, la feature è poco utile.
        """
        if self.target not in self.df.columns:
            print(f"  [Skip] Colonna target '{self.target}' non presente.")
            return

        continue_presenti = [c for c in self.colonne_continue if c in self.df.columns]
        if not continue_presenti:
            print("  [Skip] Nessuna feature continua presente.")
            return

        # Griglia dinamica: 2 colonne fisse, righe quante ne servono
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
                # showfliers=False evita che gli outlier estremi schiaccino il grafico
                # (es. age può arrivare a 995 che è un valore sentinella)
                showfliers=False,
            )
            ax.set_title(f"{colonna} per classe di danno", fontsize=11)
            ax.set_xlabel("Classe di danno")
            ax.set_ylabel(colonna)
            ax.set_xticks(range(3))
            ax.set_xticklabels(["1 Lieve", "2 Medio", "3 Distr."])
            ax.grid(axis="y", alpha=0.3)

        # Nasconde eventuali pannelli vuoti (quando n è dispari)
        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            "Distribuzione delle feature continue per classe di danno (outlier nascosti)",
            fontsize=14, y=1.00,
        )
        plt.tight_layout()
        self._salva_o_mostra(fig, "04_numeriche_vs_target.png")

    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 5: Distribuzione geografica del danno
    # ─────────────────────────────────────────────────────────────────────────

    def plot_geografia_danno(self):
        """
        Analisi del danno per geo_level_1_id: la regione geografica più ampia.

        Produce due subplot:
          a) Numero di edifici per regione
          b) Distribuzione delle classi di danno per regione (proporzioni stacked)

        Utile per capire se alcune zone sono state colpite più duramente di altre.
        """
        if "geo_level_1_id" not in self.df.columns or self.target not in self.df.columns:
            print("  [Skip] Colonne geografiche o target non presenti.")
            return

        fig, axes = plt.subplots(2, 1, figsize=_FIGSIZE_LARGE)

        # ── (a) Numero di edifici per regione ──────────────────────────────
        conteggi_regione = self.df["geo_level_1_id"].value_counts().sort_index()
        axes[0].bar(
            conteggi_regione.index,
            conteggi_regione.values,
            color="#5B9BD5", edgecolor="white",
        )
        axes[0].set_title("Numero di edifici per regione (geo_level_1_id)", fontsize=12)
        axes[0].set_xlabel("geo_level_1_id")
        axes[0].set_ylabel("Numero di edifici")
        axes[0].grid(axis="y", alpha=0.3)

        # ── (b) Proporzioni di danno per regione (stacked bar normalizzato) ──
        # crosstab normalizzato per riga: ogni regione somma a 100%
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
        axes[1].legend(self.target_names, title="Classe", loc="center left",
                       bbox_to_anchor=(1.01, 0.5))
        axes[1].set_ylim(0, 100)
        # Le tick label sono numeri di regione: ruotiamo solo se sono tante
        if len(tabella) > 15:
            axes[1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        self._salva_o_mostra(fig, "05_geografia_danno.png")

    # ─────────────────────────────────────────────────────────────────────────
    # METODO AGGREGATORE
    # ─────────────────────────────────────────────────────────────────────────

    def esegui_tutto(self):
        """
        Esegue tutte le analisi esplorative in sequenza.

        Utile da chiamare in un'unica riga dal main dopo aver caricato i dati grezzi.
        """
        sep = "=" * 62
        print(f"\n{sep}")
        print("  ANALISI ESPLORATIVA DEI DATI ")
        print(sep)
        print(f"  {'Righe:':<40} {self.df.shape[0]:>8,}")
        print(f"  {'Colonne:':<40} {self.df.shape[1]:>8}")
        print(sep)

        print("\n  [1/5] Distribuzione del target...")
        self.plot_distribuzione_target()

        print("\n  [2/5] Valori mancanti per colonna...")
        self.plot_valori_mancanti()

        print("\n  [3/5] Matrice di correlazione...")
        self.plot_correlazioni()

        print("\n  [4/5] Feature continue per classe di danno...")
        self.plot_numeriche_vs_target()

        print("\n  [5/5] Distribuzione geografica del danno...")
        self.plot_geografia_danno()

        print(f"\n{sep}")
        print("  Analisi completata.")
        print(sep)

    # ─────────────────────────────────────────────────────────────────────────
    # Metodo privato: salva il grafico su disco oppure lo mostra a schermo
    # ─────────────────────────────────────────────────────────────────────────

    def _salva_o_mostra(self, fig: plt.Figure, nome_file: str):
        """
        Salva il grafico nella cartella `output_dir` (se specificata),
        altrimenti lo mostra direttamente a schermo.

        Stessa logica usata nel modulo evaluation.py per coerenza.
        """
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            percorso = os.path.join(self.output_dir, nome_file)
            fig.savefig(percorso, dpi=150, bbox_inches="tight")
            print(f"    [Grafico salvato] -> {percorso}")
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)
