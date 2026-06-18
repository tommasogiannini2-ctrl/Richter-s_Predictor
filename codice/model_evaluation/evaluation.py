"""
evaluation.py
=============
Modulo di valutazione delle prestazioni del modello per il progetto Richter's Predictor.

La metrica ufficiale della competizione DrivenData è la **micro-averaged F1 score**,
calcolata su tre classi ordinate:
    1 = danno lieve
    2 = danno medio
    3 = distruzione quasi totale

Struttura del modulo:
    ModelEvaluator
    ├── calcola_metriche()          → resoconto testuale + dizionario risultati
    ├── salva_metriche_csv()        → persistenza su disco dei risultati
    ├── plot_confusion_matrix()     → heatmap della matrice di confusione
    ├── plot_roc_curves()           → curve ROC one-vs-rest per ogni classe
    ├── plot_class_report()         → bar chart Precision / Recall / F1 per classe
    ├── plot_distribuzione_classi() → confronto distribuzione reale vs predetta
    └── valuta_tutto()              → esegue tutte le valutazioni in sequenza

Uso tipico:
    evaluator = ModelEvaluator(y_true, y_pred, y_score, output_dir="output/eval")
    metriche  = evaluator.valuta_tutto()
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    recall_score,
    precision_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


# ---------------------------------------------------------------------------
# Costanti di stile condivise tra i grafici
# ---------------------------------------------------------------------------
_PALETTE      = ["#4878CF", "#6ACC65", "#D65F5F"]   # blu / verde / rosso (una per classe)
_FIGSIZE_MED  = (10, 7)
_FIGSIZE_WIDE = (12, 6)


# ===========================================================================
# Classe principale
# ===========================================================================

class ModelEvaluator:
    """
    Valutatore delle prestazioni di un classificatore multi-classe (3 classi).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Etichette reali. Devono contenere esattamente i valori {1, 2, 3}.
    y_pred : array-like of shape (n_samples,)
        Etichette predette dal modello.
    y_score : array-like of shape (n_samples, 3), optional
        Probabilità di appartenenza a ciascuna classe (output di predict_proba).
        Richiesto solo per le curve ROC.
    target_names : list of str, optional
        Nomi leggibili delle classi. Default: ["Danno Lieve (1)", ...].
    output_dir : str, optional
        Cartella in cui salvare i grafici PNG.
        Se None, i grafici vengono mostrati a schermo (plt.show).
    """

    def __init__(self, y_true, y_pred, y_score=None, target_names=None, output_dir=None):

        # --- Conversione in numpy array e validazione base ---
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)

        # Verifica che y_true e y_pred abbiano la stessa dimensione
        if len(self.y_true) != len(self.y_pred):
            raise ValueError(
                f"y_true ({len(self.y_true)} elem.) e y_pred ({len(self.y_pred)} elem.) "
                "devono avere la stessa lunghezza."
            )

        # Gestione y_score: se fornito, convertiamo e verifichiamo la shape
        if y_score is not None:
            self.y_score = np.asarray(y_score)
            if self.y_score.ndim != 2:
                raise ValueError("y_score deve essere un array 2D di shape (n_samples, n_classes).")
        else:
            self.y_score = None

        # Ricava le classi uniche in ordine crescente dal vettore reale
        self.classes   = np.unique(self.y_true)
        self.n_classes = len(self.classes)

        # Nomi delle classi usati nelle etichette dei grafici
        self.target_names = target_names if target_names is not None else [
            "Danno Lieve (1)",
            "Danno Medio (2)",
            "Distruzione (3)",
        ]

        # Verifica coerenza tra numero di classi e nomi forniti
        if len(self.target_names) != self.n_classes:
            raise ValueError(
                f"target_names ha {len(self.target_names)} elementi "
                f"ma y_true contiene {self.n_classes} classi distinte."
            )

        self.output_dir = output_dir

    # -----------------------------------------------------------------------
    # METRICA PRINCIPALE: micro-F1
    # -----------------------------------------------------------------------

    def micro_f1(self) -> float:
        """
        Calcola la micro-averaged F1 score (metrica ufficiale DrivenData).

        La micro-F1 aggrega i conteggi di TP, FP e FN su TUTTE le classi
        prima di calcolare precision e recall, dando uguale peso a ogni
        singola predizione indipendentemente dalla classe di appartenenza.

        Su dataset sbilanciati (come questo, dove la classe 2 domina),
        la micro-F1 è più informativa della macro-F1 perché non assegna
        uguale importanza a classi con supporto molto diverso.

        Returns
        -------
        float : valore tra 0 e 1 (1 = predizioni perfette).
        """
        return f1_score(self.y_true, self.y_pred, average="micro")

    # -----------------------------------------------------------------------
    # METRICHE TESTUALI
    # -----------------------------------------------------------------------

    def calcola_metriche(self) -> dict:
        """
        Calcola e stampa un resoconto completo delle metriche di classificazione.

        Metriche calcolate
        ------------------
        Aggregate:
            - Micro-F1  : metrica ufficiale della competizione
            - Macro-F1  : media non pesata della F1 per classe (utile per
                          valutare l'equità tra classi)
            - Accuracy  : % predizioni corrette (fuorviante su dataset sbilanciati,
                          inclusa per completezza)
            - G-Mean    : sqrt(TPR_medio × TNR_medio) — penalizza i modelli che
                          sacrificano le classi minoritarie

        Per ogni classe:
            - TPR (Recall / Sensibilità) : quota di positivi reali correttamente
                                           identificati. Critico per la classe 1
                                           (danno lieve, molto rara).
            - TNR (Specificità)          : quota di negativi reali correttamente
                                           rigettati.
            - Precision                  : quota di positivi predetti effettivamente
                                           corretti.
            - F1-score                   : media armonica di Precision e Recall.
            - Supporto                   : numero di campioni reali per classe.

        Returns
        -------
        dict con chiavi: micro_f1, macro_f1, accuracy, g_mean, per_classe.
        """
        # --- Matrice di confusione (base per tutti i calcoli manuali) ---
        cm = confusion_matrix(self.y_true, self.y_pred, labels=self.classes)

        # --- Metriche aggregate ---
        micro_f1 = f1_score(self.y_true, self.y_pred, average="micro")
        macro_f1 = f1_score(self.y_true, self.y_pred, average="macro")
        accuracy = np.diag(cm).sum() / cm.sum()   # equivalente a accuracy_score

        # --- Metriche per singola classe (calcolate dalla CM) ---
        tpr_list, tnr_list, prec_list, f1_list = [], [], [], []
        risultati_per_classe = {}

        for i, cls in enumerate(self.classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp        # riga i, colonne diverse da i
            fp = cm[:, i].sum() - tp        # colonna i, righe diverse da i
            tn = cm.sum() - tp - fn - fp    # tutto il resto

            # Calcolo con protezione dalla divisione per zero
            tpr  = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # Recall
            tnr  = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # Specificità
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0   # Precision
            f1   = (2 * prec * tpr) / (prec + tpr) if (prec + tpr) > 0 else 0.0

            tpr_list.append(tpr)
            tnr_list.append(tnr)
            prec_list.append(prec)
            f1_list.append(f1)

            # Salviamo il dizionario con chiave = valore della classe (es. 1, 2, 3)
            risultati_per_classe[cls] = {
                "TPR (Recall)": tpr,
                "TNR (Spec.)":  tnr,
                "Precision":    prec,
                "F1-score":     f1,
                "Supporto":     int(cm[i, :].sum()),
            }

        # Geometric Mean: sqrt(TPR_medio * TNR_medio)
        # Un G-Mean elevato indica che il modello è bilanciato su tutte le classi,
        # non solo su quella dominante.
        g_mean = np.sqrt(np.mean(tpr_list) * np.mean(tnr_list))

        # --- Stampa del resoconto ---
        self._stampa_resoconto(
            micro_f1, macro_f1, accuracy, g_mean,
            risultati_per_classe, tpr_list, tnr_list, prec_list
        )

        # --- Classification report dettagliato di sklearn (complementare) ---
        print("\n  CLASSIFICATION REPORT (sklearn):")
        print(
            classification_report(
                self.y_true, self.y_pred,
                labels=self.classes,
                target_names=self.target_names,
            )
        )

        return {
            "micro_f1":   micro_f1,
            "macro_f1":   macro_f1,
            "accuracy":   accuracy,
            "g_mean":     g_mean,
            "per_classe": risultati_per_classe,
        }

    def _stampa_resoconto(
        self, micro_f1, macro_f1, accuracy, g_mean,
        risultati_per_classe, tpr_list, tnr_list, prec_list
    ):
        """
        Stampa formattata del resoconto metriche.
        Separata da calcola_metriche per leggibilità e testabilità.
        """
        sep = "=" * 62
        print(f"\n{sep}")
        print("  RESOCONTO VALUTAZIONE MODELLO")
        print(sep)
        print(f"  {'Micro-F1 (metrica ufficiale):':<35} {micro_f1:.4f}")
        print(f"  {'Macro-F1:':<35} {macro_f1:.4f}")
        print(f"  {'Accuracy:':<35} {accuracy:.4f}")
        print(f"  {'Geometric Mean (TPR × TNR):':<35} {g_mean:.4f}")
        print(sep)

        header = f"  {'CLASSE':<22} {'TPR':>8} {'TNR':>8} {'Prec':>8} {'F1':>8} {'N':>8}"
        print(header)
        print(f"  {'-' * 60}")

        for i, cls in enumerate(self.classes):
            r    = risultati_per_classe[cls]
            nome = self.target_names[i]
            print(
                f"  {nome:<22} "
                f"{r['TPR (Recall)']:>8.4f} "
                f"{r['TNR (Spec.)']:>8.4f} "
                f"{r['Precision']:>8.4f} "
                f"{r['F1-score']:>8.4f} "
                f"{r['Supporto']:>8d}"
            )

        print(f"  {'-' * 60}")
        print(
            f"  {'Media Macro':<22} "
            f"{np.mean(tpr_list):>8.4f} "
            f"{np.mean(tnr_list):>8.4f} "
            f"{np.mean(prec_list):>8.4f} "
            f"{macro_f1:>8.4f}"
        )
        print(sep)

    # -----------------------------------------------------------------------
    # PERSISTENZA: salva le metriche su CSV
    # -----------------------------------------------------------------------

    def salva_metriche_csv(self, metriche: dict, nome_file: str = "metriche.csv"):
        """
        Salva il dizionario delle metriche in un file CSV nella output_dir.

        Genera due sezioni:
            - metriche aggregate (micro_f1, macro_f1, accuracy, g_mean)
            - metriche per classe (una riga per classe)

        Parameters
        ----------
        metriche  : dict restituito da calcola_metriche()
        nome_file : str, nome del file CSV da creare
        """
        if self.output_dir is None:
            print("  [Avviso] output_dir non impostato: salvataggio CSV saltato.")
            return

        os.makedirs(self.output_dir, exist_ok=True)

        # --- Sezione 1: metriche aggregate ---
        righe_aggregate = [
            {"Metrica": "Micro-F1",   "Valore": metriche["micro_f1"]},
            {"Metrica": "Macro-F1",   "Valore": metriche["macro_f1"]},
            {"Metrica": "Accuracy",   "Valore": metriche["accuracy"]},
            {"Metrica": "G-Mean",     "Valore": metriche["g_mean"]},
        ]

        # --- Sezione 2: metriche per classe ---
        righe_per_classe = []
        for i, cls in enumerate(self.classes):
            r = metriche["per_classe"][cls]
            righe_per_classe.append({
                "Metrica":  self.target_names[i],
                "TPR":      round(r["TPR (Recall)"], 4),
                "TNR":      round(r["TNR (Spec.)"],  4),
                "Precision":round(r["Precision"],    4),
                "F1":       round(r["F1-score"],     4),
                "Supporto": r["Supporto"],
            })

        percorso = os.path.join(self.output_dir, nome_file)
        with open(percorso, "w") as f:
            # Scrivi metriche aggregate
            pd.DataFrame(righe_aggregate).to_csv(f, index=False)
            f.write("\n")
            # Scrivi metriche per classe
            pd.DataFrame(righe_per_classe).to_csv(f, index=False)

        print(f"  [CSV salvato] -> {percorso}")

    # -----------------------------------------------------------------------
    # GRAFICO 1: Matrice di Confusione
    # -----------------------------------------------------------------------

    def plot_confusion_matrix(self, normalizza: bool = False):
        """
        Disegna la matrice di confusione come heatmap.

        La diagonale principale rappresenta le predizioni corrette.
        Gli elementi fuori diagonale sono gli errori:
            - Riga i, colonna j (i≠j): edifici di classe i predetti come classe j.

        Parameters
        ----------
        normalizza : bool
            Se True, ogni riga è divisa per il totale reale della classe
            (frequenze relative %). Utile per dataset sbilanciati dove i
            conteggi assoluti favoriscono visivamente la classe dominante.
        """
        cm = confusion_matrix(self.y_true, self.y_pred, labels=self.classes)

        if normalizza:
            # Normalizzazione per riga: cm[i,j] / Σ_j cm[i,j]
            cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fmt    = ".2%"
            titolo = "Matrice di Confusione — normalizzata per riga"
            nome_file = "confusion_matrix_norm.png"
        else:
            cm_plot   = cm
            fmt       = "d"
            titolo    = "Matrice di Confusione — conteggi assoluti"
            nome_file = "confusion_matrix.png"

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm_plot,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=self.target_names,
            yticklabels=self.target_names,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(titolo, fontsize=14, pad=14)
        ax.set_ylabel("Classe Reale",    fontsize=12)
        ax.set_xlabel("Classe Predetta", fontsize=12)
        ax.tick_params(axis="x", rotation=15)
        ax.tick_params(axis="y", rotation=0)

        plt.tight_layout()
        self._salva_o_mostra(fig, nome_file)

    # -----------------------------------------------------------------------
    # GRAFICO 2: Curve ROC One-vs-Rest
    # -----------------------------------------------------------------------

    def plot_roc_curves(self):
        """
        Disegna le curve ROC con approccio One-vs-Rest (OvR) per ogni classe.

        Richiede che `y_score` (probabilità di predict_proba) sia stato passato
        al costruttore. Per ogni classe i, la curva mostra il trade-off tra:
            - FPR (False Positive Rate = 1 - Specificità) sull'asse x
            - TPR (Recall) sull'asse y

        L'AUC (Area Under Curve) misura la capacità discriminativa:
            AUC = 1.0 → separazione perfetta
            AUC = 0.5 → equivale a un classificatore casuale

        NOTA: label_binarize e y_score devono essere allineati sullo stesso
        ordinamento di classi. Lo verifichiamo esplicitamente.
        """
        if self.y_score is None:
            print("  [Skip] y_score non fornito: le curve ROC non possono essere calcolate.")
            return

        # Verifica che y_score abbia il numero corretto di colonne
        if self.y_score.shape[1] != self.n_classes:
            print(
                f"  [Skip] y_score ha {self.y_score.shape[1]} colonne "
                f"ma ci sono {self.n_classes} classi: le curve ROC vengono saltate."
            )
            return

        # Binarizzazione OvR: colonna i = 1 se il campione appartiene alla classe i
        y_true_bin = label_binarize(self.y_true, classes=self.classes)

        fig, ax = plt.subplots(figsize=_FIGSIZE_MED)

        for i, (cls, nome, colore) in enumerate(
            zip(self.classes, self.target_names, _PALETTE)
        ):
            # Controlliamo che la classe sia presente in y_true (evita errori su subset piccoli)
            if y_true_bin[:, i].sum() == 0:
                print(f"  [Skip ROC] Classe '{nome}' assente nel test set.")
                continue

            fpr, tpr, _ = roc_curve(y_true_bin[:, i], self.y_score[:, i])
            roc_auc     = auc(fpr, tpr)

            ax.plot(fpr, tpr, color=colore, lw=2,
                    label=f"{nome}  (AUC = {roc_auc:.3f})")

        # Linea di riferimento: classificatore completamente casuale
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6,
                label="Classificatore casuale (AUC = 0.500)")

        ax.set_title("Curve ROC – One vs Rest", fontsize=14)
        ax.set_xlabel("False Positive Rate  (1 – Specificità)", fontsize=12)
        ax.set_ylabel("True Positive Rate  (Recall)",           fontsize=12)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.02])

        plt.tight_layout()
        self._salva_o_mostra(fig, "roc_curves.png")

    # -----------------------------------------------------------------------
    # GRAFICO 3: Bar chart Precision / Recall / F1 per classe
    # -----------------------------------------------------------------------

    def plot_class_report(self):
        """
        Bar chart che confronta Precision, Recall e F1 per ogni classe.

        Permette di individuare visivamente le classi in cui il modello è debole.
        La classe minoritaria (danno lieve, grado 1 — ~10% dei campioni) è
        tipicamente quella con Recall più basso a causa dello sbilanciamento.

        Una linea tratteggiata mostra la Micro-F1 globale come riferimento.
        """
        precision_vals = precision_score(
            self.y_true, self.y_pred, labels=self.classes, average=None, zero_division=0
        )
        recall_vals = recall_score(
            self.y_true, self.y_pred, labels=self.classes, average=None, zero_division=0
        )
        f1_vals = f1_score(
            self.y_true, self.y_pred, labels=self.classes, average=None, zero_division=0
        )

        # Costruiamo un DataFrame long-format per seaborn
        df_plot = pd.DataFrame({
            "Classe":  self.target_names * 3,
            "Valore":  list(precision_vals) + list(recall_vals) + list(f1_vals),
            "Metrica": (["Precision"] * self.n_classes +
                        ["Recall"]    * self.n_classes +
                        ["F1-score"]  * self.n_classes),
        })

        fig, ax = plt.subplots(figsize=(10, 6))
        barplot = sns.barplot(
            data=df_plot,
            x="Classe", y="Valore", hue="Metrica",
            palette=["#5B9BD5", "#ED7D31", "#A9D18E"],
            ax=ax,
        )

        # Aggiunge il valore numerico sopra ogni barra (approccio compatibile con seaborn ≥ 0.12)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)

        # Linea di riferimento: Micro-F1 globale
        micro = self.micro_f1()
        ax.axhline(micro, color="red", linestyle="--", lw=1.5,
                   label=f"Micro-F1 = {micro:.4f}")

        ax.set_title("Precision, Recall e F1-score per Classe", fontsize=14)
        ax.set_ylabel("Valore",           fontsize=12)
        ax.set_xlabel("Classe di Danno",  fontsize=12)
        ax.set_ylim(0, 1.15)   # spazio per le etichette numeriche
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        self._salva_o_mostra(fig, "class_report.png")

    # -----------------------------------------------------------------------
    # GRAFICO 4: Distribuzione predizioni vs valori reali
    # -----------------------------------------------------------------------

    def plot_distribuzione_classi(self):
        """
        Confronto delle distribuzioni: classi reali vs classi predette.

        Utile per rilevare bias sistematici del modello:
            - Se predetti >> reali per una classe → il modello sovra-predice quella classe.
            - Se predetti << reali → il modello sotto-predice (tipico per la classe 1).

        Su dataset sbilanciati questo grafico è più informativo dell'accuracy,
        perché mostra SE il modello ha imparato la struttura delle classi.
        """
        conteggi_reali    = [np.sum(self.y_true == c) for c in self.classes]
        conteggi_predetti = [np.sum(self.y_pred == c) for c in self.classes]

        x         = np.arange(self.n_classes)
        larghezza = 0.35

        fig, ax = plt.subplots(figsize=(9, 6))
        bars1 = ax.bar(x - larghezza / 2, conteggi_reali,    larghezza,
                       label="Reali",    color="#4878CF", alpha=0.85)
        bars2 = ax.bar(x + larghezza / 2, conteggi_predetti, larghezza,
                       label="Predetti", color="#D65F5F", alpha=0.85)

        ax.bar_label(bars1, fmt="%d", fontsize=9, padding=3)
        ax.bar_label(bars2, fmt="%d", fontsize=9, padding=3)

        ax.set_title("Distribuzione Classi: Reali vs Predetti", fontsize=14)
        ax.set_xlabel("Classe di Danno",        fontsize=12)
        ax.set_ylabel("Numero di Record",        fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(self.target_names)
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        self._salva_o_mostra(fig, "distribuzione_classi.png")

    # -----------------------------------------------------------------------
    # METODO AGGREGATORE
    # -----------------------------------------------------------------------

    def valuta_tutto(self, normalizza_cm: bool = True) -> dict:
        """
        Esegue l'intera valutazione in sequenza: metriche testuali + tutti i grafici.

        Ordine di esecuzione:
            1. Resoconto testuale (calcola_metriche)
            2. Salvataggio CSV dei risultati (se output_dir è impostato)
            3. Matrice di confusione — conteggi assoluti
            4. Matrice di confusione — normalizzata (se normalizza_cm=True)
            5. Curve ROC one-vs-rest (solo se y_score è disponibile)
            6. Bar chart Precision / Recall / F1 per classe
            7. Distribuzione classi reali vs predette

        Parameters
        ----------
        normalizza_cm : bool
            Se True, produce anche la versione normalizzata della CM.

        Returns
        -------
        dict : stesso output di calcola_metriche()
        """
        print("\n[Evaluation] Avvio valutazione completa del modello...")

        # 1. Resoconto testuale
        metriche = self.calcola_metriche()

        # 2. Salvataggio CSV (solo se output_dir è configurato)
        self.salva_metriche_csv(metriche)

        # 3–4. Matrice di confusione
        self.plot_confusion_matrix(normalizza=False)
        if normalizza_cm:
            self.plot_confusion_matrix(normalizza=True)

        # 5. Curve ROC
        self.plot_roc_curves()

        # 6. Bar chart metriche per classe
        self.plot_class_report()

        # 7. Distribuzione classi
        self.plot_distribuzione_classi()

        print("\n[Evaluation] Valutazione completata.")
        if self.output_dir:
            print(f"[Evaluation] Grafici salvati in: {self.output_dir}")

        return metriche

    # -----------------------------------------------------------------------
    # Metodo privato: salva o mostra il grafico
    # -----------------------------------------------------------------------

    def _salva_o_mostra(self, fig: plt.Figure, nome_file: str):
        """
        Salva il grafico su disco oppure lo mostra a schermo.

        Se output_dir è impostato, salva come PNG (dpi=150) e chiude la figura
        per liberare memoria. Altrimenti chiama plt.show() per la visualizzazione
        interattiva (utile in notebook o durante lo sviluppo).

        Parameters
        ----------
        fig       : figura matplotlib da salvare/mostrare
        nome_file : nome del file PNG (ignorato se output_dir è None)
        """
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            percorso = os.path.join(self.output_dir, nome_file)
            fig.savefig(percorso, dpi=150, bbox_inches="tight")
            print(f"  [Grafico salvato] -> {percorso}")
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)
