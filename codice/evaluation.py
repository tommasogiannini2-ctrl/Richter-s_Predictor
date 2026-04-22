"""
evaluation.py
=============
Modulo di valutazione delle prestazioni del modello per il progetto Richter's Predictor.

La metrica ufficiale della competizione DrivenData è la **micro-averaged F1 score**,
calcolata su tre classi ordinate (1 = danno lieve, 2 = danno medio, 3 = distruzione quasi totale).

Struttura del modulo:
    - ModelEvaluator : classe principale di valutazione
        ├── calcola_metriche()     → stampa un resoconto testuale completo
        ├── plot_confusion_matrix()→ heatmap della matrice di confusione
        ├── plot_roc_curves()      → curve ROC one-vs-rest per ogni classe
        ├── plot_class_report()    → bar chart con precision/recall/F1 per classe
        └── valuta_tutto()         → esegue tutte le valutazioni in un'unica chiamata
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

_PALETTE = ["#4878CF", "#6ACC65", "#D65F5F"]   # blu, verde, rosso – una per classe
_FIGSIZE_LARGE = (18, 6)
_FIGSIZE_MEDIUM = (10, 7)

class ModelEvaluator:
    """
    Valutatore delle prestazioni di un classificatore multi-classe (3 classi).
    """

    def __init__(
        self,
        y_true,
        y_pred,
        y_score=None,
        target_names=None,
        output_dir=None,
    ):
        # Converti in array numpy per uniformità
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.y_score = np.asarray(y_score) if y_score is not None else None

        # Ricava le classi uniche in ordine crescente
        self.classes = np.unique(self.y_true)
        self.n_classes = len(self.classes)

        # Nomi delle classi (usati nelle etichette dei grafici)
        if target_names is not None:
            self.target_names = target_names
        else:
            self.target_names = [
                "Danno Lieve (1)",
                "Danno Medio (2)",
                "Distruzione (3)",
            ]

        # Cartella di output per i grafici (opzionale)
        self.output_dir = output_dir

    # -----------------------------------------------------------------------
    # METRICA PRINCIPALE: micro-F1 (metrica ufficiale della competizione)
    # -----------------------------------------------------------------------

    def micro_f1(self) -> float:
        """
        Calcola la micro-averaged F1 score.

        La micro-F1 aggrega i conteggi di TP, FP e FN su tutte le classi
        prima di calcolare F1, dando uguale peso a ogni singola predizione
        indipendentemente dalla classe. È la metrica ufficiale di DrivenData.

        Returns
        -------
        float
            Valore della micro-F1 (tra 0 e 1).
        """
        return f1_score(self.y_true, self.y_pred, average="micro")

    # -----------------------------------------------------------------------
    # METODO PRINCIPALE: resoconto testuale completo
    # -----------------------------------------------------------------------

    def calcola_metriche(self) -> dict:
        """
        Stampa un resoconto testuale completo delle metriche e lo restituisce
        come dizionario per un eventuale uso programmatico.

        Metriche calcolate
        ------------------
        - Micro-F1          : metrica ufficiale della competizione
        - Accuracy          : percentuale di predizioni corrette
        - Per ogni classe   : TPR (Recall / Sensibilità), TNR (Specificità),
                              Precision, F1-score, supporto
        - Media macro       : media non pesata tra le classi
        - Geometric Mean    : sqrt(avg_TPR * avg_TNR) – penalizza squilibri
                              tra classi

        Returns
        -------
        dict
            Dizionario con tutte le metriche calcolate.
        """
        # --- Matrice di confusione ---
        cm = confusion_matrix(self.y_true, self.y_pred, labels=self.classes)

        # --- Metriche aggregate ---
        micro_f1   = f1_score(self.y_true, self.y_pred, average="micro")
        macro_f1   = f1_score(self.y_true, self.y_pred, average="macro")
        accuracy   = np.sum(np.diag(cm)) / np.sum(cm)

        # --- Metriche per classe (TPR e TNR) ---
        tpr_list = []   # True Positive Rate  (Recall / Sensibilità)
        tnr_list = []   # True Negative Rate  (Specificità)
        prec_list = []  # Precision per classe
        f1_list   = []  # F1 per classe

        risultati_per_classe = {}

        for i, cls in enumerate(self.classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp        # falsi negativi: stessa riga, colonne diverse
            fp = cm[:, i].sum() - tp        # falsi positivi: stessa colonna, righe diverse
            tn = cm.sum() - (tp + fp + fn)  # tutto il resto è TN

            # Calcolo sicuro (evita divisione per zero)
            tpr  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            tnr  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1   = (2 * prec * tpr) / (prec + tpr) if (prec + tpr) > 0 else 0.0

            tpr_list.append(tpr)
            tnr_list.append(tnr)
            prec_list.append(prec)
            f1_list.append(f1)

            risultati_per_classe[cls] = {
                "TPR (Recall)":  tpr,
                "TNR (Spec.)":   tnr,
                "Precision":     prec,
                "F1-score":      f1,
                "Supporto":      int(cm[i, :].sum()),
            }

        # Geometric Mean: radice del prodotto di TPR medio e TNR medio.
        # Un valore elevato richiede che il modello sia buono su TUTTE le classi.
        avg_tpr   = np.mean(tpr_list)
        avg_tnr   = np.mean(tnr_list)
        g_mean    = np.sqrt(avg_tpr * avg_tnr)

        # ---- STAMPA RISULTATI ----
        sep = "=" * 62
        print(f"\n{sep}")
        print("  RESOCONTO VALUTAZIONE MODELLO")
        print(sep)
        print(f"  {'Micro-F1 (metrica ufficiale):':<35} {micro_f1:.4f}")
        print(f"  {'Macro-F1:':<35} {macro_f1:.4f}")
        print(f"  {'Accuracy:':<35} {accuracy:.4f}")
        print(f"  {'Geometric Mean (TPR * TNR):':<35} {g_mean:.4f}")
        print(sep)

        # Tabella per classe
        header = f"  {'CLASSE':<22} {'TPR':>8} {'TNR':>8} {'Prec':>8} {'F1':>8} {'N':>8}"
        print(header)
        print(f"  {'-' * 60}")
        for i, cls in enumerate(self.classes):
            nome = self.target_names[i]
            r = risultati_per_classe[cls]
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
            f"{avg_tpr:>8.4f} "
            f"{avg_tnr:>8.4f} "
            f"{np.mean(prec_list):>8.4f} "
            f"{macro_f1:>8.4f}"
        )
        print(sep)

        # Anche il classification_report dettagliato di sklearn
        print("\n  CLASSIFICATION REPORT (sklearn):")
        print(
            classification_report(
                self.y_true,
                self.y_pred,
                labels=self.classes,
                target_names=self.target_names,
            )
        )

        # Restituisce un dizionario per uso programmatico
        return {
            "micro_f1":          micro_f1,
            "macro_f1":          macro_f1,
            "accuracy":          accuracy,
            "g_mean":            g_mean,
            "per_classe":        risultati_per_classe,
        }

    # -----------------------------------------------------------------------
    # GRAFICO 1: Matrice di Confusione
    # -----------------------------------------------------------------------

    def plot_confusion_matrix(self, normalizza: bool = False):
        """
        Disegna la matrice di confusione come heatmap.

        Parameters
        ----------
        normalizza : bool
            Se True, mostra le frequenze relative (%) invece dei conteggi assoluti.
            Utile per confrontare dataset di dimensioni diverse.
        """
        cm = confusion_matrix(self.y_true, self.y_pred, labels=self.classes)

        if normalizza:
            # Normalizza per riga: ogni cella mostra la % rispetto al totale reale
            cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fmt     = ".2%"
            titolo  = "Matrice di Confusione (normalizzata per riga)"
        else:
            cm_plot = cm
            fmt     = "d"
            titolo  = "Matrice di Confusione (conteggi assoluti)"

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
        ax.set_ylabel("Classe Reale", fontsize=12)
        ax.set_xlabel("Classe Predetta", fontsize=12)
        ax.tick_params(axis="x", rotation=15)
        ax.tick_params(axis="y", rotation=0)

        plt.tight_layout()
        self._salva_o_mostra(fig, "confusion_matrix.png")

    # -----------------------------------------------------------------------
    # GRAFICO 2: Curve ROC one-vs-rest
    # -----------------------------------------------------------------------

    def plot_roc_curves(self):
        """
        Disegna le curve ROC con approccio One-vs-Rest (OvR) per ogni classe.

        Richiede che `y_score` sia stato passato al costruttore.
        L'area sotto la curva (AUC) è indicata nella legenda per ogni classe.
        Una linea tratteggiata rappresenta il classificatore casuale (AUC = 0.5).
        """
        if self.y_score is None:
            print("[AVVISO] y_score non fornito: le curve ROC non possono essere calcolate.")
            return

        # Binarizzazione delle etichette (OvR: una colonna per classe)
        y_true_bin = label_binarize(self.y_true, classes=self.classes)

        fig, ax = plt.subplots(figsize=_FIGSIZE_MEDIUM)

        for i, (cls, nome, colore) in enumerate(
            zip(self.classes, self.target_names, _PALETTE)
        ):
            # Curva ROC per la classe i contro tutte le altre
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], self.y_score[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr, tpr,
                color=colore,
                lw=2,
                label=f"{nome}  (AUC = {roc_auc:.3f})",
            )

        # Linea del classificatore casuale
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Casuale (AUC = 0.500)")

        ax.set_title("Curve ROC – One vs Rest", fontsize=14)
        ax.set_xlabel("False Positive Rate  (1 – Specificità)", fontsize=12)
        ax.set_ylabel("True Positive Rate  (Recall)", fontsize=12)
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
        Disegna un bar chart che confronta Precision, Recall e F1 per ogni classe.

        Permette di individuare visivamente le classi in cui il modello è debole,
        ad esempio la classe minoritaria (danno lieve, grado 1) spesso risulta
        penalizzata per lo squilibrio del dataset.
        """
        # Calcola metriche per classe usando sklearn
        precision_vals = precision_score(
            self.y_true, self.y_pred, labels=self.classes, average=None
        )
        recall_vals = recall_score(
            self.y_true, self.y_pred, labels=self.classes, average=None
        )
        f1_vals = f1_score(
            self.y_true, self.y_pred, labels=self.classes, average=None
        )

        # Costruisce un DataFrame per facilitare il plotting con seaborn
        df_plot = pd.DataFrame(
            {
                "Classe":    self.target_names * 3,
                "Valore":    list(precision_vals) + list(recall_vals) + list(f1_vals),
                "Metrica":   (["Precision"] * self.n_classes +
                               ["Recall"]    * self.n_classes +
                               ["F1-score"]  * self.n_classes),
            }
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=df_plot,
            x="Classe",
            y="Valore",
            hue="Metrica",
            palette=["#5B9BD5", "#ED7D31", "#A9D18E"],
            ax=ax,
        )

        # Aggiunge il valore numerico sopra ogni barra
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)

        # Linea orizzontale alla micro-F1 come riferimento
        micro = self.micro_f1()
        ax.axhline(micro, color="red", linestyle="--", lw=1.5,
                   label=f"Micro-F1 = {micro:.4f}")

        ax.set_title("Precision, Recall e F1-score per Classe", fontsize=14)
        ax.set_ylabel("Valore", fontsize=12)
        ax.set_xlabel("Classe di Danno", fontsize=12)
        ax.set_ylim(0, 1.12)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        self._salva_o_mostra(fig, "class_report.png")

    # -----------------------------------------------------------------------
    # GRAFICO 4: Distribuzione predizioni vs valori reali
    # -----------------------------------------------------------------------

    def plot_distribuzione_classi(self):
        """
        Confronta la distribuzione delle classi reali con quelle predette.

        Utile per rilevare bias sistematici: se il modello tende a sovra- o
        sotto-predire una classe, le barre differiscono significativamente.
        """
        # Conta le occorrenze di ogni classe nel vettore reale e in quello predetto
        conteggi_reali  = [np.sum(self.y_true == c) for c in self.classes]
        conteggi_predetti = [np.sum(self.y_pred == c) for c in self.classes]

        x = np.arange(self.n_classes)
        larghezza = 0.35

        fig, ax = plt.subplots(figsize=(9, 6))
        bars1 = ax.bar(x - larghezza / 2, conteggi_reali,   larghezza,
                       label="Reali",    color="#4878CF", alpha=0.85)
        bars2 = ax.bar(x + larghezza / 2, conteggi_predetti, larghezza,
                       label="Predetti", color="#D65F5F", alpha=0.85)

        ax.bar_label(bars1, fmt="%d", fontsize=9, padding=3)
        ax.bar_label(bars2, fmt="%d", fontsize=9, padding=3)

        ax.set_title("Distribuzione Classi: Reali vs Predetti", fontsize=14)
        ax.set_xlabel("Classe di Danno", fontsize=12)
        ax.set_ylabel("Numero di Record", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(self.target_names)
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        self._salva_o_mostra(fig, "distribuzione_classi.png")

    # -----------------------------------------------------------------------
    # METODO AGGREGATORE: esegue tutto in un'unica chiamata
    # -----------------------------------------------------------------------

    def valuta_tutto(self, normalizza_cm: bool = True) -> dict:
        """
        Esegue l'intera valutazione: metriche testuali + tutti i grafici.

        Parameters
        ----------
        normalizza_cm : bool
            Se True, la matrice di confusione viene mostrata normalizzata
            (frequenze relative) oltre che con i valori assoluti.

        Returns
        -------
        dict
            Dizionario delle metriche (stesso output di calcola_metriche).
        """
        print("\n[Evaluation] Avvio valutazione completa del modello...")

        # 1. Resoconto testuale
        metriche = self.calcola_metriche()

        # 2. Matrice di confusione (valori assoluti)
        self.plot_confusion_matrix(normalizza=False)

        # 3. Matrice di confusione normalizzata (opzionale)
        if normalizza_cm:
            self.plot_confusion_matrix(normalizza=True)

        # 4. Curve ROC (solo se y_score è disponibile)
        self.plot_roc_curves()

        # 5. Bar chart metriche per classe
        self.plot_class_report()

        # 6. Distribuzione classi reali vs predette
        self.plot_distribuzione_classi()

        print("\n[Evaluation] Valutazione completata.")
        return metriche

    # -----------------------------------------------------------------------
    # Metodo privato: salva il grafico su disco oppure lo mostra a schermo
    # -----------------------------------------------------------------------

    def _salva_o_mostra(self, fig: plt.Figure, nome_file: str):
        """
        Salva il grafico nella cartella `output_dir` (se specificata),
        altrimenti lo mostra direttamente a schermo.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            La figura da salvare o mostrare.
        nome_file : str
            Nome del file PNG (usato solo se output_dir è impostato).
        """
        import os
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            percorso = os.path.join(self.output_dir, nome_file)
            fig.savefig(percorso, dpi=150, bbox_inches="tight")
            print(f"  [Grafico salvato] → {percorso}")
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)


# ---------------------------------------------------------------------------
# Blocco di test rapido (eseguibile direttamente con: python evaluation.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Esempio di utilizzo con dati sintetici.

    Simula un scenario realistico con:
      - 10.000 campioni
      - Squilibrio di classe tipico del dataset Nepal Earthquake
        (classe 1: ~10%, classe 2: ~57%, classe 3: ~33%)
    """
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    print("=" * 60)
    print("  TEST RAPIDO – ModelEvaluator con dati sintetici")
    print("=" * 60)

    # Riproduzione dello squilibrio di classe del dataset reale
    rng = np.random.default_rng(42)
    n_totale = 10_000
    y_demo = rng.choice([1, 2, 3], size=n_totale, p=[0.10, 0.57, 0.33])

    # Feature casuali (solo per dimostrare il funzionamento)
    X_demo = rng.standard_normal((n_totale, 10))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_demo, y_demo, test_size=0.3, random_state=42, stratify=y_demo
    )

    # Addestramento di un classificatore di esempio
    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)

    y_pred_demo  = clf.predict(X_te)
    y_score_demo = clf.predict_proba(X_te)   # probabilità per le curve ROC

    # Istanziazione del valutatore
    evaluator = ModelEvaluator(
        y_true=y_te,
        y_pred=y_pred_demo,
        y_score=y_score_demo,
        output_dir=None,   # None → mostra i grafici a schermo
    )

    # Valutazione completa
    risultati = evaluator.valuta_tutto()

    print(f"\nMicro-F1 finale: {risultati['micro_f1']:.4f}")