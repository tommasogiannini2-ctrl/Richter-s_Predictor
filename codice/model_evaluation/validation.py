"""
feature_selection_search.py
============================
Random search condizionale su selettore + modello per il progetto Richter's Predictor.

STRUTTURA GENERALE
------------------
Il problema da risolvere è trovare la combinazione ottimale di:
    (selettore, iperparametri selettore, modello, iperparametri modello)

Approccio scelto: Opzione B — lista di dizionari per RandomizedSearchCV.
    - Ogni dizionario rappresenta una coppia (selettore, modello) specifica.
    - All'interno del dizionario, solo i parametri sensati per quella coppia.
    - RandomizedSearchCV campiona uniformemente tra i dizionari e, per ogni
      dizionario estratto, campiona i valori degli iperparametri.

Questo produce 5 selettori × 2 modelli = 10 dizionari nello spazio di ricerca.

PIPELINE
--------
La ricerca usa una sklearn Pipeline con due step nominati:
    - "selector" : uno dei 5 selettori di feature_select_extract.py
    - "model"    : RandomForestClassifier oppure KNeighborsClassifier

I parametri nella lista di dizionari usano la convenzione sklearn:
    "selector__k"           → parametro k dello step "selector"
    "model__n_estimators"   → parametro n_estimators dello step "model"
    "selector"              → l'oggetto selettore stesso (lista con 1 elemento)
    "model"                 → l'oggetto modello stesso   (lista con 1 elemento)

CAMPIONAMENTO
-------------
Con n_iter=N e M dizionari, ogni dizionario riceve circa N/M campionamenti.
I valori continui usano scipy.stats (randint, uniform) per il campionamento;
i valori discreti usano liste Python semplici.

NOTA SU SFS
-----------
SFSSelector è molto lento (k × cv training per ogni chiamata a fit).
Viene incluso nella ricerca ma con k basso e cv=2 per limitare il costo.
In produzione si consiglia di escluderlo o ridurre ulteriormente n_iter.
"""

import os
import warnings

import numpy as np
import pandas as pd

from scipy.stats import randint

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

from model_evaluation.feature_select_extract import (
    AllFeaturesSelector,
    MutualInfoSelector,
    ReliefFSelector,
    SFSSelector,
    EmbeddedDTSelector,
)


# ===========================================================================
# COSTRUZIONE DELLO SPAZIO DI RICERCA CONDIZIONALE
# ===========================================================================

def _build_search_space(include_sfs: bool = True) -> list:
    """
    Costruisce la lista di dizionari che definisce lo spazio di ricerca.

    Ogni dizionario rappresenta una coppia (selettore, modello) e contiene
    solo i parametri sensati per quella coppia specifica.

    Parameters
    ----------
    include_sfs : bool
        Se False, esclude i dizionari con SFSSelector per velocizzare
        la ricerca (utile in fase di sviluppo o con dataset grandi).

    Returns
    -------
    list of dict
        Lista di dizionari da passare a RandomizedSearchCV come
        param_distributions.
    """

    spazio = []

    # -----------------------------------------------------------------------
    # ① AllFeaturesSelector + RandomForest
    # -----------------------------------------------------------------------
    spazio.append({
        "selector": [AllFeaturesSelector()],
        "model": [RandomForestClassifier(random_state=42)],
        "model__n_estimators":  randint(100, 400),
        "model__max_depth":     [None, 10, 20, 30],
        "model__min_samples_leaf": randint(1, 10),
        "model__class_weight":  [None, "balanced"],
    })

    # -----------------------------------------------------------------------
    # ② AllFeaturesSelector + KNN
    # -----------------------------------------------------------------------
    spazio.append({
        "selector": [AllFeaturesSelector()],
        "model": [KNeighborsClassifier()],
        "model__n_neighbors": randint(3, 25),
        "model__weights":     ["uniform", "distance"],
        "model__metric":      ["euclidean", "manhattan"],
        "model__leaf_size":   randint(20, 50),
    })

    # -----------------------------------------------------------------------
    # ③ AllFeaturesSelector + AdaBoost
    # -----------------------------------------------------------------------
    spazio.append({
        "selector": [AllFeaturesSelector()],
        "model": [AdaBoostClassifier(random_state=42)],
        "model__n_estimators": randint(50, 300),
        "model__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        "model__estimator": [DecisionTreeClassifier(max_depth=1), 
                             DecisionTreeClassifier(max_depth=2),
                             DecisionTreeClassifier(max_depth=3)],
    })

    # -----------------------------------------------------------------------
    # ④ MutualInfoSelector + RandomForest
    # -----------------------------------------------------------------------
    spazio.append({
        "selector": [MutualInfoSelector()],
        "selector__k":            randint(15, 40),
        "model": [RandomForestClassifier(random_state=42)],
        "model__n_estimators":    randint(100, 400),
        "model__max_depth":       [None, 10, 20, 30],
        "model__min_samples_leaf": randint(1, 10),
        "model__class_weight":    [None, "balanced"],
    })

    # -----------------------------------------------------------------------
    # ⑤ MutualInfoSelector + KNN
    # -----------------------------------------------------------------------
    spazio.append({
        "selector": [MutualInfoSelector()],
        "selector__k":        randint(15, 40),
        "model": [KNeighborsClassifier()],
        "model__n_neighbors": randint(3, 25),
        "model__weights":     ["uniform", "distance"],
        "model__metric":      ["euclidean", "manhattan"],
        "model__leaf_size":   randint(20, 50),
    })

    # -----------------------------------------------------------------------
    # ⑥ MutualInfoSelector + AdaBoost
    # -----------------------------------------------------------------------
    spazio.append({
        "selector": [MutualInfoSelector()],
        "selector__k":        randint(15, 40),
        "model": [AdaBoostClassifier(random_state=42)],
        "model__n_estimators": randint(50, 300),
        "model__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        "model__estimator": [DecisionTreeClassifier(max_depth=1), 
                             DecisionTreeClassifier(max_depth=2),
                             DecisionTreeClassifier(max_depth=3)],
    })

    # -----------------------------------------------------------------------
    # ⑦ ReliefFSelector + RandomForest
    # -----------------------------------------------------------------------
    spazio.append({
        "selector": [ReliefFSelector()],
        "selector__k":            randint(15, 40),
        "selector__n_neighbors":  randint(5, 20),
        "selector__n_samples":    [500, 1000, 2000],
        "model": [RandomForestClassifier(random_state=42)],
        "model__n_estimators":    randint(100, 400),
        "model__max_depth":       [None, 10, 20, 30],
        "model__min_samples_leaf": randint(1, 10),
        "model__class_weight":    [None, "balanced"],
    })

    # -----------------------------------------------------------------------
    # ⑧ ReliefFSelector + KNN
    # -----------------------------------------------------------------------
    spazio.append({
        "selector": [ReliefFSelector()],
        "selector__k":           randint(15, 40),
        "selector__n_neighbors": randint(5, 20),
        "selector__n_samples":   [500, 1000, 2000],
        "model": [KNeighborsClassifier()],
        "model__n_neighbors": randint(3, 25),
        "model__weights":     ["uniform", "distance"],
        "model__metric":      ["euclidean", "manhattan"],
        "model__leaf_size":   randint(20, 50),
    })

    # -----------------------------------------------------------------------
    # ⑨ ReliefFSelector + AdaBoost
    # -----------------------------------------------------------------------
    spazio.append({
        "selector": [ReliefFSelector()],
        "selector__k":           randint(15, 40),
        "selector__n_neighbors": randint(5, 20),
        "selector__n_samples":   [500, 1000, 2000],
        "model": [AdaBoostClassifier(random_state=42)],
        "model__n_estimators": randint(50, 300),
        "model__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        "model__estimator": [DecisionTreeClassifier(max_depth=1), 
                             DecisionTreeClassifier(max_depth=2),
                             DecisionTreeClassifier(max_depth=3)],
    })

    # -----------------------------------------------------------------------
    # ⑩ EmbeddedDTSelector + RandomForest
    # -----------------------------------------------------------------------
    spazio.append({
        "selector": [EmbeddedDTSelector()],
        "selector__soglia":    ["mean", "median", 0.005, 0.01, 0.02],
        "selector__max_depth": randint(8, 20),
        "model": [RandomForestClassifier(random_state=42)],
        "model__n_estimators":    randint(100, 400),
        "model__max_depth":       [None, 10, 20, 30],
        "model__min_samples_leaf": randint(1, 10),
        "model__class_weight":    [None, "balanced"],
    })

    # -----------------------------------------------------------------------
    # ⑪ EmbeddedDTSelector + KNN
    # -----------------------------------------------------------------------
    spazio.append({
        "selector": [EmbeddedDTSelector()],
        "selector__soglia":    ["mean", "median", 0.005, 0.01, 0.02],
        "selector__max_depth": randint(8, 20),
        "model": [KNeighborsClassifier()],
        "model__n_neighbors": randint(3, 25),
        "model__weights":     ["uniform", "distance"],
        "model__metric":      ["euclidean", "manhattan"],
        "model__leaf_size":   randint(20, 50),
    })

    # -----------------------------------------------------------------------
    # ⑫ EmbeddedDTSelector + AdaBoost
    # -----------------------------------------------------------------------
    spazio.append({
        "selector": [EmbeddedDTSelector()],
        "selector__soglia":    ["mean", "median", 0.005, 0.01, 0.02],
        "selector__max_depth": randint(8, 20),
        "model": [AdaBoostClassifier(random_state=42)],
        "model__n_estimators": randint(50, 300),
        "model__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        "model__estimator": [DecisionTreeClassifier(max_depth=1), 
                             DecisionTreeClassifier(max_depth=2),
                             DecisionTreeClassifier(max_depth=3)],
    })

    # -----------------------------------------------------------------------
    # ⑬ SFSSelector + RandomForest
    # -----------------------------------------------------------------------
    if include_sfs:
        spazio.append({
            "selector": [SFSSelector()],
            "selector__k":           randint(10, 25),
            "selector__cv":          [2, 3],
            "selector__max_depth_dt": randint(5, 12),
            "model": [RandomForestClassifier(random_state=42)],
            "model__n_estimators":    randint(100, 300),
            "model__max_depth":       [None, 10, 20],
            "model__class_weight":    [None, "balanced"],
        })

    # -----------------------------------------------------------------------
    # ⑭ SFSSelector + KNN
    # -----------------------------------------------------------------------
    if include_sfs:
        spazio.append({
            "selector": [SFSSelector()],
            "selector__k":           randint(10, 25),
            "selector__cv":          [2, 3],
            "selector__max_depth_dt": randint(5, 12),
            "model": [KNeighborsClassifier()],
            "model__n_neighbors": randint(3, 25),
            "model__weights":     ["uniform", "distance"],
            "model__metric":      ["euclidean", "manhattan"],
        })

    # -----------------------------------------------------------------------
    # ⑮ SFSSelector + AdaBoost
    # -----------------------------------------------------------------------
    if include_sfs:
        spazio.append({
            "selector": [SFSSelector()],
            "selector__k":           randint(10, 25),
            "selector__cv":          [2, 3],
            "selector__max_depth_dt": randint(5, 12),
            "model": [AdaBoostClassifier(random_state=42)],
            "model__n_estimators": randint(50, 300),
            "model__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
            "model__estimator": [DecisionTreeClassifier(max_depth=1), 
                                 DecisionTreeClassifier(max_depth=2),
                                 DecisionTreeClassifier(max_depth=3)],
        })

    return spazio


# ===========================================================================
# CLASSE PRINCIPALE
# ===========================================================================

class FeatureSelectionSearch:
    """
    Random search condizionale su selettore di feature + modello.

    Costruisce una sklearn Pipeline con due step (selector, model) e
    usa RandomizedSearchCV con uno spazio di ricerca a lista di dizionari
    per esplorare le combinazioni (selettore, iperparametri, modello,
    iperparametri) in modo condizionale ed efficiente.

    Parameters
    ----------
    n_iter : int
        Numero totale di configurazioni da campionare.
        Con 10 dizionari nello spazio, n_iter=50 produce ~5 campioni
        per dizionario in media.
    cv : int
        Fold di cross-validation per valutare ogni configurazione.
        Valori consigliati: 3 (veloce) o 5 (più robusto).
    scoring : str
        Metrica di valutazione. Default "f1_micro" (metrica DrivenData).
    n_jobs : int
        Parallelismo della ricerca (-1 = tutti i core).
    random_state : int
        Seed per riproducibilità del campionamento.
    include_sfs : bool
        Se False, esclude SFSSelector dalla ricerca (più veloce).
        Utile in fase di sviluppo o su macchine con poche risorse.
    verbose : int
        Livello di verbosità di RandomizedSearchCV (0=silenzioso, 2=dettagliato).
    output_dir : str, optional
        Cartella dove salvare i risultati in CSV. Se None, non salva.
    """

    def __init__(
        self,
        n_iter: int = 50,
        cv: int = 3,
        scoring: str = "f1_micro",
        n_jobs: int = -1,
        random_state: int = 42,
        include_sfs: bool = True,
        verbose: int = 1,
        output_dir: str = None,
    ):
        self.n_iter       = n_iter
        self.cv           = cv
        self.scoring      = scoring
        self.n_jobs       = n_jobs
        self.random_state = random_state
        self.include_sfs  = include_sfs
        self.verbose      = verbose
        self.output_dir   = output_dir

        # Attributi popolati dopo fit()
        self.search_        = None   # oggetto RandomizedSearchCV fittato
        self.best_pipeline_ = None   # Pipeline con la migliore configurazione
        self.results_df_    = None   # DataFrame con tutti i risultati

    # -----------------------------------------------------------------------
    # METODO PRINCIPALE: fit
    # -----------------------------------------------------------------------

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Esegue la random search sull'intero spazio condizionale.

        Costruisce la Pipeline base, lo spazio di ricerca (lista di
        dizionari) e lancia RandomizedSearchCV. Al termine popola
        best_pipeline_ e results_df_.

        Parameters
        ----------
        X_train : pd.DataFrame
            Feature di training (già preprocessate).
        y_train : pd.Series
            Target di training.

        Returns
        -------
        self
        """
        sep = "=" * 62

        print(f"\n{sep}")
        print(f"  FEATURE SELECTION SEARCH")
        print(sep)
        print(f"  {'n_iter:':<35} {self.n_iter}")
        print(f"  {'cv folds:':<35} {self.cv}")
        print(f"  {'scoring:':<35} {self.scoring}")
        print(f"  {'include_sfs:':<35} {self.include_sfs}")
        print(f"  {'n_jobs:':<35} {self.n_jobs}")
        print(sep)

        # --- Pipeline base (selector + model sono placeholder) ---
        # I valori iniziali non contano: RandomizedSearchCV li sovrascrive
        # ad ogni campionamento usando set_params().
        pipeline = Pipeline([
            ("selector", AllFeaturesSelector()),      # placeholder
            ("model",    RandomForestClassifier()),   # placeholder
        ])

        # --- Spazio di ricerca condizionale ---
        spazio = _build_search_space(include_sfs=self.include_sfs)
        n_dizionari = len(spazio)
        print(f"\n  Dizionari nello spazio di ricerca: {n_dizionari}")
        print(f"  Campionamenti medi per dizionario: ~{self.n_iter // n_dizionari}")

        # --- RandomizedSearchCV ---
        # refit=True (default): al termine riaddestra la pipeline migliore
        # sull'intero X_train. Questo è il modello che useremo per predict.
        self.search_ = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=spazio,
            n_iter=self.n_iter,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            refit=True,          # riaddestra il migliore su tutto X_train
            error_score=np.nan,  # non interrompere la ricerca su errori isolati
        )

        print(f"\n  Avvio RandomizedSearchCV...")
        with warnings.catch_warnings():
            # Sopprimiamo ConvergenceWarning e simili durante la ricerca
            warnings.simplefilter("ignore")
            self.search_.fit(X_train, y_train)

        # --- Estrazione risultati ---
        self.best_pipeline_ = self.search_.best_estimator_
        self.results_df_    = self._estrai_risultati()

        # --- Stampa riepilogo ---
        self._stampa_riepilogo()

        # --- Salvataggio su disco (opzionale) ---
        if self.output_dir is not None:
            self._salva_risultati()

        return self

    # -----------------------------------------------------------------------
    # PREDICT / TRANSFORM
    # -----------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predice le etichette usando la pipeline migliore trovata.

        Applica in sequenza il selettore migliore (già fittato) e il
        modello migliore (già fittato), entrambi su tutto X_train grazie
        a refit=True in RandomizedSearchCV.

        Parameters
        ----------
        X : pd.DataFrame
            Feature del set da predire (stesso schema di X_train).

        Returns
        -------
        np.ndarray
            Etichette predette.
        """
        self._verifica_fit()
        return self.best_pipeline_.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Restituisce le probabilità di classe usando la pipeline migliore.

        Utile per le curve ROC in ModelEvaluator.

        Parameters
        ----------
        X : pd.DataFrame
            Feature del set da predire.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Probabilità per ogni classe.
        """
        self._verifica_fit()
        if not hasattr(self.best_pipeline_["model"], "predict_proba"):
            raise AttributeError(
                f"Il modello '{type(self.best_pipeline_['model']).__name__}' "
                "non supporta predict_proba."
            )
        return self.best_pipeline_.predict_proba(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applica solo il selettore migliore a X (senza il modello).

        Utile per ottenere il dataset ridotto da passare ad altri modelli.

        Parameters
        ----------
        X : pd.DataFrame
            Feature originali.

        Returns
        -------
        pd.DataFrame
            Feature filtrate dal selettore migliore.
        """
        self._verifica_fit()
        return self.best_pipeline_["selector"].transform(X)

    # -----------------------------------------------------------------------
    # REPORT E UTILITÀ
    # -----------------------------------------------------------------------

    def get_best_params(self) -> dict:
        """Restituisce i parametri della configurazione migliore."""
        self._verifica_fit()
        return self.search_.best_params_

    def get_best_score(self) -> float:
        """Restituisce lo score CV della configurazione migliore."""
        self._verifica_fit()
        return self.search_.best_score_

    def get_results(self) -> pd.DataFrame:
        """
        Restituisce un DataFrame con tutti i risultati della ricerca,
        ordinato per score decrescente.
        """
        self._verifica_fit()
        return self.results_df_

    def _estrai_risultati(self) -> pd.DataFrame:
        """
        Converte cv_results_ di sklearn in un DataFrame leggibile.

        Estrae per ogni configurazione testata:
            - nome del selettore e del modello
            - score medio e deviazione standard in CV
            - numero di feature selezionate (dal selettore fittato)
            - rank di sklearn
        """
        cv_res = self.search_.cv_results_

        righe = []
        for i in range(len(cv_res["params"])):
            params = cv_res["params"][i]

            # Estrai il nome del selettore e del modello dalla configurazione
            nome_selector = type(params.get("selector", "")).__name__
            nome_model    = type(params.get("model", "")).__name__

            righe.append({
                "rank":          cv_res["rank_test_score"][i],
                "selector":      nome_selector,
                "model":         nome_model,
                "mean_cv_score": round(cv_res["mean_test_score"][i], 5),
                "std_cv_score":  round(cv_res["std_test_score"][i],  5),
                "params":        params,
            })

        df = pd.DataFrame(righe).sort_values("rank").reset_index(drop=True)
        return df

    def _stampa_riepilogo(self):
        """Stampa un riepilogo della ricerca al termine del fit."""
        sep = "=" * 62
        best = self.search_.best_params_

        print(f"\n{sep}")
        print(f"  RISULTATI FEATURE SELECTION SEARCH")
        print(sep)
        print(f"  {'Miglior score CV (' + self.scoring + '):':<35} "
              f"{self.search_.best_score_:.4f}")
        print(f"  {'Miglior selettore:':<35} "
              f"{type(best['selector']).__name__}")
        print(f"  {'Miglior modello:':<35} "
              f"{type(best['model']).__name__}")
        print(sep)

        # Top-5 configurazioni
        print(f"\n  Top-5 configurazioni per {self.scoring}:")
        print(f"  {'Rank':<6} {'Selector':<25} {'Model':<30} {'Score CV':<12} {'Std'}")
        print(f"  {'-' * 80}")
        for _, row in self.results_df_.head(5).iterrows():
            print(
                f"  {row['rank']:<6} "
                f"{row['selector']:<25} "
                f"{row['model']:<30} "
                f"{row['mean_cv_score']:<12.4f} "
                f"± {row['std_cv_score']:.4f}"
            )
        print(sep)

    def _salva_risultati(self):
        """Salva il DataFrame dei risultati su CSV nella output_dir."""
        os.makedirs(self.output_dir, exist_ok=True)
        percorso = os.path.join(self.output_dir, "feature_selection_results.csv")

        # Escludiamo la colonna "params" (non serializzabile in CSV in modo pulito)
        df_save = self.results_df_.drop(columns=["params"], errors="ignore")
        df_save.to_csv(percorso, index=False)
        print(f"  [CSV salvato] → {percorso}")

    def _verifica_fit(self):
        """Lancia RuntimeError se fit() non è stato ancora chiamato."""
        if self.search_ is None:
            raise RuntimeError(
                "FeatureSelectionSearch non è stato ancora addestrato. "
                "Chiamare fit(X_train, y_train) prima."
            )


# ===========================================================================
# BLOCCO DI TEST RAPIDO
# ===========================================================================

if __name__ == "__main__":
    """
    Test rapido con dati sintetici per verificare che la pipeline funzioni
    end-to-end prima di lanciare la ricerca sul dataset reale.

    Nota: escludiamo SFS dal test rapido (troppo lento su dati sintetici).
    """
    from sklearn.datasets import make_classification

    print("=" * 62)
    print("  TEST RAPIDO — FeatureSelectionSearch su dati sintetici")
    print("=" * 62)

    # Dataset sintetico con squilibrio di classe simile al dataset reale
    X_demo, y_demo = make_classification(
        n_samples=3000,
        n_features=40,
        n_informative=15,
        n_redundant=10,
        n_classes=3,
        weights=[0.10, 0.57, 0.33],
        random_state=42,
    )
    # Convertiamo in DataFrame per rispettare l'interfaccia dei selettori
    col_names = [f"feature_{i}" for i in range(40)]
    X_df = pd.DataFrame(X_demo, columns=col_names)
    # Le classi devono essere 1, 2, 3 (come nel dataset reale)
    y_s  = pd.Series(y_demo + 1)

    # Ricerca con parametri ridotti per il test (n_iter basso, no SFS)
    search = FeatureSelectionSearch(
        n_iter=20,
        cv=2,
        include_sfs=False,   # SFS escluso per velocità nel test
        verbose=1,
        output_dir=None,
    )
    search.fit(X_df, y_s)

    print(f"\nMiglior score CV: {search.get_best_score():.4f}")
    print(f"Parametri migliori:")
    best = search.get_best_params()
    print(f"  Selettore : {type(best['selector']).__name__}")
    print(f"  Modello   : {type(best['model']).__name__}")

    # Verifica predict
    y_pred = search.predict(X_df)
    print(f"\nPredizioni (prime 10): {y_pred[:10]}")
    print("\nTest completato con successo.")