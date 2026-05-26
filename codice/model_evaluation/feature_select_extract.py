"""
feature_select_extract.py
=========================
Implementazione dei selettori di feature per il progetto Richter's Predictor.

Secondo la tassonomia studiata, testiamo un metodo per ciascuna famiglia:
    - Filter pair-wise       → MutualInfoSelector   (Mutual Information)
    - Filter simultaneo      → ReliefFSelector       (ReliefF da skrebate)
    - Subset selection       → SFSSelector           (Sequential Forward Selection + DT)
    - Embedded               → EmbeddedDTSelector    (Feature importance DT + soglia)
    - Baseline               → AllFeaturesSelector   (nessuna selezione)

Tutti i selettori ereditano da sklearn BaseEstimator + TransformerMixin, quindi:
    - sono compatibili con sklearn Pipeline
    - supportano get_params() / set_params() (necessari per RandomizedSearchCV)
    - ottengono fit_transform() gratuitamente da TransformerMixin

La PCA è stata esclusa intenzionalmente: trasforma le feature in combinazioni
lineari non interpretabili, perdendo il legame con il dominio strutturale
degli edifici. Come rappresentante dei filter simultanei usiamo ReliefF,
che mantiene le feature originali e ne valuta le interazioni reciproche.
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    SequentialFeatureSelector,
)
from sklearn.tree import DecisionTreeClassifier

# ===========================================================================
# 0) BASELINE — nessuna selezione
# ===========================================================================

class AllFeaturesSelector(BaseEstimator, TransformerMixin):
    """
    Selettore di riferimento: restituisce il DataFrame invariato.

    Serve come baseline nel confronto finale: se nessun metodo di feature
    selection batte questo baseline, significa che la riduzione dimensionale
    non porta beneficio su questo dataset — informazione comunque utile.

    Non ha iperparametri da ottimizzare: il dizionario nella random search
    conterrà solo i parametri del modello.
    """

    def fit(self, X, y=None):
        """Memorizza i nomi delle colonne in input (nessuna selezione)."""
        # Salviamo i nomi delle colonne per poterli riapplicare in transform
        self.feature_names_in_ = list(X.columns)
        self.feature_names_selected_ = list(X.columns)
        return self

    def transform(self, X):
        """Restituisce X invariato (tutte le feature)."""
        return X[self.feature_names_selected_]

    def get_info(self) -> dict:
        """Report testuale sulla selezione effettuata."""
        n = len(self.feature_names_selected_) if hasattr(self, "feature_names_selected_") else 0
        return {
            "metodo": "AllFeatures",
            "feature_in_input": n,
            "feature_selezionate": n,
            "riduzione_pct": 0.0,
        }


# ===========================================================================
# 1) FILTER PAIR-WISE — Mutual Information
# ===========================================================================

class MutualInfoSelector(BaseEstimator, TransformerMixin):
    """
    Selettore filter pair-wise basato su Mutual Information.

    La MI misura quanta informazione una singola feature porta sul target,
    rispetto all'entropia totale del target. A differenza del coefficiente
    di Pearson, la MI cattura anche relazioni non lineari, ed è indicata
    quando si ha un mix di feature continue e binarie come nel nostro caso.

    È "pair-wise" perché valuta ogni feature INDIPENDENTEMENTE dal resto:
    non considera le interazioni tra feature (limite rispetto a ReliefF).

    Parameters
    ----------
    k : int
        Numero di feature da mantenere (le top-k per punteggio MI).
    random_state : int
        Seed per riproducibilità (la stima della MI usa componenti stocastiche).
    """

    def __init__(self, k: int = 30, random_state: int = 42, verbose: bool = True):
        # IMPORTANTE: i parametri del costruttore devono avere lo stesso nome
        # degli attributi di istanza — requisito di BaseEstimator per
        # get_params() / set_params() che usa l'introspezione.
        self.k = k
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        """Calcola i punteggi MI e seleziona le top-k feature."""
        self.feature_names_in_ = list(X.columns)

        # Non superiamo mai il numero di feature disponibili
        k_effettivo = min(self.k, X.shape[1])

        selector = SelectKBest(
            score_func=lambda X_, y_: mutual_info_classif(
                X_, y_, random_state=self.random_state
            ),
            k=k_effettivo,
        )
        selector.fit(X, y)

        # Salviamo i punteggi per eventuali grafici / report
        self.scores_ = pd.Series(selector.scores_, index=self.feature_names_in_)

        # Recuperiamo le feature selezionate tramite la maschera booleana
        mask = selector.get_support()
        self.feature_names_selected_ = self.scores_.index[mask].tolist()

        if self.verbose:
            print(f"  [MutualInfo] {len(self.feature_names_selected_)}"
                  f"/{len(self.feature_names_in_)} feature selezionate (k={k_effettivo}).")
        return self

    def transform(self, X):
        """Filtra X mantenendo solo le feature selezionate in fit()."""
        return X[self.feature_names_selected_]

    def get_info(self) -> dict:
        n_in  = len(self.feature_names_in_)
        n_out = len(self.feature_names_selected_)
        return {
            "metodo": "MutualInfo",
            "feature_in_input": n_in,
            "feature_selezionate": n_out,
            "riduzione_pct": round(100 * (1 - n_out / n_in), 1),
        }


# ===========================================================================
# 2) FILTER SIMULTANEO — ReliefF  (via skrebate)
# ===========================================================================

class ReliefFSelector(BaseEstimator, TransformerMixin):
    """
    Selettore filter simultaneo basato su ReliefF (implementazione skrebate).

    ReliefF valuta la rilevanza di una feature osservando, per ogni campione,
    quanto quella feature differisce dai "near-hit" (campione più vicino
    della stessa classe) e dai "near-miss" (campione più vicino di classi
    diverse). Una feature è rilevante se:
        - Varia POCO tra campioni della stessa classe  (near-hit simili)
        - Varia MOLTO tra campioni di classi diverse   (near-miss diversi)

    A differenza della Mutual Information, ReliefF è "simultaneo": considera
    le relazioni TRA feature, il che lo rende adatto quando si sospettano
    interazioni. Svantaggio: O(n²) sui campioni → usiamo n_samples per
    limitare il costo computazionale.

    Usiamo skrebate invece di un'implementazione custom: è mantenuta,
    testata e produce risultati consistenti con la letteratura.

    Parameters
    ----------
    k : int
        Numero di feature da mantenere.
    n_neighbors : int
        Numero di near-hit / near-miss per campione (ReliefF originale usa 10).
    n_samples : int
        Campioni su cui stimare i pesi (subsample per velocità su dataset grandi).
    random_state : int
        Seed per il campionamento.
    """

    def __init__(self, k: int = 30, n_neighbors: int = 10,
                 n_samples: int = 500, random_state: int = 42, verbose: bool = True):
        super().__init__()
        self.k = k
        self.n_neighbors = n_neighbors
        self.n_samples = n_samples
        self.random_state = random_state
        self.verbose = verbose
        self.scores_ = None

    def fit(self, x, y):
        self.feature_names_in_ = x.columns.tolist()

        # Conversione a numpy per velocità (indicizzazione più rapida)
        X_np = x.values.astype(float)
        y_np = np.asarray(y)
        n_campioni, n_feature = X_np.shape

        # Normalizzazione min-max in [0,1] come previsto da Relief.
        # Le feature già scalate passano sostanzialmente invariate,
        # le binarie restano 0/1.
        X_min = X_np.min(axis=0)
        X_range = X_np.max(axis=0) - X_min
        X_range[X_range == 0] = 1   # evita divisione per zero su colonne costanti
        X_norm = (X_np - X_min) / X_range

        # Campionamento casuale degli istanze su cui stimare i pesi
        rng = np.random.default_rng(self.random_state)
        n_campioni_usati = min(self.n_samples, n_campioni)
        indici = rng.choice(n_campioni, size=n_campioni_usati, replace=False)

        # Vettore dei pesi: uno per feature, inizializzato a zero
        pesi = np.zeros(n_feature)

        # Classi presenti e loro probabilità (usate per pesare i near-miss
        # multiclasse secondo la formulazione di ReliefF)
        classi_uniche = np.unique(y_np)
        prob_classi = {c: np.mean(y_np == c) for c in classi_uniche}

        for idx in indici:
            x_i = X_norm[idx]
            y_i = y_np[idx]

            # Calcolo distanza Manhattan (equivalente a Euclidea per feature
            # normalizzate in [0,1], ma più veloce)
            diff = np.abs(X_norm - x_i)
            dist = diff.sum(axis=1)
            dist[idx] = np.inf   # escludo il campione stesso dai vicini

            # NEAR-HIT: k più vicini della stessa classe
            mask_stessa_classe = (y_np == y_i)
            dist_hit = np.where(mask_stessa_classe, dist, np.inf)
            indici_hit = np.argsort(dist_hit)[:self.n_neighbors]

            # Aggiornamento del peso: una feature "buona" ha near-hit SIMILI,
            # quindi diff piccolo -> contributo negativo piccolo -> peso alto
            pesi -= diff[indici_hit].sum(axis=0) / (self.n_neighbors * n_campioni_usati)

            # NEAR-MISS: k più vicini per ogni classe diversa,
            # pesati per p(C)/(1-p(y_i)) come in ReliefF multiclasse
            for c in classi_uniche:
                if c == y_i:
                    continue
                mask_c = (y_np == c)
                dist_miss = np.where(mask_c, dist, np.inf)
                indici_miss = np.argsort(dist_miss)[:self.n_neighbors]

                peso_classe = prob_classi[c] / (1 - prob_classi[y_i])

                # Una feature "buona" ha near-miss DIVERSI,
                # quindi diff grande -> contributo positivo grande -> peso alto
                pesi += peso_classe * diff[indici_miss].sum(axis=0) / (self.n_neighbors * n_campioni_usati)

        # Salvo i punteggi
        self.scores_ = pd.Series(pesi, index=self.feature_names_in_)

        # Selezione top-k
        k_effettivo = min(self.k, n_feature)
        top_k_idx = np.argsort(pesi)[-k_effettivo:][::-1]   # decrescente
        self.feature_names_selected_ = [self.feature_names_in_[i] for i in top_k_idx]

        if self.verbose:
            print(f"  [ReliefF] {len(self.feature_names_selected_)}/{n_feature} feature selezionate "
                  f"(stimate su {n_campioni_usati} campioni).")
        return self


    def transform(self, X):
        """Filtra X mantenendo solo le feature selezionate in fit()."""
        return X[self.feature_names_selected_]

    def get_info(self) -> dict:
        n_in  = len(self.feature_names_in_)
        n_out = len(self.feature_names_selected_)
        return {
            "metodo": "ReliefF",
            "feature_in_input": n_in,
            "feature_selezionate": n_out,
            "riduzione_pct": round(100 * (1 - n_out / n_in), 1),
        }


# ===========================================================================
# 3) SUBSET SELECTION — Sequential Forward Selection
# ===========================================================================

class SFSSelector(BaseEstimator, TransformerMixin):
    """
    Wrapper per Sequential Forward Selection (SFS) di sklearn.

    Parte da un subset vuoto e aggiunge una feature alla volta, scegliendo
    ad ogni iterazione quella che massimizza lo score di cross-validation
    del modello interno (DecisionTree leggero).

    Soffre di "nesting effect": una feature aggiunta non può essere rimossa.
    In compenso è computazionalmente più economico di SBS perché i primi
    step lavorano su subset piccoli.

    Come stimatore interno usiamo un DecisionTree poco profondo: veloce,
    informativo sulle interazioni, coerente con i metodi embedded del progetto.
    Non ha senso usare RF o KNN qui — SFS richiede molti training ripetuti
    (k × cv volte), e un modello pesante renderebbe la ricerca impraticabile.

    Parameters
    ----------
    k : int
        Numero di feature da selezionare.
    cv : int
        Fold di cross-validation interna per valutare ogni subset candidato.
    max_depth_dt : int
        Profondità del DecisionTree interno. Valori bassi = valutazione veloce
        ma meno accurata; valori alti = più lento ma più informativo.
    random_state : int
        Seed per il DT interno.
    n_jobs : int
        Parallelismo (-1 = tutti i core disponibili).
    """

    def __init__(self, k: int = 20, cv: int = 3,
                 max_depth_dt: int = 8, random_state: int = 42, n_jobs: int = -1, verbose: bool = True):
        self.k = k
        self.cv = cv
        self.max_depth_dt = max_depth_dt
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y):
        """Esegue la ricerca sequenziale e memorizza le feature selezionate."""
        self.feature_names_in_ = list(X.columns)
        k_effettivo = min(self.k, X.shape[1])

        stimatore = DecisionTreeClassifier(
            max_depth=self.max_depth_dt,
            random_state=self.random_state,
        )

        sfs = SequentialFeatureSelector(
            estimator=stimatore,
            n_features_to_select=k_effettivo,
            direction="forward",
            scoring="f1_micro",     # coerente con la metrica ufficiale DrivenData
            cv=self.cv,
            n_jobs=self.n_jobs,
        )

        if self.verbose:
            print(f"  [SFS] Avvio (k={k_effettivo}, cv={self.cv})... "
                  f"(può richiedere qualche minuto)")
        sfs.fit(X, y)

        mask = sfs.get_support()
        self.feature_names_selected_ = pd.Index(self.feature_names_in_)[mask].tolist()

        if self.verbose:
            print(f"  [SFS] {len(self.feature_names_selected_)}"
                  f"/{len(self.feature_names_in_)} feature selezionate.")
        return self

    def transform(self, X):
        """Filtra X mantenendo solo le feature selezionate in fit()."""
        return X[self.feature_names_selected_]

    def get_info(self) -> dict:
        n_in  = len(self.feature_names_in_)
        n_out = len(self.feature_names_selected_)
        return {
            "metodo": "SFS",
            "feature_in_input": n_in,
            "feature_selezionate": n_out,
            "riduzione_pct": round(100 * (1 - n_out / n_in), 1),
        }


# ===========================================================================
# 4) EMBEDDED — Feature importance da Decision Tree
# ===========================================================================

class EmbeddedDTSelector(BaseEstimator, TransformerMixin):
    """
    Metodo embedded: allena un Decision Tree e mantiene le feature con
    importanza superiore a una soglia.

    È "embedded" perché la selezione avviene all'interno del processo
    di training del modello stesso (non è un filtro esterno). Il DT usa
    naturalmente solo le feature più discriminative per costruire i nodi,
    quindi le importanze riflettono utilità reale sul task.

    Vantaggi rispetto ai filter:
        - Cattura interazioni non lineari (MI è pair-wise, questo no)
        - Un solo training (più veloce di SFS)

    Nota: la soglia "mean" / "median" produce un numero di feature variabile
    in base alla distribuzione delle importanze su quel subset — comportamento
    atteso e documentato.

    Parameters
    ----------
    soglia : float or "mean" or "median"
        Soglia sull'importanza normalizzata.
        "mean"   → media delle importanze (seleziona circa metà feature)
        "median" → mediana (più robusta agli outlier di importanza)
        float    → soglia numerica fissa (es. 0.01)
    max_depth : int
        Profondità del DT. Troppo basso sottovaluta feature minori;
        troppo alto introduce rumore da overfitting.
    random_state : int
        Seed.
    """

    def __init__(self, soglia="mean", max_depth: int = 12, random_state: int = 42, verbose: bool = True):
        self.soglia = soglia
        self.max_depth = max_depth
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        """Allena il DT, calcola le importanze e applica la soglia."""
        self.feature_names_in_ = list(X.columns)

        dt = DecisionTreeClassifier(
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        dt.fit(X, y)

        importanze = pd.Series(dt.feature_importances_, index=self.feature_names_in_)
        self.importanze_ = importanze

        # Determina la soglia numerica effettiva
        if self.soglia == "mean":
            soglia_val = importanze.mean()
        elif self.soglia == "median":
            soglia_val = importanze.median()
        elif isinstance(self.soglia, (int, float)):
            soglia_val = float(self.soglia)
        else:
            raise ValueError(
                f"Soglia non valida: '{self.soglia}'. "
                f"Usare 'mean', 'median' o un valore float."
            )

        self.soglia_valore_ = soglia_val
        self.feature_names_selected_ = importanze[importanze > soglia_val].index.tolist()

        # Safety net: se la soglia elimina tutto, teniamo le top-10.
        # Evita crash a valle quando il selettore restituisce 0 colonne.
        if len(self.feature_names_selected_) == 0:
            if self.verbose:
                print(f"  [EmbeddedDT] Soglia troppo aggressiva -> fallback top-10.")
            self.feature_names_selected_ = importanze.nlargest(10).index.tolist()

        if self.verbose:
            print(f"  [EmbeddedDT] {len(self.feature_names_selected_)}"
                  f"/{len(self.feature_names_in_)} feature selezionate "
                  f"(soglia={soglia_val:.5f}).")
        return self

    def transform(self, X):
        """Filtra X mantenendo solo le feature selezionate in fit()."""
        return X[self.feature_names_selected_]

    def get_info(self) -> dict:
        n_in  = len(self.feature_names_in_)
        n_out = len(self.feature_names_selected_)
        return {
            "metodo": "EmbeddedDT",
            "feature_in_input": n_in,
            "feature_selezionate": n_out,
            "riduzione_pct": round(100 * (1 - n_out / n_in), 1),
            "soglia_valore": round(self.soglia_valore_, 6),
        }


# ===========================================================================
# FACTORY — restituisce il selettore richiesto per nome
# ===========================================================================

def crea_selector(nome: str, **kwargs) -> BaseEstimator:
    """
    Factory per creare il selettore richiesto per nome.

    Permette all'orchestratore di istanziare selettori senza if/elif,
    semplicemente ciclando su una lista di nomi.

    Parameters
    ----------
    nome : str
        Uno tra: 'all', 'mutual_info', 'relief', 'sfs', 'embedded_dt'.
    **kwargs
        Iperparametri da passare al costruttore del selettore.

    Returns
    -------
    BaseEstimator
        Istanza del selettore richiesto.

    Examples
    --------
    >>> sel = crea_selector("mutual_info", k=25)
    >>> sel = crea_selector("relief", k=20, n_neighbors=15)
    """
    mapping = {
        "all":         AllFeaturesSelector,
        "mutual_info": MutualInfoSelector,
        "relief":      ReliefFSelector,
        "sfs":         SFSSelector,
        "embedded_dt": EmbeddedDTSelector,
    }

    nome = nome.lower().strip()
    if nome not in mapping:
        raise ValueError(
            f"Selettore sconosciuto: '{nome}'. "
            f"Disponibili: {list(mapping.keys())}"
        )

    return mapping[nome](**kwargs)