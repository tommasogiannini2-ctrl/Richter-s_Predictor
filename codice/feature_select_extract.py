"""

Secondo la tassonomia studiata, testiamo un metodo per ciascuna famiglia:
    - Filter pair-wise       -> Mutual Information
    - Filter simultaneo      -> ReliefF
    - Subset selection       -> Sequential Forward Selection (SFS) con Decision Tree
    - Embedded               -> Feature importance da Decision Tree con soglia
    - Baseline (riferimento) -> AllFeatures (nessuna selezione)

Tutti i selector condividono la stessa interfaccia (fit/transform) in stile
scikit-learn, quindi sono intercambiabili nel resto della pipeline.

La PCA è stata esclusa intenzionalmente: trasforma le feature in combinazioni
lineari non interpretabili, perdendo il legame con il dominio strutturale
degli edifici. Come rappresentante dei filter simultanei usiamo ReliefF,
che mantiene le feature originali.
"""

import numpy as np
import pandas as pd

from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    SequentialFeatureSelector,
)
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------------------------
# CLASSE BASE (interfaccia comune)
# ---------------------------------------------------------------------------

class _BaseSelector:
    """
    Interfaccia comune a tutti i selector.

    Tutti i selector devono esporre:
        - fit(X, y)          -> impara quali feature tenere
        - transform(X)       -> applica la selezione
        - fit_transform(X,y) -> shortcut combinato
        - feature_names_selected_ -> lista delle feature mantenute (per report)

    Usare una classe base astratta mantiene coerente la pipeline e permette
    a validation.py e addestramento_modelli.py di trattare tutti i selector
    nello stesso modo, indipendentemente dal metodo specifico.
    """

    def __init__(self):
        # Inizializzati da fit()
        self.feature_names_selected_ = None
        self.feature_names_in_ = None

    def fit(self, x, y):
        raise NotImplementedError("Implementare nella sottoclasse.")

    def transform(self, x):
        """Seleziona le colonne che fit() ha memorizzato."""
        if self.feature_names_selected_ is None:
            raise RuntimeError("Il selector non è stato addestrato: chiamare fit() prima.")
        # Usiamo i nomi delle colonne per essere robusti rispetto all'ordine
        # (X in ingresso potrebbe avere colonne in un altro ordine)
        return x[self.feature_names_selected_].copy()

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)

    def get_info(self) -> dict:
        """Informazioni testuali sulla selezione, utili per il report."""
        n_in = len(self.feature_names_in_) if self.feature_names_in_ is not None else 0
        n_out = len(self.feature_names_selected_) if self.feature_names_selected_ is not None else 0
        return {
            "metodo": self.__class__.__name__,
            "feature_in_input": n_in,
            "feature_selezionate": n_out,
            "riduzione_pct": 100 * (1 - n_out / n_in) if n_in > 0 else 0,
            "feature_mantenute": self.feature_names_selected_,
        }


# ---------------------------------------------------------------------------
# 0) BASELINE -- nessuna selezione
# ---------------------------------------------------------------------------

class AllFeaturesSelector(_BaseSelector):
    """
    Selector di riferimento: tiene TUTTE le feature.

    Serve come baseline nel confronto finale: se nessun metodo di FR batte
    questo baseline, vuol dire che la feature reduction non porta beneficio
    su questo dataset (informazione comunque utile da riportare).
    """

    def fit(self, x, y):
        self.feature_names_in_ = x.columns.tolist()
        self.feature_names_selected_ = x.columns.tolist()
        return self


# ---------------------------------------------------------------------------
# 1) FILTER PAIR-WISE -- Mutual Information
# ---------------------------------------------------------------------------

class MutualInfoSelector(_BaseSelector):
    """
    Filter pair-wise basato su Mutual Information.

    La MI misura quanta informazione una feature porta sul target (rispetto
    all'entropia totale del target). A differenza del coefficiente di
    Pearson, la MI cattura anche relazioni non lineari, ed è indicata
    quando si ha un mix di feature continue e binarie come nel nostro caso.

    Parameters
    ----------
    k : int
        Numero di feature da mantenere (le top-k per punteggio MI).
    random_state : int
        Seed per riproducibilità (la stima della MI usa componenti stocastiche).
    """

    def __init__(self, k: int = 30, random_state: int = 42):
        super().__init__()
        self.k = k
        self.random_state = random_state
        self.scores_ = None   # salviamo i punteggi MI per eventuali grafici

    def fit(self, x, y):
        self.feature_names_in_ = x.columns.tolist()

        # Non superiamo mai il numero di feature disponibili
        k_effettivo = min(self.k, x.shape[1])

        # mutual_info_classif con seed per riproducibilità
        selector = SelectKBest(
            score_func=lambda x_, y_: mutual_info_classif(
                x_, y_, random_state=self.random_state
            ),
            k=k_effettivo,
        )
        selector.fit(x, y)

        # Salvo i punteggi e le feature selezionate
        self.scores_ = pd.Series(selector.scores_, index=self.feature_names_in_)
        mask = selector.get_support()

        self.feature_names_selected_ = self.scores_.index[mask].tolist()

        print(f"  [MutualInfo] {len(self.feature_names_selected_)}/{len(self.feature_names_in_)} feature selezionate.")
        return self


# ---------------------------------------------------------------------------
# 2) FILTER SIMULTANEO -- ReliefF
# ---------------------------------------------------------------------------

class ReliefFSelector(_BaseSelector):
    """
    Filter simultaneo basato su ReliefF.

    ReliefF valuta la rilevanza di una feature osservando, per ogni campione,
    quanto quella feature differisce dai "near-hit" (campione più vicino
    della stessa classe) e dai "near-miss" (campione più vicino di classi
    diverse). Una feature è rilevante se:
        - Varia POCO tra campioni della stessa classe (near-hit simili)
        - Varia MOLTO tra campioni di classi diverse (near-miss diversi)

    A differenza della Mutual Information, ReliefF considera le relazioni
    TRA feature (è "simultaneo", non "pair-wise"), il che lo rende adatto
    quando ci si aspettano interazioni. Svantaggio: computazionalmente
    oneroso (O(n^2) sui campioni nel caso peggiore).

    NOTA: poiché ReliefF non è in sklearn, implementiamo una versione
    semplificata. Alternative più complete sono nel pacchetto `skrebate`
    (non incluso in requirements.txt per mantenere le dipendenze minime).

    Parameters
    ----------
    k : int
        Numero di feature da mantenere.
    n_neighbors : int
        Numero di near-hit e near-miss da considerare per ogni campione.
    n_samples : int
        Numero di campioni casuali su cui stimare i pesi. Su dataset grandi
        campioniamo per accelerare (Relief originale usa tutti i campioni).
    random_state : int
        Seed per riproducibilità.
    """

    def __init__(self, k: int = 30, n_neighbors: int = 10,
                 n_samples: int = 500, random_state: int = 42):
        super().__init__()
        self.k = k
        self.n_neighbors = n_neighbors
        self.n_samples = n_samples
        self.random_state = random_state
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

        print(f"  [ReliefF] {len(self.feature_names_selected_)}/{n_feature} feature selezionate "
              f"(stimate su {n_campioni_usati} campioni).")
        return self


# ---------------------------------------------------------------------------
# 3) SUBSET SELECTION -- Sequential Forward Selection (SFS)
# ---------------------------------------------------------------------------

class SFSSelector(_BaseSelector):
    """
    Wrapper Sequential Forward Selection.

    Parte da un subset vuoto e aggiunge una feature alla volta, scegliendo
    ad ogni iterazione quella che aumenta di più lo score di cross-validation
    del modello interno.

    Soffre di "nesting effect": una feature aggiunta non può più essere
    rimossa. In compenso è computazionalmente più economico di SBS perché
    addestra il modello su subset piccoli all'inizio.

    Come modello interno usiamo un DecisionTree poco profondo: veloce e
    abbastanza informativo sulle interazioni. Non ha senso usare qui un
    modello pesante come MLP, perché SFS richiede molti training ripetuti.

    Parameters
    ----------
    k : int
        Numero di feature da selezionare.
    cv : int
        Fold di cross-validation interna per valutare ogni subset candidato.
    random_state : int
        Seed per il DT interno.
    n_jobs : int
        Parallelismo (-1 = tutti i core).
    """

    def __init__(self, k: int = 20, cv: int = 3,
                 random_state: int = 42, n_jobs: int = -1):
        super().__init__()
        self.k = k
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, x, y):
        self.feature_names_in_ = x.columns.tolist()
        k_effettivo = min(self.k, x.shape[1])

        # Modello "valutatore" interno: DT leggero, veloce da addestrare
        # su subset piccoli e sensibile alle interazioni tra feature.
        stimatore = DecisionTreeClassifier(
            max_depth=8,
            random_state=self.random_state,
        )

        sfs = SequentialFeatureSelector(
            estimator=stimatore,
            n_features_to_select=k_effettivo,
            direction="forward",
            scoring="f1_micro",     # coerente con la metrica ufficiale
            cv=self.cv,
            n_jobs=self.n_jobs,
        )

        print(f"  [SFS] Avvio ricerca sequenziale (k={k_effettivo}, cv={self.cv})...")
        print(f"         ATTENZIONE: può richiedere qualche minuto.")
        sfs.fit(x, y)

        mask = sfs.get_support()

        self.feature_names_selected_ = pd.Index(self.feature_names_in_)[mask].tolist()


        print(f"  [SFS] {len(self.feature_names_selected_)}/{len(self.feature_names_in_)} feature selezionate.")
        return self


# ---------------------------------------------------------------------------
# 4) EMBEDDED -- Feature importance da Decision Tree
# ---------------------------------------------------------------------------

class EmbeddedDTSelector(_BaseSelector):
    """
    Metodo embedded: allena un Decision Tree completo, poi tiene solo le
    feature con importanza superiore a una soglia.

    Il DT di per sé fa già selezione interna (non usa feature inutili),
    ma qui la usiamo esplicitamente per ridurre lo spazio prima di passare
    ai modelli successivi (KNN, MLP, un altro DT tuned).

    Vantaggi:
        - Velocissimo (un solo training)
        - Cattura interazioni non lineari
        - Coerente con la famiglia embedded del PDF

    Parameters
    ----------
    soglia : float or 'mean' or 'median'
        Soglia sull'importanza. 'mean' = media delle importanze (default
        ragionevole, tende a selezionare circa metà delle feature).
    max_depth : int
        Profondità del DT interno. Un valore troppo basso sottovaluta
        le feature minori; troppo alto rumorizza.
    random_state : int
        Seed.
    """

    def __init__(self, soglia="mean", max_depth: int = 12, random_state: int = 42):
        super().__init__()
        self.soglia = soglia
        self.max_depth = max_depth
        self.random_state = random_state
        self.importanze_ = None
        self.soglia_valore_ = None   # soglia numerica effettiva usata

    def fit(self, x, y):
        self.feature_names_in_ = x.columns.tolist()

        dt = DecisionTreeClassifier(
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        dt.fit(x, y)

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
            raise ValueError(f"Soglia non valida: {self.soglia}")

        self.soglia_valore_ = soglia_val
        self.feature_names_selected_ = importanze[importanze > soglia_val].index.tolist()

        # Safety net: se la soglia taglia via tutto, teniamo la top-10
        # (evita errori a valle quando il selector restituisce 0 colonne).
        if len(self.feature_names_selected_) == 0:
            print(f"  [EmbeddedDT] Soglia troppo aggressiva, fallback su top-10.")
            self.feature_names_selected_ = importanze.nlargest(10).index.tolist()

        print(f"  [EmbeddedDT] {len(self.feature_names_selected_)}/{len(self.feature_names_in_)} feature "
              f"selezionate (soglia={soglia_val:.5f}).")
        return self


# ---------------------------------------------------------------------------
# FACTORY -- restituisce il selector richiesto per nome
# ---------------------------------------------------------------------------

def crea_selector(n : str, **kwargs) -> _BaseSelector:
    """
    Factory per creare il selector richiesto per nome.

    Questa factory rende il codice dell'orchestratore (addestramento_modelli.py)
    più pulito: invece di fare if/elif per ogni metodo, si cicla su una lista
    di nomi e si chiama crea_selector().

    Parameters
    ----------
    n : str
        Uno di: 'all', 'mutual_info', 'relief', 'sfs', 'embedded_dt'.
    **kwargs : dict
        Argomenti specifici del selector (es. k=30, cv=3, ...).

    Returns
    -------
    _BaseSelector
        Istanza del selector richiesto.

    """
    nome_metodo = n.lower().strip()
    mapping = {
        "all":          AllFeaturesSelector,
        "mutual_info":  MutualInfoSelector,
        "relief":       ReliefFSelector,
        "sfs":          SFSSelector,
        "embedded_dt":  EmbeddedDTSelector,
    }

    if nome_metodo not in mapping:
        raise ValueError(
            f"Selector sconosciuto: '{nome_metodo}'. "
            f"Disponibili: {list(mapping.keys())}"
        )

    return mapping[nome_metodo](**kwargs)
