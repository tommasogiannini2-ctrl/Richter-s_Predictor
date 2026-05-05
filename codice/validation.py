import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from feature_select_extract import crea_selector

class FeatureSelectionSearch:
    """
    Strategia di ricerca per la selezione delle feature.
    Permette di testare diversi metodi di feature selection (e i loro iperparametri)
    su un set di validazione per determinare quale garantisce le prestazioni migliori.
    """

    def __init__(self, estimator=None, metric=f1_score, metric_kwargs=None):
        """
        Parameters
        ----------
        estimator : estimator object
            Modello da usare per valutare la bontà del subset di feature.
            Di default usa un DecisionTreeClassifier.
        metric : callable
            Funzione di metrica (es. f1_score) per valutare le predizioni.
        metric_kwargs : dict
            Argomenti aggiuntivi per la funzione di metrica (es. {'average': 'micro'}).
        """
        self.estimator = estimator if estimator is not None else DecisionTreeClassifier(random_state=42)
        self.metric = metric
        self.metric_kwargs = metric_kwargs if metric_kwargs is not None else {"average": "micro"}
        
        self.best_selector_name_ = None
        self.best_selector_ = None
        self.best_score_ = -1.0
        self.results_ = []

    def search(self, X_train, y_train, X_val, y_val, selectors_config: dict = None) -> pd.DataFrame:
        """
        Esegue la ricerca testando ogni selettore definito in selectors_config.

        Parameters
        ----------
        X_train, y_train : Dati di addestramento.
        X_val, y_val     : Dati di validazione per calcolare lo score.
        selectors_config : dict, optional
            Dizionario con i nomi dei selettori e i relativi iperparametri.
            Se None, usa una configurazione di default con tutti i selettori.
            Es. {
                "mutual_info": {"k": 20},
                "relief": {"k": 20},
                "sfs": {"k": 15, "cv": 3},
                "embedded_dt": {"soglia": "mean"},
                "all": {}
            }

        Returns
        -------
        pd.DataFrame
            Un DataFrame con i risultati di ciascun selettore testato.
        """
        if selectors_config is None:
            selectors_config = {
                "all":          {},
                "mutual_info":  {"k": 30},
                "relief":       {"k": 30, "n_samples": 500},
                "sfs":          {"k": 20, "cv": 3},
                "embedded_dt":  {"soglia": "mean"}
            }

        print(f"\n{'=' * 60}")
        print(f"  AVVIO RICERCA STRATEGIA DI FEATURE SELECTION")
        print(f"{'=' * 60}")

        self.results_ = []
        self.best_score_ = -1.0

        for name, kwargs in selectors_config.items():
            print(f"\n[Test] Valutazione selettore: '{name}' con parametri: {kwargs}")
            try:
                # 1. Crea e addestra il selettore sui dati di training
                selector = crea_selector(name, **kwargs)
                X_train_sel = selector.fit_transform(X_train, y_train)
                
                # 2. Applica la trasformazione al set di validazione
                X_val_sel = selector.transform(X_val)

                # 3. Addestra il modello valutatore sui dati di training ridotti
                self.estimator.fit(X_train_sel, y_train)

                # 4. Predice e valuta sul validation set
                y_pred = self.estimator.predict(X_val_sel)
                score = self.metric(y_val, y_pred, **self.metric_kwargs)

                n_features = X_train_sel.shape[1]
                print(f"  -> Score ({self.metric.__name__}): {score:.4f} (Feature mantenute: {n_features})")

                # Salva i risultati
                self.results_.append({
                    "selector": name,
                    "kwargs": kwargs,
                    "score": score,
                    "n_features": n_features
                })

                # Aggiorna il migliore
                if score > self.best_score_:
                    self.best_score_ = score
                    self.best_selector_name_ = name
                    self.best_selector_ = selector

            except Exception as e:
                print(f"  -> Errore durante la valutazione del selettore '{name}': {e}")
                self.results_.append({
                    "selector": name,
                    "kwargs": kwargs,
                    "score": None,
                    "n_features": None
                })

        print(f"\n{'=' * 60}")
        if self.best_selector_ is not None:
            print(f"  Miglior selettore trovato: {self.best_selector_name_}")
            print(f"  Miglior score: {self.best_score_:.4f}")
        else:
            print("  Nessun selettore ha completato con successo la ricerca.")
        print(f"{'=' * 60}\n")

        return pd.DataFrame(self.results_)

    def get_best_selector(self):
        """Restituisce l'istanza del miglior selettore trovato."""
        if self.best_selector_ is None:
            raise RuntimeError("Ricerca non ancora eseguita o fallita. Chiamare search() prima.")
        return self.best_selector_

    def transform_with_best(self, X):
        """Applica la trasformazione del miglior selettore al dataset fornito."""
        return self.get_best_selector().transform(X)
