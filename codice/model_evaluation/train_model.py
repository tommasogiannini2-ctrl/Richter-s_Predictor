"""
train_model.py
==============
Addestramento del modello finale e generazione predizioni per Richter's Predictor.

Questo modulo espone tre interfacce:

  1. run(...)         — funzione chiamabile programmaticamente da main.py
                        per l'addestramento e la valutazione completa.
  2. predict_only(...) — funzione chiamabile da main.py quando si salta il
                        training per generare le predizioni partendo dal pkl.
  3. main()           — entry point CLI che legge gli argomenti da riga di comando
                        e delega tutto a run().
"""

import argparse
import os
import sys
import joblib # Sostituito pickle con joblib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from codice.model_evaluation.evaluation import ModelEvaluator
except ModuleNotFoundError:
    from model_evaluation.evaluation import ModelEvaluator


# ===========================================================================
# COSTRUZIONE DEL MODELLO
# ===========================================================================

def _build_model(
    model: str  = "rf",
    n_estimators: int  = 300,
    max_depth          = None,
    min_samples_leaf: int = 1,
    class_weight       = None,
    n_neighbors: int   = 7,
    weights: str       = "distance",
    metric: str        = "euclidean",
    learning_rate: float = 1.0,
    base_estimator_max_depth: int = 1,
):
    """
    Istanzia il modello scelto con i parametri forniti.

    Parameters
    ----------
    model : str
        'rf' per RandomForest, 'knn' per KNeighborsClassifier, 'ada' per AdaBoostClassifier.
    Gli altri parametri sono gli iperparametri specifici del modello.

    Returns
    -------
    estimator : sklearn estimator pronto per il fit.
    """
    if model == "rf":
        cw = class_weight if class_weight not in [None, "None"] else None
        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=cw,
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )
        print(f"  Modello: RandomForestClassifier")
        print(f"    n_estimators     = {n_estimators}")
        print(f"    max_depth        = {max_depth}")
        print(f"    min_samples_leaf = {min_samples_leaf}")
        print(f"    class_weight     = {cw}")
    elif model == "knn":
        estimator = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            n_jobs=-1,
        )
        print(f"  Modello: KNeighborsClassifier")
        print(f"    n_neighbors = {n_neighbors}")
        print(f"    weights     = {weights}")
        print(f"    metric      = {metric}")
    elif model == "ada":
        base_est = DecisionTreeClassifier(max_depth=base_estimator_max_depth, random_state=42)
        estimator = AdaBoostClassifier(
            estimator=base_est,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42,
        )
        print(f"  Modello: AdaBoostClassifier")
        print(f"    n_estimators   = {n_estimators}")
        print(f"    learning_rate  = {learning_rate}")
        print(f"    base_estimator = DecisionTreeClassifier(max_depth={base_estimator_max_depth})")
    else:
        raise ValueError(f"Modello '{model}' non supportato.")

    return estimator


# ===========================================================================
# FUNZIONE PRINCIPALE — chiamabile da main.py o da CLI
# ===========================================================================

def run(
    model: str        = "rf",
    output_dir: str   = "../output",
    risultati_dir: str = None,
    dataset_dir: str = None,
    no_proba: bool    = False,
    # iperparametri RF
    n_estimators: int = 300,
    max_depth         = None,
    min_samples_leaf: int = 1,
    class_weight      = None,
    # iperparametri KNN
    n_neighbors: int  = 7,
    weights: str      = "distance",
    metric: str       = "euclidean",
    # iperparametri AdaBoost
    learning_rate: float = 1.0,
    base_estimator_max_depth: int = 1,
):
    """
    Esegue l'intero flusso di training, valutazione e submission.
    """
    out_dir  = os.path.abspath(output_dir)
    dataset_dir = dataset_dir or os.path.join(out_dir, 'dataset')
    risultati_dir = risultati_dir or os.path.join(out_dir, 'risultati')
    eval_dir = os.path.join(out_dir, "eval")
    for d in [dataset_dir, risultati_dir, eval_dir]:
        os.makedirs(d, exist_ok=True)

    # ======================================================================
    # FASE 1 — CARICAMENTO DATASET FINALI
    # ======================================================================
    print(f"\n{'=' * 60}")
    print(f"  TRAINING — FASE 1: CARICAMENTO DATASET FINALI")
    print(f"{'=' * 60}")

    path_train = os.path.join(dataset_dir, "train_finale.csv")
    path_val   = os.path.join(dataset_dir, "val_finale.csv")
    path_test  = os.path.join(dataset_dir, "test_finale.csv")

    for p in [path_train, path_val, path_test]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"File non trovato: {p}\n"
                "  Assicurarsi che main.py sia stato eseguito correttamente."
            )

    df_train = pd.read_csv(path_train)
    df_val   = pd.read_csv(path_val)
    df_test  = pd.read_csv(path_test)

    print(f"  train_finale.csv  : {df_train.shape}")
    print(f"  val_finale.csv    : {df_val.shape}")
    print(f"  test_finale.csv   : {df_test.shape}")

    if "damage_grade" not in df_train.columns:
        raise ValueError("Colonna 'damage_grade' non trovata in train_finale.csv.")

    X_train = df_train.drop(columns=["damage_grade"])
    y_train = df_train["damage_grade"]

    X_val = df_val.drop(columns=["damage_grade"])
    y_val = df_val["damage_grade"]

    has_test_labels = "damage_grade" in df_test.columns
    X_test = df_test.drop(columns=["damage_grade"], errors="ignore")
    y_test = df_test["damage_grade"] if has_test_labels else None

    print(f"\n  Feature dopo selezione : {X_train.shape[1]}")
    print(f"  Distribuzione target (train):")
    for cls, cnt in y_train.value_counts().sort_index().items():
        print(f"    Classe {cls}: {cnt:>7,} ({cnt / len(y_train) * 100:.1f}%)")

    # ======================================================================
    # FASE 2 — ADDESTRAMENTO MODELLO
    # ======================================================================
    print(f"\n{'=' * 60}")
    print(f"  TRAINING — FASE 2: ADDESTRAMENTO MODELLO")
    print(f"{'=' * 60}")

    estimator = _build_model(
        model=model,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        learning_rate=learning_rate,
        base_estimator_max_depth=base_estimator_max_depth,
    )

    print(f"\n  Addestramento su {len(X_train):,} campioni...")
    estimator.fit(X_train, y_train)
    print(f"  Addestramento completato.")

    # ======================================================================
    # FASE 3 — VALUTAZIONE SU VALIDATION SET
    # ======================================================================
    print(f"\n{'=' * 60}")
    print(f"  TRAINING — FASE 3: VALUTAZIONE SU VALIDATION SET")
    print(f"{'=' * 60}")

    y_pred_val  = estimator.predict(X_val)
    y_score_val = estimator.predict_proba(X_val) if (not no_proba and hasattr(estimator, "predict_proba")) else None

    evaluator_val = ModelEvaluator(
        y_true=y_val,
        y_pred=y_pred_val,
        y_score=y_score_val,
        output_dir=os.path.join(eval_dir, "validation"),
    )
    metriche_val = evaluator_val.valuta_tutto()
    print(f"\n  [Validation] Micro-F1: {metriche_val['micro_f1']:.4f}")

    # ======================================================================
    # FASE 3.5 — SALVATAGGIO CONDIZIONALE DEL MODELLO MIGLIORE
    # ======================================================================
    print(f"\n{'=' * 60}")
    print(f"  TRAINING — FASE 3.5: SALVATAGGIO CONDIZIONALE MODELLO")
    print(f"{'=' * 60}")

    model_path = os.path.join(risultati_dir, "model_finale.pkl")
    best_score_path = os.path.join(risultati_dir, "model_finale_best_micro_f1.txt")
    current_micro_f1 = metriche_val['micro_f1']

    previous_best_micro_f1 = -1.0
    if os.path.exists(best_score_path):
        try:
            with open(best_score_path, "r") as f:
                previous_best_micro_f1 = float(f.read().strip())
            print(f"  Score precedente (Micro-F1): {previous_best_micro_f1:.4f}")
        except Exception as e:
            print(f"  [Avviso] Errore nel leggere il best score precedente: {e}. Si assume -1.0.")

    print(f"  Score attuale (Micro-F1): {current_micro_f1:.4f}")

    if current_micro_f1 > previous_best_micro_f1:
        print(f"  Nuovo miglior modello trovato! Salvataggio...")

        # --- NUOVA LOGICA: SALVIAMO MODELLO + COLONNE SELEZIONATE ---
        payload_da_salvare = {
            "model": estimator,
            "selected_features": X_train.columns.tolist()  # La mappa esatta delle feature scelte
        }

        joblib.dump(payload_da_salvare, model_path)
        # -------------------------------------------------------------

        # Aggiorna il file del punteggio
        with open(best_score_path, "w") as f:
            f.write(f"{current_micro_f1:.4f}")

        print(f"  Modello e Feature Selection salvati in: {model_path}")
        print(f"  Nuovo miglior score salvato: {best_score_path}")
    else:
        print(f"  Il modello attuale ({current_micro_f1:.4f}) non è migliore del precedente ({previous_best_micro_f1:.4f}). Non salvato.")

    # ======================================================================
    # FASE 4 — VALUTAZIONE SU TEST SET INTERNO
    # ======================================================================
    metriche_test = None

    if has_test_labels and y_test is not None:
        print(f"\n{'=' * 60}")
        print(f"  TRAINING — FASE 4: VALUTAZIONE SU TEST SET INTERNO")
        print(f"{'=' * 60}")

        y_pred_test  = estimator.predict(X_test)
        y_score_test = estimator.predict_proba(X_test) if (not no_proba and hasattr(estimator, "predict_proba")) else None

        evaluator_test = ModelEvaluator(
            y_true=y_test,
            y_pred=y_pred_test,
            y_score=y_score_test,
            output_dir=os.path.join(eval_dir, "test"),
        )
        metriche_test = evaluator_test.valuta_tutto()
        print(f"\n  [Test interno] Micro-F1: {metriche_test['micro_f1']:.4f}")
    else:
        print(f"\n  [Info] Test set senza etichette: valutazione saltata.")

    # ======================================================================
    # FASE 5 — GENERAZIONE PREDIZIONI PER SUBMISSION DRIVENDATA
    # ======================================================================
    print(f"\n{'=' * 60}")
    print(f"  TRAINING — FASE 5: GENERAZIONE SUBMISSION")
    print(f"{'=' * 60}")

    path_test_uff = os.path.join(dataset_dir, "test_ufficiale_finale.csv")

    if not os.path.exists(path_test_uff):
        print(f"  [Avviso] {path_test_uff} non trovato: submission saltata.")
    else:
        df_test_uff  = pd.read_csv(path_test_uff)
        building_ids = df_test_uff["building_id"].copy()

        df_sub_prep = df_test_uff.drop(columns=["building_id"])
        df_sub_prep = df_sub_prep.reindex(columns=X_train.columns, fill_value=0)

        print(f"  Predizione su {len(df_sub_prep):,} edifici...")
        y_sub_pred = estimator.predict(df_sub_prep)

        submission = pd.DataFrame({
            "building_id":  building_ids.values,
            "damage_grade": y_sub_pred,
        })
        sub_path = os.path.join(risultati_dir, "submission.csv")
        submission.to_csv(sub_path, index=False)

        print(f"  Submission salvata: {sub_path}")
        print(f"  Distribuzione predizioni:")
        for cls, cnt in submission["damage_grade"].value_counts().sort_index().items():
            print(f"    Classe {cls}: {cnt:>7,} ({cnt / len(submission) * 100:.1f}%)")

    # ======================================================================
    # RIEPILOGO FINALE
    # ======================================================================
    print(f"\n{'=' * 60}")
    print(f"  RIEPILOGO METRICHE FINALI")
    print(f"{'=' * 60}")
    print(f"  {'Modello:':<35} {type(estimator).__name__}")
    print(f"  {'Micro-F1 (Validation):':<35} {metriche_val['micro_f1']:.4f}")
    if metriche_test:
        print(f"  {'Micro-F1 (Test interno):':<35} {metriche_test['micro_f1']:.4f}")
    print(f"  {'Grafici eval:':<35} {eval_dir}")
    print(f"{'=' * 60}\n")

    return metriche_val, metriche_test


def predict_only(output_dir: str = "../output", risultati_dir: str = None, dataset_dir: str = None):
    """
    Carica il modello esistente salvato nel dizionario payload e genera
    esclusivamente il file di submission a partire dal test ufficiale preprocessato.
    """
    out_dir = os.path.abspath(output_dir)
    dataset_dir = dataset_dir or os.path.join(out_dir, 'dataset')
    risultati_dir = risultati_dir or os.path.join(out_dir, 'risultati')

    model_path = os.path.join(risultati_dir, "model_finale.pkl")
    path_test_uff = os.path.join(dataset_dir, "test_ufficiale_finale.csv")

    print(f"\n{'=' * 60}")
    print(f"  MODE PREDICT-ONLY: GENERAZIONE SUBMISSION AGGIORNATA")
    print(f"{'=' * 60}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modello non trovato in: {model_path}. Impossibile predire senza modello.")
    if not os.path.exists(path_test_uff):
        raise FileNotFoundError(f"Dataset test ufficiale non trovato in: {path_test_uff}. Esegui prima main.py.")

    print(f"  Caricamento del checkpoint: {model_path}...")
    checkpoint = joblib.load(model_path)

    # 1. Estrazione modello ed elenco feature storiche
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        estimator = checkpoint["model"]
        features_salvate = checkpoint["selected_features"]
        print(f"  -> Modello {type(estimator).__name__} caricato correttamente dal dizionario.")
    else:
        print("  [ATTENZIONE] Il file pkl contiene solo il modello puro.")
        estimator = checkpoint
        features_salvate = None

    # 2. Controllo difensivo fondamentale (Nativo di scikit-learn)
    # Se il modello ha salvato le feature al fit time, usiamo QUELLE. Hanno la priorità assoluta.
    if hasattr(estimator, "feature_names_in_"):
        features_modello = estimator.feature_names_in_.tolist()
        print(f"  -> Il Random Forest richiede esattamente queste {len(features_modello)} feature.")
        features_finali = features_modello
    elif features_salvate is not None:
        features_finali = features_salvate
    else:
        raise ValueError(
            "Impossibile determinare le feature originarie! Il file .pkl non contiene il dizionario "
            "e l'oggetto RandomForest non ha l'attributo 'feature_names_in_'. Devi rifare un giro di training completo (T)."
        )

    # 3. Caricamento dati e pulizia building_id
    df_test_uff = pd.read_csv(path_test_uff)
    building_ids = df_test_uff["building_id"].copy()
    df_sub_prep = df_test_uff.drop(columns=["building_id"], errors="ignore")

    # 4. Forziamo il riallineamento totale delle colonne (ordine e presenza)
    # Qualsiasi colonna in più (come age, area_percentage ecc.) viene buttata via.
    # Qualsiasi colonna mancante viene creata piena di 0.
    df_sub_prep = df_sub_prep.reindex(columns=features_finali, fill_value=0)

    # 5. Predizione e output
    print(f"  Predizione in corso su {len(df_sub_prep):,} edifici...")
    y_sub_pred = estimator.predict(df_sub_prep)

    submission = pd.DataFrame({
        "building_id": building_ids.values,
        "damage_grade": y_sub_pred,
    })
    sub_path = os.path.join(risultati_dir, "submission.csv")
    submission.to_csv(sub_path, index=False)

    print(f"  Submission salvata con successo in: {sub_path}")
    print(f"  Distribuzione delle predizioni effettuate:")
    for cls, cnt in submission["damage_grade"].value_counts().sort_index().items():
        print(f"    Classe {cls}: {cnt:>7,} ({cnt / len(submission) * 100:.1f}%)")
    print(f"{'=' * 60}\n")


# ===========================================================================
# ENTRY POINT CLI
# ===========================================================================

def _parse_args():
    """Argomenti da riga di comando (usato solo quando il file è eseguito direttamente)."""
    parser = argparse.ArgumentParser(
        description="Richter's Predictor — Addestramento modello finale",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model",            type=str, default="rf", choices=["rf", "knn", "ada"])
    parser.add_argument("--output-dir",       type=str, default="../output")
    parser.add_argument("--no-proba",         action="store_true")
    parser.add_argument("--n-estimators",     type=int, default=300)
    parser.add_argument("--max-depth",        type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--class-weight",     type=str, default=None)
    parser.add_argument("--n-neighbors",      type=int, default=7)
    parser.add_argument("--weights",          type=str, default="distance",
                        choices=["uniform", "distance"])
    parser.add_argument("--metric",           type=str, default="euclidean",
                        choices=["euclidean", "manhattan"])
    parser.add_argument("--learning-rate",    type=float, default=1.0)
    parser.add_argument("--base-estimator-max_depth", type=int, default=1)
    return parser.parse_args()


def main():
    """Entry point CLI: legge argparse e delega a run()."""
    args = _parse_args()
    try:
        run(
            model            = args.model,
            output_dir       = args.output_dir,
            no_proba         = args.no_proba,
            n_estimators     = args.n_estimators,
            max_depth        = args.max_depth,
            min_samples_leaf = args.min_samples_leaf,
            class_weight     = args.class_weight,
            n_neighbors      = args.n_neighbors,
            weights          = args.weights,
            metric           = args.metric,
            learning_rate    = args.learning_rate,
            base_estimator_max_depth = args.base_estimator_max_depth,
        )
    except Exception as ex:
        print(f"\n{'=' * 60}")
        print(f"  ERRORE DURANTE L'ESECUZIONE")
        print(f"{'=' * 60}")
        print(f"  {ex}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
