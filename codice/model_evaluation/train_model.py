"""
train_model.py
==============
Addestramento del modello finale e generazione predizioni per Richter's Predictor.

Questo modulo espone due interfacce:

  1. run(...)  — funzione chiamabile programmaticamente da main.py
                 (o da qualsiasi altro script) senza passare per argparse.

  2. main()   — entry point CLI che legge gli argomenti da riga di comando
                 e delega tutto a run().

Flusso:
  1. Carica i dataset finali prodotti da main.py (train_finale.csv, val_finale.csv,
     test_finale.csv, test_ufficiale_processato.csv).
  2. Addestra il modello scelto (RandomForest o KNN) sull'intero train_finale.
  3. Valuta il modello su val_finale e test_finale usando ModelEvaluator.
  4. Genera le predizioni finali per la submission DrivenData.
  5. Salva metriche, grafici e submission.csv.

Utilizzo da riga di comando (da codice/):
  python model_evaluation/train_model.py [--model MODEL] [--output-dir DIR] [--no-proba]

Opzioni:
  --model MODEL     Modello: 'rf' (Random Forest, default) o 'knn'.
  --output-dir DIR  Directory con i file prodotti da main.py (default: ../output).
  --no-proba        Disabilita le probabilità (curve ROC). Utile con KNN.

Output in output/eval/:
  confusion_matrix.png, confusion_matrix_norm.png, roc_curves.png,
  class_report.png, distribuzione_classi.png, metriche.csv

Output in output/:
  model_finale.pkl, submission.csv
"""

import argparse
import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from model_evaluation.evaluation import ModelEvaluator


# ===========================================================================
# COSTRUZIONE DEL MODELLO
# ===========================================================================

def _build_model(
    model: str,
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

    Può essere chiamata direttamente da main.py:
        from model_evaluation.train_model import run as avvia_training
        avvia_training(model='rf', output_dir=output_dir)

    Parameters
    ----------
    model : str
        'rf', 'knn' o 'ada'.
    output_dir : str
        Directory contenente i file prodotti da main.py.
    no_proba : bool
        Se True, salta il calcolo delle probabilità (curve ROC).
    Gli altri parametri sono iperparametri del modello scelto.
    """
    out_dir  = os.path.abspath(output_dir)
    eval_dir = os.path.join(out_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # ======================================================================
    # FASE 1 — CARICAMENTO DATASET FINALI
    # ======================================================================
    print(f"\n{'=' * 60}")
    print(f"  TRAINING — FASE 1: CARICAMENTO DATASET FINALI")
    print(f"{'=' * 60}")

    path_train = os.path.join(out_dir, "train_finale.csv")
    path_val   = os.path.join(out_dir, "val_finale.csv")
    path_test  = os.path.join(out_dir, "test_finale.csv")

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

    model_path = os.path.join(out_dir, "model_finale.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(estimator, f)
    print(f"  Modello salvato: {model_path}")

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

    # Il test ufficiale è già completamente preprocessato da main.py
    # (scaling, imputation, clustering, feature selection) → caricato direttamente.
    path_test_uff = os.path.join(out_dir, "test_ufficiale_processato.csv")

    if not os.path.exists(path_test_uff):
        print(f"  [Avviso] {path_test_uff} non trovato: submission saltata.")
    else:
        df_test_uff  = pd.read_csv(path_test_uff)
        building_ids = df_test_uff["building_id"].copy()

        df_sub_prep = df_test_uff.drop(columns=["building_id"])
        # allineamento difensivo alle colonne del train
        df_sub_prep = df_sub_prep.reindex(columns=X_train.columns, fill_value=0)

        print(f"  Predizione su {len(df_sub_prep):,} edifici...")
        y_sub_pred = estimator.predict(df_sub_prep)

        submission = pd.DataFrame({
            "building_id":  building_ids.values,
            "damage_grade": y_sub_pred,
        })
        sub_path = os.path.join(out_dir, "submission.csv")
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
    parser.add_argument("--base-estimator-max-depth", type=int, default=1)
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
