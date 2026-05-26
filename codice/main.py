import os
import pandas as pd
import gc
import joblib

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from codice.model_evaluation.train_model import predict_only
from data_pipeline.preprocessing import Preprocessing, dividi_train_validation_test
from data_pipeline.file_opener import scegli_opener
from data_reduction import DataReducer
from plot import Plotter
from data_pipeline.clustering import Clustering
from model_evaluation.validation import FeatureSelectionSearch
from model_evaluation.train_model import run as avvia_training

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    output_dir = os.path.join(current_dir, '..', 'output')

    grafici_dir = os.path.join(output_dir, 'grafici')
    dataset_dir = os.path.join(output_dir, 'dataset')
    risultati_dir = os.path.join(output_dir, 'risultati')

    # Creazione della directory di output se non esiste
    for d in [output_dir, grafici_dir, dataset_dir, risultati_dir]:
        os.makedirs(d, exist_ok=True)

    path_modello_salvato = os.path.join(risultati_dir, 'model_finale.pkl')
    eseguire_training_completo = True

    # Controllo iniziale sulla presenza del file unico del modello
    if os.path.exists(path_modello_salvato):
        print(f"\n{'=' * 60}")
        print(f"  MODELLO FINALE INTEGRATO TROVATO")
        print(f"{'=' * 60}")
        scelta = input(
            "  È stato trovato un modello pre-addestrato (include Feature Selection).\n"
            "  Vuoi usarlo per elaborare i dati (S) o rifare la ricerca completa (T)? [S/T]: "
        ).strip().upper()

        if scelta == 'S':
            eseguire_training_completo = False
            print("\n  -> Caricamento del modello e del selettore integrato...")
            # Carichiamo l'oggetto unico (che contiene internamente modello + selettore)
            modello_caricato = joblib.load(path_modello_salvato)
            print("  -> Caricato con successo! Verrà eseguito solo il preprocessing dei dati.")

    try:
        # ====================================================================
        # FASE 1 — CARICAMENTO FILE DI TRAINING (Necessario anche per ricostruire il fit del preprocessing)
        # ====================================================================
        print(f"\n{'=' * 60}")
        print(f"  FASE 1: CARICAMENTO DATI")
        print(f"{'=' * 60}")

        path_values = os.path.join(data_dir, 'Train_Values.csv')
        path_labels = os.path.join(data_dir, 'Train_Labels.csv')

        train_values = scegli_opener(path_values).open(path_values)
        train_labels = scegli_opener(path_labels).open(path_labels)

        dati_tot = train_values.merge(train_labels, on='building_id')
        dati_tot = dati_tot.drop('building_id', axis=1)

        del train_values, train_labels
        gc.collect()

        # ====================================================================
        # FASE 2 — ANALISI ESPLORATIVA (EDA)
        # ====================================================================
        if eseguire_training_completo:
            explorer = Plotter(dati_tot, output_dir=grafici_dir)
            explorer.esegui_tutto()

        # ====================================================================
        # FASE 3 — DATA REDUCTION (Attiva solo in fase di ricerca/training da zero)
        # ====================================================================
        if eseguire_training_completo:
            reducer = DataReducer(dati_tot)
            dati_tot = reducer.interfaccia_utente()

        # ====================================================================
        # FASE 4 — SPLIT INTERNO: TRAIN / VALIDATION / TEST
        # ====================================================================
        dati_train, dati_vali, dati_test = dividi_train_validation_test(dati_tot)
        del dati_tot
        gc.collect()

        # ====================================================================
        # FASE 5 — PREPROCESSING TRAIN SET
        # ====================================================================
        preprocessor = Preprocessing(dati_train, is_train=True)
        df_train_processato = preprocessor.esegui()

        scaler_addestrato = preprocessor.scaler
        imputer_num = preprocessor.imputer_num
        imputer_bin = preprocessor.imputer_bin
        imputer_cat = preprocessor.imputer_cat
        colonne_eliminate = preprocessor.colonne_eliminate
        lista_colonne_train = preprocessor.lista_colonne

        # ====================================================================
        # FASE 6 — CLUSTERING SUL TRAIN SET
        # ====================================================================
        X_train_clust = df_train_processato.drop(columns=['damage_grade'])
        engine = Clustering()

        if eseguire_training_completo:
            engine.plot_elbow_method(X_train_clust, output_dir=grafici_dir)

        k_scelto = 5
        train_clusters = engine.fit(X_train_clust, k=k_scelto)
        df_train_processato = pd.concat([df_train_processato, train_clusters], axis=1)

        output_train_path = os.path.join(dataset_dir, 'train_processato.csv')
        df_train_processato.to_csv(output_train_path, index=False)

        del dati_train, X_train_clust, train_clusters
        gc.collect()

        # ====================================================================
        # FASE 7 — PREPROCESSING VALIDATION E TEST INTERNO
        # ====================================================================
        dataset_da_processare = {
            "VALIDATION": {"dati": dati_vali, "filename": 'val_processato.csv'},
            "TEST": {"dati": dati_test, "filename": 'test_processato.csv'}
        }
        df_processati = {}

        for label, info in dataset_da_processare.items():
            pp = Preprocessing(
                info["dati"], scaler=scaler_addestrato, imputer_num=imputer_num,
                imputer_bin=imputer_bin, imputer_cat=imputer_cat,
                colonne_eliminate=colonne_eliminate, lista_colonne=lista_colonne_train, is_train=False
            )
            df_proc = pp.esegui()
            X_proc = df_proc.drop(columns=['damage_grade'])
            clusters = engine.predict(X_proc)
            df_proc = pd.concat([df_proc, clusters], axis=1)
            df_proc.to_csv(os.path.join(dataset_dir, info["filename"]), index=False)
            df_processati[label] = df_proc

        df_val_processato = df_processati["VALIDATION"]
        df_test_processato = df_processati["TEST"]

        del dati_vali, dati_test, df_processati
        gc.collect()

        # ====================================================================
        # FASE 8 — FEATURE SELECTION SEARCH / APPLICAZIONE SELETTORE
        # ====================================================================
        print(f"\n{'=' * 60}")
        print(f"  FASE 8: GESTIONE FEATURE SELECTION")
        print(f"{'=' * 60}")

        X_train_fs = df_train_processato.drop(columns=['damage_grade'])
        y_train_fs = df_train_processato['damage_grade']
        X_val_fs = df_val_processato.drop(columns=['damage_grade'])
        X_test_fs = df_test_processato.drop(columns=['damage_grade'], errors='ignore')

        if eseguire_training_completo:
            # ----------------------------------------------------------------
            # CASO A: Training completo (Lancia la ricerca RandomizedSearchCV)
            # ----------------------------------------------------------------
            print("  [TRAIN=TRUE] Avvio ricerca spaziale FeatureSelectionSearch...")
            search = FeatureSelectionSearch(n_iter=10, cv=3, include_sfs=True, verbose=1, output_dir=output_dir)
            search.fit(X_train_fs, y_train_fs)

            output_fs_path = os.path.join(risultati_dir, 'feature_selection_results.csv')
            search.get_results().to_csv(output_fs_path, index=False)

            # Trasformazione tramite l'oggetto search appena addestrato
            X_train_sel = search.transform(X_train_fs)
            X_val_sel = search.transform(X_val_fs)
            X_test_sel = search.transform(X_test_fs)

            best_model_obj = search.get_best_params()['model']
        else:
            # ----------------------------------------------------------------
            # CASO B: Recupero modello (Legge il dizionario integrato)
            # ----------------------------------------------------------------
            print("  [TRAIN=FALSE] Salto la ricerca. Caricamento di model_finale.pkl...")

            # Carichiamo il dizionario completo salvato da train_model.py
            checkpoint = joblib.load(path_modello_salvato)
            if isinstance(checkpoint, dict) and "selected_features" in checkpoint:
                best_model_obj = checkpoint["model"]
                features_salvate = checkpoint["selected_features"]  # <-- Questa serve alla Fase 9
                X_train_sel = X_train_fs[features_salvate]

            # Se l'utente ha un vecchio file .pkl che contiene solo il modello puro,
            # gestiamo un fallback indolore per evitare crash
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                best_model_obj = checkpoint["model"]
                features_salvate = checkpoint["selected_features"]
                print(f"  -> Pipeline caricata. Feature selezionate storiche trovate: {len(features_salvate)}")

                # Applichiamo la feature selection filtrando direttamente le colonne salvate
                X_train_sel = X_train_fs[features_salvate]
                X_val_sel = X_val_fs[features_salvate]
                X_test_sel = X_test_fs[features_salvate]
            else:
                # Vecchio comportamento di emergenza se il pkl non ha il dizionario
                print("  [ATTENZIONE] Il pkl caricato è un modello vecchio senza dizionario di feature.")
                best_model_obj = checkpoint
                X_train_sel = X_train_fs.copy()
                X_val_sel = X_val_fs.copy()
                X_test_sel = X_test_fs.copy()


        # Ricomposizione dei DataFrame finali
        df_train_finale = pd.concat([X_train_sel, y_train_fs.reset_index(drop=True)], axis=1)
        df_val_finale = pd.concat([X_val_sel, df_val_processato['damage_grade'].reset_index(drop=True)], axis=1)
        df_test_finale = pd.concat([X_test_sel, df_test_processato['damage_grade'].reset_index(drop=True)], axis=1)

        df_train_finale.to_csv(os.path.join(dataset_dir, 'train_finale.csv'), index=False)
        df_val_finale.to_csv(os.path.join(dataset_dir, 'val_finale.csv'), index=False)
        df_test_finale.to_csv(os.path.join(dataset_dir, 'test_finale.csv'), index=False)

        del df_train_processato, df_val_processato, df_test_processato
        gc.collect()

        # ====================================================================
        # FASE 9 — PREPROCESSING E CLUSTERING DEL TEST UFFICIALE DRIVENDATA
        # ====================================================================
        print(f"\n{'=' * 60}")
        print(f"  FASE 9: PREPROCESSING TEST UFFICIALE DRIVENDATA")
        print(f"{'=' * 60}")

        path_test_ufficiale = os.path.join(data_dir, 'Test_Values.csv')
        df_test_ufficiale = scegli_opener(path_test_ufficiale).open(path_test_ufficiale)
        building_ids_submission = df_test_ufficiale['building_id'].copy()
        df_test_ufficiale = df_test_ufficiale.drop(columns=['building_id'])

        pp_test_uff = Preprocessing(
            df_test_ufficiale, scaler=scaler_addestrato, imputer_num=imputer_num,
            imputer_bin=imputer_bin, imputer_cat=imputer_cat,
            colonne_eliminate=colonne_eliminate, lista_colonne=lista_colonne_train, is_train=False
        )
        df_test_uff_proc = pp_test_uff.esegui()

        clusters_test_uff = engine.predict(df_test_uff_proc)
        df_test_uff_proc = pd.concat([df_test_uff_proc, clusters_test_uff], axis=1)

        # ─── MODIFICA QUI: APPLICAZIONE DEL SELETTORE O DIZIONARIO ───────────
        if eseguire_training_completo:
            df_test_uff_sel = search.transform(df_test_uff_proc)
        else:
            # Se non facciamo il training, usiamo la lista di feature salvata nel dizionario checkpoint
            # caricato nella Fase 8 (assumendo che nella Fase 8 tu abbia estratto 'features_salvate')
            if 'features_salvate' in locals() or 'features_salvate' in globals():
                df_test_uff_sel = df_test_uff_proc[features_salvate]
            else:
                # Fallback di emergenza se per qualche motivo la variabile non esiste
                df_test_uff_sel = df_test_uff_proc[X_train_sel.columns]
        # ─────────────────────────────────────────────────────────────────────

        output_test_uff_path = os.path.join(dataset_dir, 'test_ufficiale_finale.csv')
        df_test_uff_proc_con_id = df_test_uff_sel.copy()
        df_test_uff_proc_con_id.insert(0, 'building_id', building_ids_submission.values)
        df_test_uff_proc_con_id.to_csv(output_test_uff_path, index=False)

        # ====================================================================
        # FASE 10 — CONFIGURAZIONE FINALE ED ESECUZIONE
        # ====================================================================
        if eseguire_training_completo:
            nome_modello = 'rf' if isinstance(best_model_obj, RandomForestClassifier) else (
                'ada' if isinstance(best_model_obj, AdaBoostClassifier) else 'knn')

            print(f"\n  Modello migliore dalla ricerca: {type(best_model_obj).__name__} -> avvio training finale...")
            params_modello = {k.replace('model__', ''): v for k, v in search.get_best_params().items() if
                              k.startswith('model__')}

            if nome_modello == 'ada' and 'estimator' in params_modello:
                base_est = params_modello.pop('estimator')
                params_modello['base_estimator_max_depth'] = base_est.max_depth

            # Caso A: Allena il nuovo modello e lancia il report completo delle metriche
            avvia_training(model=nome_modello, output_dir=output_dir, dataset_dir=dataset_dir,
                           risultati_dir=risultati_dir, **params_modello)
        else:
            # Caso B: Salta il re-addestramento e chiama la nuova funzione di sola predizione
            print(f"\n  [TRAIN=FALSE] Preprocessing completato. Avvio generazione della submission...")

            # Chiamata alla funzione di sola predizione aggiunta in train_model.py
            predict_only(output_dir=output_dir, dataset_dir=dataset_dir, risultati_dir=risultati_dir)

    except Exception as ex:
        print(f"\n  ERRORE DURANTE L'ESECUZIONE: {ex}")
        raise