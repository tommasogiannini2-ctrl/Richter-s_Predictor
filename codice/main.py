import os
from data_pipeline.preprocessing import Preprocessing, dividi_train_validation_test
from data_pipeline.file_opener import scegli_opener
from data_reduction import DataReducer
from plot import Plotter
from clustering import Clustering
import pandas as pd


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir    = os.path.join(current_dir, '..', 'data')
    output_dir  = os.path.join(current_dir, '..', 'output')

    # Creazione della directory di output se non esiste
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # ── CARICAMENTO FILE DI TRAINING ─────────────────────────────
        print(f"\n{'=' * 60}")
        print(f"  CARICAMENTO DATI")
        print(f"{'=' * 60}")

        path_values = os.path.join(data_dir, 'Train_Values.csv')
        path_labels = os.path.join(data_dir, 'Train_Labels.csv')

        print(f"  Caricamento Train_Values.csv...")
        train_values = scegli_opener(path_values).open(path_values)

        print(f"  Caricamento Train_Labels.csv...")
        train_labels = scegli_opener(path_labels).open(path_labels)

        # Unione di feature e target su building_id
        dati_tot = train_values.merge(train_labels, on='building_id')
        dati_tot = dati_tot.drop('building_id', axis=1)
        dati_tot.info()


        print(f"  {'File uniti su building_id.':<40}")
        print(f"  {'Righe totali:':<40} {dati_tot.shape[0]:>8}")
        print(f"  {'Colonne totali:':<40} {dati_tot.shape[1]:>8}")
        print(f"{'=' * 60}")

        explorer = Plotter(dati_tot, output_dir=os.path.join(output_dir, "grafici"))
        explorer.esegui_tutto()

        # ── DATA REDUCTION ────────────────────────────────────────────
        # Applicata solo al train set per alleggerire l'onere computazionale.
        # Il campionamento è stratificato: mantiene le proporzioni di damage_grade.
        reducer    = DataReducer(dati_tot)
        dati_tot = reducer.interfaccia_utente()

        # ── Separazione TRAIN, VALIDATION e TEST ────────────────────────────────────────────
        dati_train, dati_vali, dati_test = dividi_train_validation_test(dati_tot)
        print(f"DATAFRAME DI TRAIN \n {dati_train.info()}")
        print(f"DATAFRAME DI VALIDATION \n {dati_vali.info()}")
        print(f"DATAFRAME DI TEST \n {dati_test.info()}")


        # ── PREPROCESSING TRAIN SET ───────────────────────────────────
        preprocessor         = Preprocessing(dati_train, is_train=True)
        df_train_processato  = preprocessor.esegui()

        #  ── CLUSTERING TRAIN SET ───────────────────────────────────

        X_train = df_train_processato.drop(columns=['damage_grade'])

        # Trovare il numero di cluster ottimale
        engine = Clustering()
        engine.plot_elbow_method(X_train)

        # Scegli K e fai il fit sul train
        k_scelto = 5
        train_clusters = engine.fit(X_train, k=k_scelto)
        df_train_processato= pd.concat([df_train_processato, train_clusters], axis=1)
        print(df_train_processato.info())


        # Estrazione degli oggetti addestrati da riapplicare sul test set
        scaler_addestrato    = preprocessor.scaler
        imputer_num_addestrato = preprocessor.imputer_num
        imputer_cat_addestrato = preprocessor.imputer_cat
        colonne_eliminate    = preprocessor.colonne_eliminate
        lista_colonne_train  = preprocessor.lista_colonne

        # Salvataggio del training set processato
        output_train_path = os.path.join(output_dir, 'train_processato.csv')
        df_train_processato.to_csv(output_train_path, index=False)

        print(f"\n{'=' * 60}")
        print(f"  RIEPILOGO TRAIN")
        print(f"{'=' * 60}")
        print(f"  {'Righe:':<40} {df_train_processato.shape[0]:>8}")
        print(f"  {'Colonne:':<40} {df_train_processato.shape[1]:>8}")
        print(f"  {'Valori mancanti residui:':<40} {df_train_processato.isnull().sum().sum():>8}")
        print(f"  {'File salvato in:':<40} {output_train_path}")
        print(f"{'=' * 60}")

        # Definiamo una struttura dati per iterare sui set di validation e test
        datasets_to_process = {
            "VALIDATION": {
                "data": dati_vali,
                "filename": 'val_processato.csv'
            },
            "TEST": {
                "data": dati_test,
                "filename": 'test_processato.csv'
            }
        }

        for label, info in datasets_to_process.items():
            # ── PREPROCESSING ────────────────────────────────────
            # Vengono riusati scaler e imputer addestrati sul train
            preprocessor = Preprocessing(
                info["data"],
                scaler=scaler_addestrato,
                imputer_num=imputer_num_addestrato,
                imputer_cat=imputer_cat_addestrato,
                lista_colonne=lista_colonne_train,
                is_train=False
            )
            df_processato = preprocessor.esegui()

            # ── CLUSTERING ───────────────────────────────────────
            # Rimuoviamo il target se presente (nel test set potrebbe non esserci)
            X = df_processato.drop(
                columns=['damage_grade']) if 'damage_grade' in df_processato.columns else df_processato
            clusters = engine.predict(X)
            df_processato = pd.concat([df_processato, clusters], axis=1)
            print(df_processato.info())

            # ── SALVATAGGIO ──────────────────────────────────────
            output_path = os.path.join(output_dir, info["filename"])
            df_processato.to_csv(output_path, index=False)

            # ── RIEPILOGO COERENTE ───────────────────────────────
            print(f"\n{'=' * 60}")
            print(f"  RIEPILOGO {label}")
            print(f"{'=' * 60}")
            print(f"  {'Righe:':<40} {df_processato.shape[0]:>8}")
            print(f"  {'Colonne:':<40} {df_processato.shape[1]:>8}")
            print(f"  {'File salvato in:':<40} {output_path}")
            print(f"{'=' * 60}")

            if label == "VALIDATION":
                df_val_processato = df_processato
            elif label == "TEST":
                df_test_processato = df_processato
            else:
                print("ERRORE: LABEL NON VALIDA")

        # ── FEATURE SELECTION ──────────────────────────────────────────
        # In questa fase testiamo i 4 selettori implementati (più la baseline)
        # per trovare il subset di feature che massimizza la micro-F1 sul validation set.
        
        from validation import FeatureSelectionSearch

        print(f"\n{'=' * 60}")
        print(f"  FASE 4: RICERCA STRATEGIA DI FEATURE SELECTION")
        print(f"{'=' * 60}")

        # Separazione feature e target
        X_train = df_train_processato.drop(columns=['damage_grade'])
        y_train = df_train_processato['damage_grade']
        X_val   = df_val_processato.drop(columns=['damage_grade'])
        y_val   = df_val_processato['damage_grade']

        # Configurazione della ricerca: testiamo tutte le famiglie studiate
        searcher = FeatureSelectionSearch()
        config_selettori = {
            "all":          {},                           # Baseline
            "mutual_info":  {"k": 25},                    # Filter pair-wise
            "relief":       {"k": 25, "n_samples": 500},  # Filter simultaneo
            "sfs":          {"k": 15, "cv": 3},           # Subset selection
            "embedded_dt":  {"soglia": "mean"}            # Embedded
        }

        # Esecuzione della ricerca
        risultati_fs = searcher.search(X_train, y_train, X_val, y_val, config_selettori)
        
        # Salvataggio dei risultati della ricerca per il report
        output_fs_path = os.path.join(output_dir, 'risultati_feature_selection.csv')
        risultati_fs.to_csv(output_fs_path, index=False)
        print(f"  Risultati della ricerca salvati in: {output_fs_path}")

        # Recupero del miglior selettore e applicazione ai dataset
        best_selector = searcher.get_best_selector()
        
        print(f"\n  Applicazione del miglior selettore ({searcher.best_selector_name_})...")
        
        X_train_sel = best_selector.transform(X_train)
        X_val_sel   = best_selector.transform(X_val)
        
        # Per il test set, gestiamo l'eventuale assenza del target
        X_test = df_test_processato.drop(columns=['damage_grade']) if 'damage_grade' in df_test_processato.columns else df_test_processato
        X_test_sel = best_selector.transform(X_test)

        # Ricomposizione dei DataFrame con target (se presente)
        df_train_final = pd.concat([X_train_sel, y_train.reset_index(drop=True)], axis=1)
        df_val_final   = pd.concat([X_val_sel, y_val.reset_index(drop=True)], axis=1)
        
        if 'damage_grade' in df_test_processato.columns:
            y_test = df_test_processato['damage_grade']
            df_test_final = pd.concat([X_test_sel, y_test.reset_index(drop=True)], axis=1)
        else:
            df_test_final = X_test_sel

        # Salvataggio dei dataset finali pronti per l'addestramento
        df_train_final.to_csv(os.path.join(output_dir, 'train_finale.csv'), index=False)
        df_val_final.to_csv(os.path.join(output_dir, 'val_finale.csv'), index=False)
        df_test_final.to_csv(os.path.join(output_dir, 'test_finale.csv'), index=False)

        print(f"  Dataset finali salvati in: {output_dir}")
        print(f"  Numero feature selezionate: {X_train_sel.shape[1]}")
        print(f"{'=' * 60}\n")

# -------------- FINE PIPELINE --------------------------------------------


    except Exception as ex:
        print(f"\n{'=' * 60}")
        print(f"  ERRORE DURANTE L'ESECUZIONE")
        print(f"{'=' * 60}")
        print(f"  {ex}")
        print(f"{'=' * 60}")
        raise