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

        engine = Clustering()

        # Trova il K
        engine.plot_elbow_method(X_train)

        # Scegli K e fai il fit sul train
        k_scelto = 4
        train_clusters = engine.fit(X_train, k=k_scelto)

        # Riattacchiamo i cluster ai dataframe originali che hanno ancora la label
        df_train_processato= pd.concat([df_train_processato, train_clusters], axis=1)

        # Ora hai tutto insieme: feature, label originale e cluster!
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

        # ── PREPROCESSING VALIDATION SET ────────────────────────────────────

        # Vengono riusati scaler e imputer addestrati sul train (no data leakage)
        preprocessor_test = Preprocessing(
            dati_vali,
            scaler=scaler_addestrato,
            imputer_num=imputer_num_addestrato,
            imputer_cat=imputer_cat_addestrato,
            colonne_eliminate=colonne_eliminate,
            lista_colonne=lista_colonne_train,
            is_train=False
        )
        df_val_processato = preprocessor_test.esegui()

        #  ── CLUSTERING TEST SET ───────────────────────────────────

        X_val = df_val_processato.drop(columns=['damage_grade'])

        # Assegna i dati di Test ai cluster generati dal Train
        val_clusters = engine.predict(X_val)

        # Riattacchiamo i cluster ai dataframe originali che hanno ancora la label
        df_val_processato= pd.concat([df_val_processato, val_clusters], axis=1)

        # Ora hai tutto insieme: feature, label originale e cluster!
        print(df_val_processato.info())

        # Salvataggio del test set processato
        output_test_path = os.path.join(output_dir, 'val_processato.csv')
        df_val_processato.to_csv(output_test_path, index=False)

        print(f"\n{'=' * 60}")
        print(f"  RIEPILOGO VALIDATION")
        print(f"{'=' * 60}")
        print(f"  {'Righe:':<40} {df_val_processato.shape[0]:>8}")
        print(f"  {'Colonne:':<40} {df_val_processato.shape[1]:>8}")
        print(f"  {'File salvato in:':<40} {output_test_path}")
        print(f"{'=' * 60}")

        # ── PREPROCESSING TEST SET ────────────────────────────────────

        # Vengono riusati scaler e imputer addestrati sul train (no data leakage)
        preprocessor_test = Preprocessing(
            dati_test,
            scaler=scaler_addestrato,
            imputer_num=imputer_num_addestrato,
            imputer_cat=imputer_cat_addestrato,
            colonne_eliminate=colonne_eliminate,
            lista_colonne=lista_colonne_train,
            is_train=False
        )
        df_test_processato = preprocessor_test.esegui()

        #  ── CLUSTERING TEST SET ───────────────────────────────────

        X_test = df_test_processato.drop(columns=['damage_grade'])

        # Assegna i dati di Test ai cluster generati dal Train
        test_clusters = engine.predict(X_test)

        # Riattacchiamo i cluster ai dataframe originali che hanno ancora la label
        df_test_processato= pd.concat([df_test_processato, test_clusters], axis=1)

        # Ora hai tutto insieme: feature, label originale e cluster!
        print(df_test_processato.info())

        # Salvataggio del test set processato
        output_test_path = os.path.join(output_dir, 'test_processato.csv')
        df_test_processato.to_csv(output_test_path, index=False)

        print(f"\n{'=' * 60}")
        print(f"  RIEPILOGO TEST")
        print(f"{'=' * 60}")
        print(f"  {'Righe:':<40} {df_test_processato.shape[0]:>8}")
        print(f"  {'Colonne:':<40} {df_test_processato.shape[1]:>8}")
        print(f"  {'File salvato in:':<40} {output_test_path}")
        print(f"{'=' * 60}")


    except Exception as ex:
        print(f"\n{'=' * 60}")
        print(f"  ERRORE DURANTE L'ESECUZIONE")
        print(f"{'=' * 60}")
        print(f"  {ex}")
        print(f"{'=' * 60}")
        raise