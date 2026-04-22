import os
from data_pipeline.preprocessing import Preprocessing
from data_pipeline.file_opener import scegli_opener
from data_reduction import DataReducer
from plot import Plotter


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
        dati_train = train_values.merge(train_labels, on='building_id')

        print(f"  {'File uniti su building_id.':<40}")
        print(f"  {'Righe totali:':<40} {dati_train.shape[0]:>8}")
        print(f"  {'Colonne totali:':<40} {dati_train.shape[1]:>8}")
        print(f"{'=' * 60}")

        explorer = Plotter(dati_train, output_dir=os.path.join(output_dir, "grafici"))
        explorer.esegui_tutto()

        # ── DATA REDUCTION ────────────────────────────────────────────
        # Applicata solo al train set per alleggerire l'onere computazionale.
        # Il campionamento è stratificato: mantiene le proporzioni di damage_grade.
        reducer    = DataReducer(dati_train)
        dati_train = reducer.interfaccia_utente()

        # ── PREPROCESSING TRAIN SET ───────────────────────────────────
        preprocessor         = Preprocessing(dati_train, is_train=True)
        df_train_processato  = preprocessor.esegui()

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

        # ── PREPROCESSING TEST SET ────────────────────────────────────
        path_test_values = os.path.join(data_dir, 'Test_Values.csv')

        if os.path.exists(path_test_values):
            print(f"\n{'=' * 60}")
            print(f"  CARICAMENTO DATI TEST")
            print(f"{'=' * 60}")
            print(f"  Caricamento Test_Values.csv...")
            test_values = scegli_opener(path_test_values).open(path_test_values)
            print(f"  {'Righe totali:':<40} {test_values.shape[0]:>8}")
            print(f"{'=' * 60}")

            # Vengono riusati scaler e imputer addestrati sul train (no data leakage)
            preprocessor_test = Preprocessing(
                test_values,
                scaler=scaler_addestrato,
                imputer_num=imputer_num_addestrato,
                imputer_cat=imputer_cat_addestrato,
                colonne_eliminate=colonne_eliminate,
                lista_colonne=lista_colonne_train,
                is_train=False
            )
            df_test_processato = preprocessor_test.esegui()

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

        else:
            print(f"\n  [AVVISO] Test_Values.csv non trovato in: {data_dir}")

    except Exception as ex:
        print(f"\n{'=' * 60}")
        print(f"  ERRORE DURANTE L'ESECUZIONE")
        print(f"{'=' * 60}")
        print(f"  {ex}")
        print(f"{'=' * 60}")
        raise