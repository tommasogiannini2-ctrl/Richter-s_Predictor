import os
from data_pipeline.preprocessing import Preprocessing
from data_pipeline.file_opener import scegli_opener
from data_reduction import DataReducer

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    output_dir = os.path.join(current_dir, '..', 'output')

    # Creazione directory di output se non esiste
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        path_values = os.path.join(data_dir, 'Train_Values.csv')
        path_labels = os.path.join(data_dir, 'Train_Labels.csv')

        print("Caricamento file di training in corso...")
        train_values = scegli_opener(path_values).open(path_values)
        train_labels = scegli_opener(path_labels).open(path_labels)

        # MERGE
        dati_train = train_values.merge(train_labels, on='building_id')
        print(f"File caricati e uniti. Righe totali: {dati_train.shape[0]}")

        # --- DATA REDUCTION ---
        # data reduction utilizzato solo per il train set per alleggerire l'onere computazionale
        reducer = DataReducer(dati_train)
        dati_train = reducer.interfaccia_utente()

        # PREPROCESSING TRAIN SET
        preprocessor = Preprocessing(dati_train, is_train=True)
        df_train_processato = preprocessor.esegui()

        scaler_addestrato = preprocessor.scaler
        imputer_num_addestrato = preprocessor.imputer_num
        imputer_cat_addestrato = preprocessor.imputer_cat
        colonne_eliminate = preprocessor.colonne_eliminate
        lista_colonne_train = preprocessor.lista_colonne

        print("\n--- RESOCONTO FINALE TRAINING ---")
        print(f"Dimensioni Righe: {df_train_processato.shape[0]}")
        print(f"Dimensioni Colonne:  {df_train_processato.shape[1]} \n")
        print("Informazioni sul dataframe:")
        df_train_processato.info()
        print(f"\nValori mancanti residui: {df_train_processato.isnull().sum().sum()}")
 
        # SALVATAGGIO FILE TRAINING PROCESSATO
        output_train_path = os.path.join(output_dir, 'train_processato.csv')
        df_train_processato.to_csv(output_train_path, index=False)
        print(f"\nFile di training processato salvato in: {output_train_path}")

        # PREPROCESSING TEST SET
        path_test_values = os.path.join(data_dir, 'Test_Values.csv')
        if os.path.exists(path_test_values):
            print("\nCaricamento file di test in corso...")
            test_values = scegli_opener(path_test_values).open(path_test_values)
 
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
 
            print("\n--- RESOCONTO FINALE TEST ---")
            print(f"Dimensioni Righe: {df_test_processato.shape[0]}")
            print(f"Dimensioni Colonne: {df_test_processato.shape[1]} \n")
 
            output_test_path = os.path.join(output_dir, 'test_processato.csv')
            df_test_processato.to_csv(output_test_path, index=False)
            print(f"File di test processato salvato in: {output_test_path}")
 
    except Exception as ex:
        print(f"Errore durante l'esecuzione: {ex}")
        raise
