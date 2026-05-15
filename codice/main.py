import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from data_pipeline.preprocessing import Preprocessing, dividi_train_validation_test
from data_pipeline.file_opener import scegli_opener
from data_reduction import DataReducer
from plot import Plotter
from data_pipeline.clustering import Clustering
from model_evaluation.validation import FeatureSelectionSearch
from model_evaluation.train_model import run as avvia_training


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir    = os.path.join(current_dir, '..', 'data')
    output_dir  = os.path.join(current_dir, '..', 'output')
 
    # Creazione della directory di output se non esiste
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
 
    try:
        # ====================================================================
        # FASE 1 — CARICAMENTO FILE DI TRAINING
        # ====================================================================
        print(f"\n{'=' * 60}")
        print(f"  FASE 1: CARICAMENTO DATI")
        print(f"{'=' * 60}")
 
        path_values = os.path.join(data_dir, 'Train_Values.csv')
        path_labels = os.path.join(data_dir, 'Train_Labels.csv')
 
        print(f"  Caricamento Train_Values.csv...")
        train_values = scegli_opener(path_values).open(path_values)
 
        print(f"  Caricamento Train_Labels.csv...")
        train_labels = scegli_opener(path_labels).open(path_labels)
 
        # Unione di feature e target su building_id.
        # building_id è un identificatore: non porta informazione predittiva,
        # quindi viene rimosso prima di qualsiasi elaborazione.
        dati_tot = train_values.merge(train_labels, on='building_id')
        dati_tot = dati_tot.drop('building_id', axis=1)
 
        print(f"  {'File uniti su building_id.':<40}")
        print(f"  {'Righe totali:':<40} {dati_tot.shape[0]:>8,}")
        print(f"  {'Colonne totali:':<40} {dati_tot.shape[1]:>8}")
        print(f"{'=' * 60}")
 
        # ====================================================================
        # FASE 2 — ANALISI ESPLORATIVA (EDA)
        # ====================================================================
        # Il Plotter riceve il DataFrame GREZZO (prima dell'encoding) perché
        # plot_categoriche_vs_target() richiede le colonne stringa originali.
        grafici_dir = os.path.join(output_dir, 'grafici')
        explorer = Plotter(dati_tot, output_dir=grafici_dir)
        explorer.esegui_tutto()
 
        # ====================================================================
        # FASE 3 — DATA REDUCTION (opzionale, interattiva)
        # ====================================================================
        # Applicata solo al training set per alleggerire l'onere computazionale.
        # Il campionamento è stratificato: mantiene le proporzioni di damage_grade.
        reducer  = DataReducer(dati_tot)
        dati_tot = reducer.interfaccia_utente()
 
        # ====================================================================
        # FASE 4 — SPLIT INTERNO: TRAIN / VALIDATION / TEST (70/15/15)
        # ====================================================================
        # Lo split è stratificato su damage_grade per preservare le proporzioni
        # delle classi in tutti e tre i subset.
        dati_train, dati_vali, dati_test = dividi_train_validation_test(dati_tot)
 
        print(f"\n{'=' * 60}")
        print(f"  RIEPILOGO SPLIT")
        print(f"{'=' * 60}")
        print(f"  {'Train:':<40} {len(dati_train):>8,} righe ({len(dati_train)/len(dati_tot):.0%})")
        print(f"  {'Validation:':<40} {len(dati_vali):>8,} righe ({len(dati_vali)/len(dati_tot):.0%})")
        print(f"  {'Test interno:':<40} {len(dati_test):>8,} righe ({len(dati_test)/len(dati_tot):.0%})")
        print(f"{'=' * 60}")
 
        # ====================================================================
        # FASE 5 — PREPROCESSING TRAIN SET
        # ====================================================================
        # is_train=True: esegue fit + transform su tutti i trasformatori
        # (scaler, imputer_*, encoder). Gli artefatti vengono poi salvati
        # per essere riapplicati su validation, test e test ufficiale.
        preprocessor = Preprocessing(dati_train, is_train=True)
        df_train_processato = preprocessor.esegui()
 
        # Salvataggio degli artefatti di preprocessing del train.
        # Vengono passati a tutti i Preprocessing successivi (is_train=False)
        # per garantire che validation e test siano trasformati con gli stessi
        # parametri (nessun data leakage).
        scaler_addestrato   = preprocessor.scaler
        imputer_num         = preprocessor.imputer_num
        imputer_bin         = preprocessor.imputer_bin
        imputer_cat         = preprocessor.imputer_cat
        colonne_eliminate   = preprocessor.colonne_eliminate
        lista_colonne_train = preprocessor.lista_colonne
 
        # ====================================================================
        # FASE 6 — CLUSTERING SUL TRAIN SET
        # ====================================================================
        # Il clustering aggiunge feature sintetiche (cluster one-hot) che
        # catturano strutture di vicinanza non lineari nel dato.
        # Il K viene scelto visivamente dal metodo del gomito (elbow).
        X_train_clust = df_train_processato.drop(columns=['damage_grade'])
 
        # Metodo del gomito: aiuta a scegliere visivamente il K ottimale.
        engine = Clustering()
        engine.plot_elbow_method(X_train_clust, output_dir=grafici_dir)
 
        # K scelto sulla base dell'analisi del gomito (default: 5).
        # Modificare questo valore dopo aver analizzato il grafico.
        k_scelto = 5
        train_clusters = engine.fit(X_train_clust, k=k_scelto)
        df_train_processato = pd.concat(
            [df_train_processato, train_clusters], axis=1
        )
 
        # Salvataggio checkpoint del training set completamente processato.
        output_train_path = os.path.join(output_dir, 'train_processato.csv')
        df_train_processato.to_csv(output_train_path, index=False)
 
        print(f"\n{'=' * 60}")
        print(f"  RIEPILOGO TRAIN PROCESSATO")
        print(f"{'=' * 60}")
        print(f"  {'Righe:':<40} {df_train_processato.shape[0]:>8,}")
        print(f"  {'Colonne:':<40} {df_train_processato.shape[1]:>8}")
        print(f"  {'Valori mancanti residui:':<40} {df_train_processato.isnull().sum().sum():>8}")
        print(f"  {'Salvato in:':<40} {output_train_path}")
        print(f"{'=' * 60}")
 
        # ====================================================================
        # FASE 7 — PREPROCESSING VALIDATION E TEST INTERNO
        # ====================================================================
        # Entrambi i set vengono preprocessati con is_train=False:
        # SOLO transform, usando gli artefatti fittati sul train.
        # Ciò garantisce che non ci sia data leakage dal validation/test al train.
        dataset_da_processare = {
            "VALIDATION": {
                "dati":      dati_vali,
                "filename":  'val_processato.csv',
                "ha_target": True,   # il validation ha damage_grade (split interno)
            },
            "TEST": {
                "dati":      dati_test,
                "filename":  'test_processato.csv',
                "ha_target": True,   # anche il test interno ha il target
            },
        }
 
        df_processati = {}   # raccoglie i DataFrame processati per uso successivo
 
        for label, info in dataset_da_processare.items():
 
            # Preprocessing con artefatti del train (is_train=False → solo transform)
            pp = Preprocessing(
                info["dati"],
                scaler=scaler_addestrato,
                imputer_num=imputer_num,
                imputer_bin=imputer_bin,
                imputer_cat=imputer_cat,
                colonne_eliminate=colonne_eliminate,
                lista_colonne=lista_colonne_train,
                is_train=False,
            )
            df_proc = pp.esegui()
 
            # Clustering: predict con il modello fittato sul train.
            # Il target viene rimosso prima di passare al cluster engine
            # perché non è una feature di input del modello K-Means.
            X_proc   = df_proc.drop(columns=['damage_grade']) if info["ha_target"] else df_proc
            clusters = engine.predict(X_proc)
            df_proc  = pd.concat([df_proc, clusters], axis=1)
 
            # Salvataggio checkpoint
            output_path = os.path.join(output_dir, info["filename"])
            df_proc.to_csv(output_path, index=False)
 
            df_processati[label] = df_proc
 
            print(f"\n{'=' * 60}")
            print(f"  RIEPILOGO {label} PROCESSATO")
            print(f"{'=' * 60}")
            print(f"  {'Righe:':<40} {df_proc.shape[0]:>8,}")
            print(f"  {'Colonne:':<40} {df_proc.shape[1]:>8}")
            print(f"  {'Valori mancanti residui:':<40} {df_proc.isnull().sum().sum():>8}")
            print(f"  {'Salvato in:':<40} {output_path}")
            print(f"{'=' * 60}")
 
        # Estrazione comoda dei due DataFrame processati
        df_val_processato  = df_processati["VALIDATION"]
        df_test_processato = df_processati["TEST"]
 
        # ====================================================================
        # FASE 8 — FEATURE SELECTION SEARCH (RandomizedSearchCV condizionale)
        # ====================================================================
        # Esplora lo spazio condizionale:
        #   5 selettori × 2 modelli = fino a 10 dizionari di iperparametri
        #
        # La ricerca usa cross-validation INTERNA sul train set: il validation
        # rimane intatto per la valutazione finale, senza nessun data leakage.
        #
        # Al termine, search.best_pipeline_ contiene il selettore + modello
        # riaddestrati su tutto X_train (refit=True è impostato internamente).
        print(f"\n{'=' * 60}")
        print(f"  FASE 8: FEATURE SELECTION SEARCH")
        print(f"{'=' * 60}")
 
        # Separazione feature e target per tutti e tre i set
        X_train_fs = df_train_processato.drop(columns=['damage_grade'])
        y_train_fs = df_train_processato['damage_grade']
 
        X_val_fs   = df_val_processato.drop(columns=['damage_grade'])
        y_val_fs   = df_val_processato['damage_grade']
 
        # Il test interno ha sempre il target (è una porzione del dataset etichettato)
        X_test_fs  = df_test_processato.drop(columns=['damage_grade'], errors='ignore')
 
        # Istanziazione e avvio della ricerca.
        # include_sfs=True  → include SFSSelector (più lento, ~10 min in più)
        # include_sfs=False → esclude SFSSelector per test rapidi o macchine lente
        search = FeatureSelectionSearch(
            n_iter=30,           # configurazioni totali da campionare
            cv=3,                # fold di cross-validation interna
            include_sfs=True,    # False per test rapidi o macchine lente
            verbose=1,
            output_dir=output_dir,
        )
 
        # fit() lancia RandomizedSearchCV e riaddestra la pipeline migliore
        # su tutto X_train_fs (refit=True è impostato internamente in FeatureSelectionSearch)
        search.fit(X_train_fs, y_train_fs)
 
        # Salvataggio tabella completa dei risultati (utile per il report finale)
        output_fs_path = os.path.join(output_dir, 'feature_selection_results.csv')
        search.get_results().to_csv(output_fs_path, index=False)
        print(f"  Risultati ricerca salvati in: {output_fs_path}")
 
        # Applicazione del miglior selettore ai tre set.
        # transform() usa il selettore già fittato dentro best_pipeline_,
        # quindi non c'è rischio di data leakage.
        X_train_sel = search.transform(X_train_fs)
        X_val_sel   = search.transform(X_val_fs)
        X_test_sel  = search.transform(X_test_fs)
 
        # Riepilogo della configurazione vincente
        best_params = search.get_best_params()
        print(f"\n  Miglior selettore : {type(best_params['selector']).__name__}")
        print(f"  Miglior modello   : {type(best_params['model']).__name__}")
        print(f"  Score CV migliore : {search.get_best_score():.4f}")
        print(f"  Feature selezionate: {X_train_sel.shape[1]}")
 
        # Ricomposizione dei DataFrame con il target affiancato alle feature.
        # reset_index(drop=True) garantisce allineamento degli indici dopo lo split.
        df_train_finale = pd.concat(
            [X_train_sel, y_train_fs.reset_index(drop=True)], axis=1
        )
        df_val_finale = pd.concat(
            [X_val_sel, y_val_fs.reset_index(drop=True)], axis=1
        )
 
        # Il test interno ha il target; gestiamo anche il caso in cui non ci fosse
        # (es. se in futuro si passa direttamente Test_Values.csv di DrivenData)
        if 'damage_grade' in df_test_processato.columns:
            y_test_fs      = df_test_processato['damage_grade']
            df_test_finale = pd.concat(
                [X_test_sel, y_test_fs.reset_index(drop=True)], axis=1
            )
        else:
            df_test_finale = X_test_sel
 
        # Salvataggio dataset finali pronti per train_model.py
        df_train_finale.to_csv(os.path.join(output_dir, 'train_finale.csv'), index=False)
        df_val_finale.to_csv(os.path.join(output_dir,   'val_finale.csv'),   index=False)
        df_test_finale.to_csv(os.path.join(output_dir,  'test_finale.csv'),  index=False)
 
        # ====================================================================
        # FASE 9 — PREPROCESSING E CLUSTERING DEL TEST UFFICIALE DRIVENDATA
        # ====================================================================
        # Test_Values.csv è il dataset ufficiale di DrivenData:
        #   - NON ha la colonna damage_grade (il target è sconosciuto)
        #   - building_id deve essere preservato per costruire il file di submission
        #   - va preprocessato con gli stessi artefatti del training set
        print(f"\n{'=' * 60}")
        print(f"  FASE 9: PREPROCESSING TEST UFFICIALE DRIVENDATA")
        print(f"{'=' * 60}")
 
        path_test_ufficiale = os.path.join(data_dir, 'Test_Values.csv')
        print(f"  Caricamento Test_Values.csv...")
        df_test_ufficiale = scegli_opener(path_test_ufficiale).open(path_test_ufficiale)
 
        print(f"  {'Righe test ufficiale:':<40} {df_test_ufficiale.shape[0]:>8,}")
        print(f"  {'Colonne test ufficiale:':<40} {df_test_ufficiale.shape[1]:>8}")
 
        # Salviamo building_id PRIMA di qualsiasi preprocessing:
        # il preprocessing rimuove colonne non informative e non deve
        # alterare l'indice identificativo usato per la submission.
        building_ids_submission = df_test_ufficiale['building_id'].copy()
 
        # Rimuoviamo building_id dal DataFrame di input: non è una feature
        # e potrebbe essere confuso come variabile numerica dal modello.
        df_test_ufficiale = df_test_ufficiale.drop(columns=['building_id'])
 
        # Preprocessing test ufficiale (is_train=False): SOLO transform,
        # usando gli stessi artefatti fittati sul training set.
        pp_test_uff = Preprocessing(
            df_test_ufficiale,
            scaler=scaler_addestrato,
            imputer_num=imputer_num,
            imputer_bin=imputer_bin,
            imputer_cat=imputer_cat,
            colonne_eliminate=colonne_eliminate,
            lista_colonne=lista_colonne_train,
            is_train=False,   
        )
        df_test_uff_proc = pp_test_uff.esegui()
 
        # Clustering test ufficiale: predict con il modello fittato sul train.
        # Il test ufficiale non ha il target, quindi non serve drop.
        clusters_test_uff = engine.predict(df_test_uff_proc)
        df_test_uff_proc  = pd.concat([df_test_uff_proc, clusters_test_uff], axis=1)
 
        # Applicazione del selettore di feature al test ufficiale.
        # Usa lo stesso selettore già fittato sul train (nessun leakage).
        df_test_uff_sel = search.transform(df_test_uff_proc)
 
        # Salvataggio del test ufficiale processato e pronto per la submission.
        # building_id viene incluso direttamente nel CSV: train_model.py lo leggerà
        # da qui senza dover rielaborare il file grezzo.
        output_test_uff_path = os.path.join(output_dir, 'test_ufficiale_processato.csv')
        df_test_uff_proc_con_id = df_test_uff_sel.copy()
        df_test_uff_proc_con_id.insert(
            0, 'building_id', building_ids_submission.values
        )
        df_test_uff_proc_con_id.to_csv(output_test_uff_path, index=False)
 
        print(f"\n{'=' * 60}")
        print(f"  RIEPILOGO TEST UFFICIALE PROCESSATO")
        print(f"{'=' * 60}")
        print(f"  {'Righe:':<40} {df_test_uff_proc.shape[0]:>8,}")
        print(f"  {'Colonne (senza building_id):':<40} {df_test_uff_sel.shape[1]:>8}")
        print(f"  {'Valori mancanti residui:':<40} {df_test_uff_proc.isnull().sum().sum():>8}")
        print(f"  {'Salvato in:':<40} {output_test_uff_path}")
        print(f"{'=' * 60}")
 
        # ====================================================================
        # FASE 10 — RIEPILOGO DATASET E AVVIO TRAINING
        # ====================================================================
        print(f"\n{'=' * 60}")
        print(f"  RIEPILOGO DATASET FINALI")
        print(f"{'=' * 60}")
        print(f"  {'Train finale — righe:':<40} {df_train_finale.shape[0]:>8,}")
        print(f"  {'Train finale — colonne:':<40} {df_train_finale.shape[1]:>8}")
        print(f"  {'Val finale — righe:':<40} {df_val_finale.shape[0]:>8,}")
        print(f"  {'Val finale — colonne:':<40} {df_val_finale.shape[1]:>8}")
        print(f"  {'Test finale — righe:':<40} {df_test_finale.shape[0]:>8,}")
        print(f"  {'Test finale — colonne:':<40} {df_test_finale.shape[1]:>8}")
        print(f"  {'Test ufficiale — righe:':<40} {df_test_uff_sel.shape[0]:>8,}")
        print(f"  {'Test ufficiale — colonne:':<40} {df_test_uff_sel.shape[1]:>8}")
        print(f"  Dataset salvati in: {output_dir}")
        print(f"{'=' * 60}")
        print(f"\n  Nota: analizzare grafici/clustering_elbow.png per confermare k={k_scelto}")

        # Determina il modello migliore trovato dalla feature selection search
        # e avvia direttamente training, valutazione e generazione submission.
        best_model_obj  = search.get_best_params()['model']
        nome_modello    = 'rf' if isinstance(best_model_obj, RandomForestClassifier) else 'knn'

        print(f"\n  Modello migliore dalla feature selection search: "
              f"{type(best_model_obj).__name__} → avvio training finale...")

        avvia_training(model=nome_modello, output_dir=output_dir)
 
    # ── GESTIONE ERRORI ───────────────────────────────────────────────────
    except Exception as ex:
        print(f"\n{'=' * 60}")
        print(f"  ERRORE DURANTE L'ESECUZIONE")
        print(f"{'=' * 60}")
        print(f"  {ex}")
        print(f"{'=' * 60}")
        raise