# Guida alla struttura del progetto

Questa guida descrive le cartelle e i file presenti nel repository `Ritchter-s_Predictor`.
Il progetto implementa una pipeline di machine learning per la competizione DrivenData
"Richter's Predictor", con preprocessing, selezione delle feature, addestramento,
valutazione e generazione della submission.

Nota: la cartella `.git/` non e' documentata file per file perche' contiene metadata
interno di Git. La cartella `output/`, se presente dopo l'esecuzione della pipeline,
contiene artefatti generati ed e' ignorata da Git.

## Vista rapida

```text
.
|-- .gitignore
|-- README.md
|-- config.example.yml
|-- config.full-training.yml
|-- requirements.txt
|-- codice/
|   |-- config_loader.py
|   |-- data_reduction.py
|   |-- file_test.py
|   |-- main.py
|   |-- plot.py
|   |-- data_pipeline/
|   |-- model_evaluation/
|   `-- tests/
|-- data/
|   |-- Test_Values.csv
|   |-- Train_Labels.csv
|   `-- Train_Values.csv
`-- docs/
    `-- PROJECT-GUIDE.md
```

## Root del repository

### `requirements.txt`

Elenco delle dipendenze Python bloccate a versione specifica. Include le librerie
principali per il progetto: `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`,
`seaborn`, `joblib`, `PyYAML`, `pytest` e `xgboost`.


### `config.full-training.yml`

Configurazione per un run completo e rappresentativo. Non riduce il dataset
(`data_reduction.enabled: false`), usa `k=5`, genera un elbow piu' ampio fino a
`12`, campiona fino a `50000` righe per il grafico elbow e imposta la feature
selection a `150` iterazioni.

Comando previsto:

```bash
python codice/main.py --config config.full-training.yml
```

### Configurazione operativa

La pipeline puo' essere guidata da YAML tramite `codice/main.py --config ...`.
Le chiavi effettivamente lette dal codice sono:

- `run.use_saved_model`: se `true` e `output/risultati/model_finale.pkl` esiste,
  la pipeline ricostruisce il preprocessing del test ufficiale e usa
  `predict_only()` per rigenerare la submission senza rifare ricerca e training.
- `data_reduction.enabled` e `data_reduction.max_memory_mb`: controllano il
  campionamento stratificato di debug tramite `DataReducer`.
- `clustering.k`: numero di cluster K-Means trasformati in dummy `cluster_*`.
- `clustering.elbow_max_k` e `clustering.elbow_sample_size`: influenzano solo il
  grafico elbow, non il numero finale di cluster.
- `feature_selection.n_iter`, `feature_selection.cv`,
  `feature_selection.include_sfs` e `feature_selection.verbose`: configurano
  `FeatureSelectionSearch` e il costo della ricerca.

`config.example.yml` e' pensato per run leggeri o debug perche' riduce il dataset.
`config.full-training.yml` e' il profilo piu' rappresentativo per produrre una
submission completa.

## Cartella `.idea/`

Cartella di configurazione dell'IDE JetBrains/PyCharm. Non serve all'esecuzione
della pipeline e normalmente resta locale allo sviluppatore.

### `.idea/.gitignore`

Regole Git specifiche per la cartella `.idea/`, create dall'IDE.


## Cartella `data/`

Contiene i dataset CSV originali della competizione. Questi file sono input della
pipeline e vengono letti da `codice/main.py`.

### `data/Train_Values.csv`

Feature di training. Contiene `260601` righe dati piu' intestazione. La chiave
primaria e' `building_id`; le altre colonne rappresentano geografia, dimensioni
dell'edificio, materiali, tipologie costruttive, proprieta' legale e usi secondari.

### `data/Train_Labels.csv`

Etichette di training. Contiene `260601` righe dati piu' intestazione, con colonne
`building_id` e `damage_grade`. Viene unito a `Train_Values.csv` su `building_id`.

### `data/Test_Values.csv`

Feature del test ufficiale DrivenData. Contiene `86868` righe dati piu' intestazione.
Ha lo stesso schema di feature di `Train_Values.csv`, ma non include `damage_grade`.

## Cartella `codice/`

Package Python principale del progetto. Contiene l'orchestratore della pipeline,
utility generali, plotting, riduzione dati, pipeline di preprocessing, valutazione
modelli e test.


### `codice/main.py`

Entry point principale della pipeline end-to-end. Puo' essere eseguito direttamente
o come modulo:

```bash
python codice/main.py
python -m codice.main
python codice/main.py --config config.example.yml
```

Responsabilita' principali:

- carica `Train_Values.csv` e `Train_Labels.csv`;
- unisce i dati e rimuove `building_id` dal training interno;
- genera grafici EDA tramite `Plotter`;
- applica eventuale riduzione stratificata tramite `DataReducer`;
- divide i dati in train, validation e test interno;
- applica preprocessing senza data leakage;
- addestra K-Means sul train e aggiunge feature `cluster_*`;
- esegue feature selection e ricerca iperparametri;
- preprocessa il test ufficiale;
- addestra o riusa il modello finale;
- genera metriche, grafici e `submission.csv`.

### `codice/config_loader.py`

Utility per configurazioni YAML:

- `load_config(path)`: carica un file YAML e restituisce un dizionario.
- `get_nested(config, path, default)`: legge chiavi annidate con notazione puntata,
  ad esempio `feature_selection.n_iter`.

### `codice/data_reduction.py`

Contiene `DataReducer`, classe che stima la memoria occupata dal dataset e puo'
ridurlo tramite campionamento stratificato preservando la distribuzione di
`damage_grade`.

Metodi principali:

- `get_info()`: restituisce numero di righe e memoria in MB.
- `riduci_per_memoria(limite_mb)`: riduce il dataset in base al limite indicato.
- `interfaccia_utente()`: modalita' interattiva per scegliere se ridurre i dati.

### `codice/plot.py`

Contiene `Plotter`, classe per l'analisi esplorativa automatica. Genera grafici su:

- distribuzione del target;
- valori mancanti;
- correlazioni;
- feature numeriche rispetto al target;
- feature categoriche rispetto al target;
- geografia e danno;
- distribuzione dell'eta' degli edifici.

I grafici vengono salvati in `output/grafici/` quando `output_dir` e' configurato.

### `codice/file_test.py`

File di test legacy basato su `unittest`. Copre riduzione dati, pulizia, selettori
di feature e ricerca di feature selection. Convive con i test piu' modulari presenti
in `codice/tests/`.

## Cartella `codice/data_pipeline/`

Contiene gli stadi riusabili della pipeline dati: apertura file, pulizia,
imputazione, target encoding geografico, one-hot encoding, standardizzazione,
split train/validation/test e clustering.


### `codice/data_pipeline/file_opener.py`

Astrazione per aprire file tabellari con Pandas.

Classi e funzioni:

- `AbstractOpener`: classe base che verifica esistenza file e delega il caricamento.
- `CSVOpener`: usa `pandas.read_csv`.
- `XLSOpener`: usa `pandas.read_excel`.
- `JSONOpener`: usa `pandas.read_json`.
- `scegli_opener(path)`: sceglie l'opener in base all'estensione.

### `codice/data_pipeline/data_cleaning.py`

Contiene `DataCleaning`, responsabile della pulizia dati.

Operazioni principali:

- rimozione duplicati solo in training;
- conversione a `NaN` di valori numerici fuori dominio;
- rimozione righe senza `damage_grade` in training;
- controllo dei domini per variabili categoriche e binarie;
- rimozione righe anomale in training;
- conversione ad `NaN` dei valori fuori dominio in test;
- rimozione righe o colonne con troppi null in training;
- applicazione al test delle colonne eliminate durante il training.

### `codice/data_pipeline/data_imputation.py`

Contiene `DataImputation`, che gestisce i valori mancanti evitando data leakage.

Strategie:

- numeriche continue: mediana;
- binarie `has_*`: moda;
- categoriche stringa: moda.

In training addestra i `SimpleImputer`; in validation/test riusa gli imputer gia'
addestrati.

### `codice/data_pipeline/data_encoding.py`

Contiene `DataEncoding`, che applica one-hot encoding alle feature categoriche:
`land_surface_condition`, `foundation_type`, `roof_type`, `ground_floor_type`,
`other_floor_type`, `position`, `plan_configuration`, `legal_ownership_status`.

In modalita' test riallinea le colonne allo schema del train con `reindex`.

### `codice/data_pipeline/data_standardization.py`

Contiene:

- `COLONNE_CONTINUE`: `age`, `area_percentage`, `height_percentage`,
  `count_floors_pre_eq`, `count_families`.
- `DataScaling`: standardizza solo queste colonne con `StandardScaler`.

In training esegue `fit_transform`; in validation/test usa solo `transform`.

### `codice/data_pipeline/preprocessing.py`

Contiene `Preprocessing`, orchestratore degli stadi di preprocessing, e
`dividi_train_validation_test`.

Sequenza gestita da `Preprocessing.esegui()`:

1. pulizia con `DataCleaning`;
2. imputazione con `DataImputation`;
3. target encoding delle colonne geografiche `geo_level_1_id`, `geo_level_2_id`,
   `geo_level_3_id`;
4. one-hot encoding con `DataEncoding`;
5. standardizzazione con `DataScaling`;
6. riallineamento di colonne e conservazione degli oggetti addestrati sul train.

`dividi_train_validation_test()` crea uno split stratificato 70% / 15% / 15%.

Contratto operativo:

- il train esegue `fit_transform` per pulizia, imputazione, target encoding,
  one-hot encoding e scaling;
- validation, test interno e test ufficiale riusano gli oggetti addestrati sul
  train, evitando data leakage;
- le colonne eliminate o create durante il train vengono conservate e riusate per
  riallineare validation/test con lo stesso schema;
- `damage_grade` resta presente solo dove serve alla valutazione, mentre
  `building_id` viene preservato per il test ufficiale per costruire
  `submission.csv`;
- se una categoria compare solo in validation/test, viene gestita con
  riallineamento delle colonne invece di modificare lo schema appreso sul train.

### `codice/data_pipeline/clustering.py`

Contiene `Clustering`, wrapper per K-Means.

Funzioni principali:

- `plot_elbow_method(...)`: genera `clustering_elbow.png`;
- `fit(train_df, k)`: addestra K-Means e restituisce colonne dummy `cluster_*`;
- `predict(test_df)`: assegna nuovi record ai cluster gia' addestrati e riallinea
  le colonne dummy.

## Cartella `codice/model_evaluation/`

Contiene feature selection, ricerca iperparametri, training finale, predizione,
valutazione e generazione dei report.


### `codice/model_evaluation/feature_select_extract.py`

Definisce selettori compatibili con l'API scikit-learn (`fit`, `transform`,
`get_info`) e una factory.

Selettori disponibili:

- `AllFeaturesSelector`: baseline che mantiene tutte le feature.
- `MutualInfoSelector`: seleziona le feature con mutual information piu' alta.
- `ReliefFSelector`: implementa una variante ReliefF campionata.
- `SFSSelector`: usa Sequential Feature Selection.
- `EmbeddedDTSelector`: usa importanze da Decision Tree.
- `crea_selector(nome, **kwargs)`: crea il selettore richiesto per nome.

### `codice/model_evaluation/validation.py`

Contiene la ricerca combinata di feature selection e modello.

Componenti principali:

- `SimpleProgressBar`: barra testuale per rendere leggibile l'avanzamento.
- `simple_progress_joblib(total)`: context manager che intercetta Joblib.
- `_build_search_space(include_sfs)`: costruisce lo spazio condizionale per
  selettori e modelli.
- `FeatureSelectionSearch`: incapsula `RandomizedSearchCV` su una pipeline
  `selector -> model`.

La ricerca prova combinazioni con Random Forest, KNN e AdaBoost, usando
`f1_micro` come metrica predefinita.

Dettagli importanti:

- lo spazio di ricerca e' una lista di dizionari condizionali: ogni dizionario
  abbina un selettore a un modello e ai soli iperparametri compatibili;
- `include_sfs=true` aggiunge Sequential Feature Selection, ma questa e' la parte
  piu' costosa perche' moltiplica i fit interni;
- `n_iter` indica il numero totale di configurazioni campionate, non il numero
  di prove per ogni modello;
- i risultati vengono convertiti in `feature_selection_results.csv` con rank,
  selettore, modello, score medio CV e deviazione standard;
- il best estimator resta disponibile per predizione, trasformazione delle
  feature e recupero dei parametri migliori.

### `codice/model_evaluation/train_model.py`

Modulo per addestramento finale e predizione.

Interfacce principali:

- `_build_model(...)`: costruisce Random Forest, KNN o AdaBoost.
- `run(...)`: carica i dataset finali, addestra il modello, valuta validation e
  test interno, salva il miglior checkpoint e genera `submission.csv`.
- `predict_only(...)`: carica `model_finale.pkl` e genera solo la submission.
- `main()`: entry point CLI.

Output principali in `output/risultati/`:

- `model_finale.pkl`;
- `model_finale_best_micro_f1.txt`;
- `submission.csv`.

`model_finale.pkl` non contiene solo l'estimatore: viene salvato come checkpoint
con le informazioni necessarie a riutilizzare il modello finale in modalita'
predict-only. `run(...)` valuta validation e test interno se i rispettivi file
sono presenti, poi produce la submission sul test ufficiale preprocessato.
`predict_only(...)` richiede invece un checkpoint gia' esistente e
`output/dataset/test_ufficiale_finale.csv`.

### `codice/model_evaluation/evaluation.py`

Contiene `ModelEvaluator`, responsabile del calcolo metriche e dei grafici di
valutazione.

Funzionalita' principali:

- calcolo Micro-F1;
- metriche aggregate e per classe;
- salvataggio metriche CSV;
- confusion matrix;
- curve ROC one-vs-rest quando sono disponibili probabilita';
- grafico precision/recall/F1;
- distribuzione delle classi;
- metodo `valuta_tutto()` per eseguire l'intero report.

## Cartella `codice/tests/`

Suite di test basata su `unittest`. I test sono granulari e verificano moduli
specifici della pipeline.


### `codice/tests/run_all_test.py`

Runner manuale che scopre ed esegue tutti i file `*_test.py` nella cartella.

Comando:

```bash
python codice/tests/run_all_test.py
```

### `codice/tests/config_loader_test.py`

Testa `load_config()` e `get_nested()`, inclusi caricamento YAML e recupero di
chiavi annidate.

### `codice/tests/file_opener_test.py`

Testa opener CSV, Excel e JSON, gestione file mancanti, eccezioni Pandas e scelta
dell'opener in base all'estensione.

### `codice/tests/data_cleaning_test.py`

Testa copia del DataFrame, rimozione duplicati, pulizia variabili numeriche,
gestione outlier strutturali, soglie sui null e sequenza di pulizia in train/test.

### `codice/tests/data_imputation_test.py`

Testa categorizzazione delle colonne, imputazione con `fit_transform` in training,
riuso degli imputer in test e gestione delle colonne escluse.

### `codice/tests/data_encoding_test.py`

Testa one-hot encoding delle variabili categoriche e riallineamento delle colonne
in modalita' test.

### `codice/tests/data_standardization_test.py`

Testa standardizzazione, gestione assenza colonne continue, condivisione delle
colonne continue con il clustering e errore se manca lo scaler in test.

### `codice/tests/preprocessing_test.py`

Testa il flusso completo di `Preprocessing`, sia in train sia in test, inclusa la
conservazione di `damage_grade` e lo split train/validation/test.

### `codice/tests/adaboost_test.py`

Test di integrazione per AdaBoost: costruzione modello, presenza nello spazio di
ricerca, training completo su dati sintetici e compatibilita' con
`FeatureSelectionSearch`.

## Cartella `docs/`

Contiene documentazione aggiuntiva del repository.

### `docs/PROJECT-GUIDE.md`

Questo documento. Serve come mappa file-per-file del progetto.

## Cartella `output/` generata a runtime

`output/` non e' presente come sorgente versionata, ma viene creata da `main.py`
durante l'esecuzione. E' ignorata da `.gitignore`.

Struttura attesa:

```text
output/
|-- dataset/
|-- grafici/
|-- risultati/
`-- eval/
```

### `output/dataset/`

Dataset intermedi e finali generati dalla pipeline:

- `train_processato.csv`, `val_processato.csv`, `test_processato.csv`;
- `train_finale.csv`, `val_finale.csv`, `test_finale.csv`;
- `test_ufficiale_finale.csv`.

I file `*_processato.csv` sono prodotti dopo preprocessing e clustering. I file
`*_finale.csv` sono quelli usati dal training finale dopo selezione/allineamento
delle feature. `test_ufficiale_finale.csv` conserva `building_id`, necessario per
scrivere la submission DrivenData.

### `output/grafici/`

Grafici EDA e grafico elbow del clustering.

### `output/risultati/`

Risultati finali della ricerca e del training:

- `feature_selection_results.csv`;
- `model_finale.pkl`;
- `model_finale_best_micro_f1.txt`;
- `submission.csv`.

`feature_selection_results.csv` serve per ispezionare le configurazioni provate
da `RandomizedSearchCV`. `model_finale.pkl` abilita la modalita'
`run.use_saved_model=true`. `submission.csv` e' il file finale da caricare sulla
piattaforma della competizione.

### `output/eval/`

Metriche e grafici di valutazione, separati in sottocartelle per validation e test
interno.

## Flusso operativo consigliato

1. Installare le dipendenze:

```bash
pip install -r requirements.txt
```

2. Eseguire un run leggero non interattivo:

```bash
python codice/main.py --config config.example.yml
```

3. Eseguire un run completo:

```bash
python codice/main.py --config config.full-training.yml
```

4. Eseguire i test:

```bash
python codice/tests/run_all_test.py
```
