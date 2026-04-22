# 🏔️ Richter's Predictor — Earthquake Damage Prediction

> Algoritmo di intelligenza artificiale per la predizione del livello di danno subito dagli edifici durante il terremoto di Gorkha (Nepal, 2015).

**Fonte competizione:**
DrivenData. (2019). *Richter's Predictor: Modeling Earthquake Damage.*
https://www.drivendata.org/competitions/57/nepal-earthquake/


## 📝 Descrizione del problema

L'obiettivo è predire la variabile **`damage_grade`**, che rappresenta il livello di danno subito da un edificio a seguito del terremoto di Gorkha (Nepal, 2015). Si tratta di un problema di **classificazione multi-classe ordinale** su tre livelli:

| Damage Grade | Descrizione |
| :---: | :--- |
| **1** | Danno lieve |
| **2** | Danno medio |
| **3** | Distruzione quasi totale |

La metrica ufficiale della competizione è la **micro-averaged F1 score**.

## ⚙️ Requisiti e installazione

**Python richiesto:** 3.14 

### Installazione delle dipendenze

```bash
pip install -r requirements.txt
```

---

## 📊 Struttura dei dati

Il dataset contiene **39 colonne**: `building_id` (identificatore univoco) e 38 feature predittive.

### Feature principali

| Feature | Tipo | Descrizione | Valori / Range |
| :--- | :---: | :--- | :--- |
| `geo_level_1_id` | Int | Regione geografica più ampia | 0 – 30 |
| `geo_level_2_id` | Int | Sotto-regione intermedia | 0 – 1427 |
| `geo_level_3_id` | Int | Sotto-regione più specifica | 0 – 12567 |
| `count_floors_pre_eq` | Int | Numero di piani prima del terremoto | Intero |
| `age` | Int | Età dell'edificio in anni | Intero |
| `area_percentage` | Int | Area normalizzata dell'edificio | Intero |
| `height_percentage` | Int | Altezza normalizzata dell'edificio | Intero |
| `land_surface_condition` | Cat | Condizione della superficie del terreno | n, o, t |
| `foundation_type` | Cat | Tipo di fondamenta | h, i, r, u, w |
| `roof_type` | Cat | Tipo di tetto | n, q, x |
| `ground_floor_type` | Cat | Tipo di piano terra | f, m, v, x, z |
| `other_floor_type` | Cat | Tipo costruttivo dei piani superiori | j, q, s, x |
| `position` | Cat | Posizione dell'edificio | j, o, s, t |
| `plan_configuration` | Cat | Configurazione planimetrica | a, c, d, f, m, n, o, q, s, u |
| `legal_ownership_status` | Cat | Status legale di proprietà del terreno | a, r, v, w |
| `count_families` | Int | Numero di famiglie nell'edificio | Intero |
| `has_superstructure_*` | Binary | Materiale di costruzione della sovrastruttura | 0 / 1 |
| `has_secondary_use_*` | Binary | Uso secondario dell'edificio | 0 / 1 |

---

## ▶️ Istruzioni per l'esecuzione

### 1. Preprocessing completo (train + test)

```bash
python main.py
```

All'avvio, il programma chiede interattivamente se ridurre la dimensione del dataset:

```
--- ANALISI DIMENSIONI DATASET ---
Record attuali: 260601
Memoria occupata: 74.35 MB

La dimensione del dataset è ottimale? (s/n):
```

**Opzioni di input:**

| Input | Effetto |
| :--- | :--- |
| `s` | Prosegue con il dataset completo (260.601 record) |
| `n` | Richiede il limite in MB da rispettare, poi esegue il campionamento stratificato |

> **Consiglio:** su macchine con RAM limitata, rispondere `n` e impostare un limite di 20–30 MB (circa 70.000–100.000 record). La stratificazione garantisce che le proporzioni delle classi rimangano invariate.

**Output prodotti:**

- `output/train_processato.csv` — training set pronto per l'addestramento
- `output/test_processato.csv` — test set pronto per la predizione

## 🔧 Pipeline di preprocessing

La pipeline viene eseguita automaticamente da `main.py` tramite la classe `Preprocessing`. Le fasi sono le seguenti:

### Fase 1 — Pulizia (`data_cleaning.py`)

| Operazione | Dettaglio |
| :--- | :--- |
| Rimozione duplicati | Righe identiche eliminate |
| Correzione range numerici | `age > 800` o `< 0` → NaN; `count_floors_pre_eq > 15` o `≤ 0` → NaN; percentuali fuori `(0, 100]` → NaN |
| Rimozione record con target nullo | Solo sul train set |
| Rimozione outlier strutturali | Record con valori categoriali fuori dominio eliminati (train) o convertiti in NaN (test) |
| Eliminazione record con troppi null | Record con oltre il 30% di valori mancanti rimossi (solo train) |
| Eliminazione colonne con troppi null | Colonne con oltre il 40% di valori mancanti rimosse (solo train) |

### Fase 2 — Imputazione (`data_imputation.py`)

| Tipo di feature | Strategia |
| :--- | :--- |
| Numeriche | `IterativeImputer` (regressione multivariata, max 10 iterazioni) |
| Categoriche | `SimpleImputer` con strategia `most_frequent` (moda) |

Sul **test set** vengono riapplicati gli imputer già addestrati sul train, evitando data leakage.

### Fase 3 — Encoding (`data_encoding.py`)

Le 8 variabili categoriali vengono trasformate in **dummy variables** (One-Hot Encoding senza drop della prima categoria). 

### Fase 4 — Standardizzazione (`data_standardization.py`)

Le 5 feature continue (`age`, `area_percentage`, `height_percentage`, `count_floors_pre_eq`, `count_families`) vengono standardizzate con `StandardScaler` (media 0, deviazione standard 1). Sul test si applica il `transform` dello scaler addestrato sul train.

## 🔍 Interpretazione dei risultati

### Metrica principale: Micro-F1

La competizione valuta i modelli con la **micro-averaged F1 score**. Questo valore aggrega TP, FP e FN su tutte le classi prima di calcolare F1, dando peso uguale a ogni singola predizione. Un valore più alto è migliore; il benchmark di riferimento su DrivenData è circa **0.75**.

### Cosa osservare nei grafici

**Matrice di confusione normalizzata**
Ogni cella mostra la percentuale di record della classe reale (riga) classificati nella classe predetta (colonna). La diagonale principale deve essere la più alta possibile. Errori comuni nel dataset Nepal: confusione tra classe 2 e classe 3, e scarso riconoscimento della classe 1 (minoritaria).

**Curve ROC**
Ogni curva mostra il trade-off tra TPR e FPR per una classe (approccio One-vs-Rest). Un'AUC vicina a 1.0 indica un'ottima separazione; AUC = 0.5 equivale a un classificatore casuale. La classe 1 (danno lieve) tende ad avere AUC inferiore per via dello squilibrio del dataset.

**Bar chart Precision / Recall / F1**
Permette di confrontare le prestazioni classe per classe. Se la barra del Recall è molto più bassa della Precision su una classe, il modello tende a non riconoscerla (troppi falsi negativi). La linea rossa tratteggiata indica la Micro-F1 globale come riferimento.

**Distribuzione classi reali vs predette**
Se le barre "Predetti" differiscono significativamente da quelle "Reali", il modello ha un bias sistematico verso alcune classi. In presenza di forte squilibrio (classe 1 è circa il 10% del dataset), considerare tecniche di bilanciamento come SMOTE o pesi di classe.

### Squilibrio del dataset

Il dataset è sbilanciato secondo questa distribuzione approssimativa:

| Classe | % approssimativa |
| :---: | :---: |
| 1 — Danno lieve | ~10% |
| 2 — Danno medio | ~57% |
| 3 — Distruzione | ~33% |

Questo squilibrio impatta le metriche per classe: la classe 1 sarà generalmente quella con F1 più bassa. La Micro-F1 riflette questa proporzione, mentre la Macro-F1 (media non pesata) può essere usata per valutare le prestazioni in modo più equo tra le classi.

---

## 📐 Formato di submission

Il file di submission deve essere un CSV con intestazione e due colonne:

| Colonna | Tipo | Descrizione |
| :--- | :--- | :--- |
| `building_id` | Int | Identificatore dal test set |
| `damage_grade` | Int (1–3) | Livello di danno predetto |

**Esempio:**
```csv
building_id,damage_grade
1148,1
5842,3
2593,2
```

---
