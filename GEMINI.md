# Richter's Predictor: Modeling Earthquake Damage 🇳🇵💥

Questo repository contiene il codice e l'analisi per la competizione di DrivenData **[Richter's Predictor: Modeling Earthquake Damage](https://www.drivendata.org/competitions/57/nepal-earthquake/page/136/)**.

## 📖 Descrizione del Problema
Nell'aprile del 2015, il Nepal è stato colpito dal devastante terremoto di Gorkha, di magnitudo 7.8, che ha causato la perdita di migliaia di vite e distrutto centinaia di migliaia di abitazioni. A seguito del disastro, è stata condotta un'imponente campagna di raccolta dati per documentare il livello di danno subito dagli edifici e le loro caratteristiche costruttive.

L'obiettivo di questa sfida è costruire un modello di Machine Learning in grado di prevedere il *livello di danno* agli edifici sulla base delle loro caratteristiche strutturali, di localizzazione e di utilizzo.

## 🎯 Obiettivo
Il task consiste in un problema di *classificazione ordinale* (o multiclasse). Dobbiamo prevedere la variabile target damage_grade, che rappresenta il livello di danno subito dall'edificio su una scala da 1 a 3:
* *1*: Danno lieve (Low damage)
* *2*: Danno medio (Medium amount of damage)
* *3*: Distruzione quasi totale (Almost complete destruction)

## 📊 I Dati
Il dataset è costituito da informazioni sulla struttura degli edifici, sul loro posizionamento e sullo stato di proprietà legale. Ogni riga rappresenta un edificio specifico colpito dal terremoto. Ci sono 39 colonne in totale: un identificatore univoco (building_id) e 38 feature predittive.
Nota: Le variabili categoriche sono state offuscate usando caratteri ASCII casuali.

Le feature includono:
* *Dati geografici*: geo_level_1_id, geo_level_2_id, geo_level_3_id (indicanti le macro e micro regioni).
* *Caratteristiche dell'edificio*: età (age), numero di piani pre-sisma (count_floors_pre_eq), area normalizzata (area_percentage), altezza normalizzata (height_percentage).
* *Materiali e struttura*: tipo di fondazione (foundation_type), tipo di tetto (roof_type), tipologia di piano terra (ground_floor_type), e numerose flag binarie (es. has_superstructure_adobe_mud, has_superstructure_cement_mortar_brick) che indicano i materiali costruttivi usati per la sovrastruttura.
* *Uso e proprietà*: stato legale della proprietà (legal_ownership_status), numero di famiglie residenti (count_families), e usi secondari dell'edificio (agricolo, alberghiero, affitto, ecc.).

## 📏 Metrica di Valutazione
La performance dei modelli viene valutata utilizzando il *Micro-Averaged F1 Score*.
Questa metrica è adatta per bilanciare precision e recall in problemi multiclasse ed è definita come:

$$F_{micro} = \frac{2 \cdot P_{micro} \cdot R_{micro}}{P_{micro} + R_{micro}}$$

Dove Precision ($P_{micro}$) e Recall ($R_{micro}$) sono calcolate aggregando i Veri Positivi, Falsi Positivi e Falsi Negativi globalmente su tutte le tre classi.
In Python (con scikit-learn), corrisponde a: f1_score(y_true, y_pred, average='micro').

## 📤 Formato della Sottomissione
Le previsioni devono essere esportate in un file CSV contenente esattamente due colonne: building_id e damage_grade. I valori previsti devono essere numeri interi (1, 2 o 3) senza decimali.

Esempio di formato:
```csv
building_id,damage_grade
11456,1
16528,2
3253,3
...