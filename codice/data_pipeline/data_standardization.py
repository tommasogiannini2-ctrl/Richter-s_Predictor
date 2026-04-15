import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataScaling:
    def __init__(self, dataframe: pd.DataFrame, scaler, is_train=True):
        self.df = dataframe.copy()
        if is_train and scaler is None:
            self.scaler = StandardScaler()
        else:
            self.scaler = scaler
        self.is_train = is_train

    def standardizza(self):
        """
        Standardizza esclusivamente le feature numeriche continue (evitando di modificare ID o variabili categoriche).
        """
        print("\n--- STANDARDIZZAZIONE ---")
        # Identifichiamo le variabili continue previste dal problema
        colonne_continue = [
            'age',
            'area_percentage',
            'height_percentage',
            'count_floors_pre_eq',
            'count_families'
        ]

        # Filtriamo le colonne per accertarci che siano presenti nel dataframe
        colonne_da_standardizzare = [col for col in colonne_continue if col in self.df.columns]

        if not colonne_da_standardizzare:
            print("Nessuna colonna continua trovata per la standardizzazione.")
            return

        if self.is_train:
            # Per i dati di Train calcoliamo (fit) e applichiamo (transform) la standardizzazione
            self.df[colonne_da_standardizzare] = self.scaler.fit_transform(self.df[colonne_da_standardizzare])
            print(f"Standardizzazione calcolata e applicata (fit_transform) su: {colonne_da_standardizzare}")
        else:
            # Per i dati di Test applichiamo (transform) le metriche calcolate precedentemente sul Train per evitare Data Leakage
            self.df[colonne_da_standardizzare] = self.scaler.transform(self.df[colonne_da_standardizzare])
            print(f"Standardizzazione applicata (transform) su: {colonne_da_standardizzare}")

        print("\nEsempio di valori standardizzati (prime 5 righe delle colonne modificate):")
        print(self.df[colonne_da_standardizzare].head())

        return self.df
