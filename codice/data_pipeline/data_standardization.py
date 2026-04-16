import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataScaling:
    def __init__(self, dataframe: pd.DataFrame, scaler = None, is_train = True):
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
        colonne_continue = [
            'age',
            'area_percentage',
            'height_percentage',
            'count_floors_pre_eq',
            'count_families'
        ]

        colonne_da_standardizzare = [col for col in colonne_continue if col in self.df.columns]

        if not colonne_da_standardizzare:
            print("Nessuna colonna continua trovata per la standardizzazione.")
            return self.df
 
        if self.is_train:
            self.df[colonne_da_standardizzare] = self.scaler.fit_transform(
                self.df[colonne_da_standardizzare]
            )
            print(f"Standardizzazione fit_transform applicata su: {colonne_da_standardizzare}")
        else:
            if self.scaler is None:
                raise ValueError("Sul test è obbligatorio passare lo scaler addestrato sul train.")
            self.df[colonne_da_standardizzare] = self.scaler.transform(
                self.df[colonne_da_standardizzare]
            )
            print(f"Standardizzazione transform applicata su: {colonne_da_standardizzare}")
 
        return self.df
