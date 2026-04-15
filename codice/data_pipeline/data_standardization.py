import pandas as pd

class DataScaling:
    """Gestisce esclusivamente la standardizzazione dei dati."""

    def __init__(self, dataframe: pd.DataFrame, scaler, is_train=True):
        self.df = dataframe.copy()
        self.scaler = scaler
        self.is_train = is_train

    def standardizza(self, is_train=True):
        """Standardizza le feature numeriche continue."""
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
            return

        if is_train:
            self.df[colonne_da_standardizzare] = self.scaler.fit_transform(
                self.df[colonne_da_standardizzare]
            )
            print(f"Scaler media di train: {self.scaler.mean_}")
            print(f"Standardizzazione calcolata e applicata (fit_transform) su: {colonne_da_standardizzare}")
        else:
            self.df[colonne_da_standardizzare] = self.scaler.transform(
                self.df[colonne_da_standardizzare]
            )
            print(f"Standardizzazione applicata (transform) su: {colonne_da_standardizzare}")

        print("\nEsempio di valori standardizzati (prime 5 righe):")
        print(self.df[colonne_da_standardizzare].head())

        return self.df
