import pandas as pd
from sklearn.preprocessing import StandardScaler


COLONNE_CONTINUE = [
    'age',
    'area_percentage',
    'height_percentage',
    'count_floors_pre_eq',
    'count_families'
]


class DataScaling:
    """
    Standardizzazione delle feature numeriche continue del dataset tramite StandardScaler.
    Applica il ridimensionamento (media 0, deviazione standard 1) esclusivamente
    alle variabili continue preservando le etichette ed evitando il data leakage.
    """

    def __init__(self, dataframe: pd.DataFrame, scaler=None, is_train: bool = True):
        """
        Inizializza la classe predisponendo lo StandardScaler e la modalità operativa.

        Input:
          - dataframe (pd.DataFrame): Dataset contenente le feature da standardizzare.
          - scaler (StandardScaler, opzionale): Scaler pre-addestrato sul train set (obbligatorio se is_train=False).
          - is_train (bool, default=True): Flag che indica se siamo in fase di training o test.

        Output:
          - Nessuno.
        """
        self.df = dataframe.copy()
        self.is_train = is_train
        if is_train and scaler is None:
            self.scaler = StandardScaler()
        else:
            self.scaler = scaler

    def standardizza(self) -> pd.DataFrame:
        """
        Standardizza esclusivamente le feature numeriche continue
        (evita ID e variabili binarie/categoriche).

        Input:
          - Nessuno.

        Output:
          - pd.DataFrame: Dataset standardizzato.
        """
        colonne_da_standardizzare = [col for col in COLONNE_CONTINUE if col in self.df.columns]

        if not colonne_da_standardizzare:
            print(f"  Nessuna colonna continua trovata: standardizzazione saltata.")
            return self.df

        if self.is_train:
            self.df[colonne_da_standardizzare] = self.scaler.fit_transform(
                self.df[colonne_da_standardizzare]
            )
            modalita = "fit_transform"
        else:
            if self.scaler is None:
                raise ValueError("Sul test è obbligatorio passare lo scaler addestrato sul train.")
            self.df[colonne_da_standardizzare] = self.scaler.transform(
                self.df[colonne_da_standardizzare]
            )
            modalita = "transform"

        print(f"  {'Colonne standardizzate (' + modalita + '):':<40} {len(colonne_da_standardizzare):>8}")
        print(f"  Colonne: {colonne_da_standardizzare}")

        return self.df
