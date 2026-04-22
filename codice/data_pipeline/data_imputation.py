import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer


class DataImputation:
    """Gestisce i valori mancanti con imputazione multivariata."""

    def __init__(self, dataframe: pd.DataFrame, imputer_num=None, imputer_cat=None, is_train=True):
        self.df = dataframe.copy()
        self.is_train = is_train
        self.imputer_num = imputer_num
        self.imputer_cat = imputer_cat

    def imputa(self) -> pd.DataFrame:
        """Esegue l'imputazione separata per numeriche e categoriche."""
        n_nan_prima = self.df.isnull().sum().sum()
        print(f"  {'NaN prima dell\'imputazione:':<40} {n_nan_prima:>8}")

        if n_nan_prima == 0:
            print(f"  Nessun valore mancante: imputazione saltata.")
            return self.df

        colonne_escluse = ['building_id', 'damage_grade']
        colonne_numeriche = [
            c for c in self.df.select_dtypes(include='number').columns
            if c not in colonne_escluse
        ]
        colonne_categoriche = self.df.select_dtypes(include=['object', 'string']).columns.tolist()

        if colonne_numeriche:
            if self.is_train:
                self.imputer_num = IterativeImputer(max_iter=10, random_state=42)
                self.df[colonne_numeriche] = self.imputer_num.fit_transform(self.df[colonne_numeriche])
            else:
                self.df[colonne_numeriche] = self.imputer_num.transform(self.df[colonne_numeriche])
            print(f"  {'Numeriche (IterativeImputer):':<40} {len(colonne_numeriche):>7} col")

        if colonne_categoriche:
            if self.is_train:
                self.imputer_cat = SimpleImputer(strategy='most_frequent')
                self.df[colonne_categoriche] = self.imputer_cat.fit_transform(self.df[colonne_categoriche])
            else:
                self.df[colonne_categoriche] = self.imputer_cat.transform(self.df[colonne_categoriche])
            print(f"  {'Categoriche (moda):':<40} {len(colonne_categoriche):>7} col")

        n_nan_dopo = self.df.isnull().sum().sum()
        print(f"  {'NaN dopo l\'imputazione:':<40} {n_nan_dopo:>8}")

        return self.df