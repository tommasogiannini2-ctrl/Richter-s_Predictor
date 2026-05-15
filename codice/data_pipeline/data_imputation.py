import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer


class DataImputation:
    """Gestisce i valori mancanti con imputazione multivariata."""

    def __init__(self, dataframe: pd.DataFrame, imputer_num=None, imputer_bin=None, imputer_cat=None, is_train: bool = True):
        self.df = dataframe.copy()
        self.is_train = is_train
 
        self.imputer_num = imputer_num
        self.imputer_bin = imputer_bin
        self.imputer_cat = imputer_cat
        self._escluse = {'building_id', 'damage_grade'}

    def imputa(self) -> pd.DataFrame:
        """Esegue l'imputazione separata per numeriche e categoriche."""
        n_nan_prima = self.df.isnull().sum().sum()
        print(f"  {'NaN prima dell\'imputazione:':<40} {n_nan_prima:>8}")
 
        if n_nan_prima == 0:
            print(f"  Nessun valore mancante: imputazione saltata.")
            return self.df
 
        col_num, col_bin, col_cat = self._categorizza_colonne()
 
        if col_num:
            self._imputa_gruppo(col_num, 'imputer_num', 'median',
                                'Numeriche continue (mediana)')
        if col_bin:
            self._imputa_gruppo(col_bin, 'imputer_bin', 'most_frequent',
                                'Binarie has_* (moda)')
        if col_cat:
            self._imputa_gruppo(col_cat, 'imputer_cat', 'most_frequent',
                                'Categoriche stringa (moda)')
 
        n_nan_dopo = self.df.isnull().sum().sum()
        print(f"  {'NaN dopo l\'imputazione:':<40} {n_nan_dopo:>8}")
 
        return self.df
    
    def _categorizza_colonne(self):
        """
        Suddivide le colonne del DataFrame in tre gruppi distinti.
        - col_num : numeriche continue (age, area_percentage, ecc.)
                    esclude le binarie has_* e le colonne di sistema
        - col_bin : variabili binarie 0/1 (tutte quelle che iniziano con 'has_')
                    hanno la propria strategia di imputazione (moda, non mediana,
                    perché la mediana potrebbe restituire 0.5 su dati sbilanciati)
        - col_cat : colonne di tipo object/string (le categoriche ancora non codificate)
        """
        col_num, col_bin, col_cat = [], [], []
 
        for c in self.df.columns:
            if c in self._escluse:
                continue
 
            if c.startswith('has_'):
                col_bin.append(c)
            elif self.df[c].dtype in ['object', 'string']:
                col_cat.append(c)
            elif pd.api.types.is_numeric_dtype(self.df[c]):
                col_num.append(c)
 
        print(f"  {'Colonne numeriche continue:':<40} {len(col_num):>8}")
        print(f"  {'Colonne binarie (has_*):':<40} {len(col_bin):>8}")
        print(f"  {'Colonne categoriche stringa:':<40} {len(col_cat):>8}")
 
        return col_num, col_bin, col_cat
 
    def _imputa_gruppo(self, colonne: list, attr_name: str, strategy: str, label: str):
        """
        Imputa un gruppo di colonne con la strategia specificata.
        """
        if self.is_train:
            imputer = SimpleImputer(strategy=strategy)
            self.df[colonne] = imputer.fit_transform(self.df[colonne])
            setattr(self, attr_name, imputer)
        else:
            imputer = getattr(self, attr_name)
            if imputer is None:
                raise ValueError(
                    f"[DataImputation] imputer '{attr_name}' è None in modalità TEST. "
                    f"Passa l'imputer fittato sul train nel costruttore."
                )
            self.df[colonne] = imputer.transform(self.df[colonne])
 
        n_nan = self.df[colonne].isnull().sum().sum()
        print(f"  {label + ':':<40} {len(colonne):>6} col  |  NaN residui: {n_nan}")