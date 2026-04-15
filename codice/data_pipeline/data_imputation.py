import pandas as pd
from sklearn.ensemble import RandomForestRegressor

class DataImputation:
    """Gestisce l'imputazione dei valori mancanti."""

    def __init__(self, dataframe: pd.DataFrame, scaler=None, is_train=True):
        self.df = dataframe.copy()
        self.scaler = scaler
        self.is_train = is_train

    def imputa(self) -> pd.DataFrame:
        """Esegue l'imputazione multivariata."""
        n = self.df.isnull().sum().sum()

        if n > 0:
            print(f"\nIndividuati {n} valori mancanti. Avvio imputazione multivariata (Random Forest)...")
            self.df = self.gestisci_valori_mancanti_multivariata(self.df)
            print("Pulizia completata!")
        else:
            print("\nNessun valore mancante trovato.")

        return self.df

    def gestisci_valori_mancanti_multivariata(self, da: pd.DataFrame) -> pd.DataFrame:
        """
        Predice i valori mancanti nel dataset.
        """
        cols_with_nan = da.columns[da.isnull().any()].tolist()

        for target_col in cols_with_nan:
            # Usa tutte le altre colonne che non hanno NaN come feature
            features = da.columns[da.notnull().all()].tolist()

            if not features:
                da[target_col] = da[target_col].fillna(da[target_col].mean())
                continue

            train_data = da[da[target_col].notnull()]
            predict_data = da[da[target_col].isnull()]

            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(train_data[features], train_data[target_col])

            predictions = model.predict(predict_data[features])
            da.loc[da[target_col].isnull(), target_col] = predictions

        return da
