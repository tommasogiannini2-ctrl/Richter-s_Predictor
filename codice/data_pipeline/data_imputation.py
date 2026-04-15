import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class DataImputation:
    def __init__(self, is_train=True, models=None):
        self.is_train = is_train
        self.models = models if models is not None else {}

    def imputa(self, df: pd.DataFrame) -> pd.DataFrame:
        da = df.copy()
        cols_with_nan = da.columns[da.isnull().any()].tolist()

        # Se siamo in fase di test, dobbiamo guardare le colonne che il TRAIN si aspettava di imputare
        if not self.is_train:
            cols_to_process = list(self.models.keys())
        else:
            cols_to_process = cols_with_nan

        for target_col in cols_to_process:
            features = [c for c in da.columns if c != target_col and da[c].notnull().all()]

            if not da[target_col].isnull().any():
                continue

            if self.is_train:
                # --- FASE TRAINING ---
                train_data = da[da[target_col].notnull()]
                predict_data = da[da[target_col].isnull()]

                if train_data.empty or not features:
                    mean_val = da[target_col].mean()
                    da[target_col] = da[target_col].fillna(mean_val)
                    self.models[target_col] = ("mean", mean_val)
                else:
                    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                    model.fit(train_data[features], train_data[target_col])

                    # Salviamo modello e lista feature usate
                    self.models[target_col] = (model, features)

                    predictions = model.predict(predict_data[features])
                    da.loc[da[target_col].isnull(), target_col] = predictions
            else:
                # --- FASE TEST (Prevenzione Leakage) ---
                if target_col in self.models:
                    stored_data = self.models[target_col]

                    if stored_data[0] == "mean":
                        da[target_col] = da[target_col].fillna(stored_data[1])
                    else:
                        model, features = stored_data
                        predict_data = da[da[target_col].isnull()]

                        predictions = model.predict(predict_data[features])
                        da.loc[da[target_col].isnull(), target_col] = predictions

        return da