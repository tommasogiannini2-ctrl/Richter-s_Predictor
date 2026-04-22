import pandas as pd


class DataEncoding:
    """Gestisce l'encoding delle variabili categoriche."""

    def __init__(self, dataframe: pd.DataFrame, is_train=True):
        self.df = dataframe.copy()
        self.is_train = is_train

    def trasforma(self, lista_colonne: list = None) -> pd.DataFrame:
        """Esegue l'encoding e allinea le colonne al train set (solo test)."""
        self.dummy()

        if not self.is_train and lista_colonne:
            colonne_target = [c for c in lista_colonne if c != 'damage_grade']
            self.df = self.df.reindex(columns=colonne_target, fill_value=0)

        return self.df

    def dummy(self):
        """Trasforma le feature categoriche in dummy variables (One-Hot Encoding)."""
        feature_categoriche = [
            'land_surface_condition', 'foundation_type', 'roof_type',
            'ground_floor_type', 'other_floor_type', 'position',
            'plan_configuration', 'legal_ownership_status'
        ]

        colonne_presenti = [c for c in feature_categoriche if c in self.df.columns]

        colonne_prima = set(self.df.columns)
        self.df = pd.get_dummies(
            self.df,
            columns=colonne_presenti,
            drop_first=False,
            dtype=int
        )
        colonne_dopo = set(self.df.columns)
        nuove_colonne = colonne_dopo - colonne_prima

        print(f"  {'Feature categoriche codificate:':<40} {len(colonne_presenti):>8}")
        print(f"  {'Nuove colonne dummy aggiunte:':<40} {len(nuove_colonne):>8}")