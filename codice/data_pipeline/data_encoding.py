import pandas as pd


class DataEncoding:
    """Gestisce esclusivamente la trasformazione delle variabili categoriche."""

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

    def dummy(self):
        """Trasforma le feature categoriche in dummy variables."""
        feature_categoriche = [
            'land_surface_condition', 'foundation_type', 'roof_type',
            'ground_floor_type', 'other_floor_type', 'position',
            'plan_configuration', 'legal_ownership_status'
        ]

        self.df = pd.get_dummies(
            self.df,
            columns=feature_categoriche,
            drop_first=False,
            dtype=int
        )

        nuove_colonne = [
            col for col in self.df.columns
            if any(feat in col for feat in feature_categoriche) and col not in feature_categoriche
        ]
        print(f"Aggiunte {len(nuove_colonne)} nuove colonne dummy.")

        return self.df