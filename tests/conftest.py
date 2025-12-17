import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def make_df():
    def _make(n=200, seed=0):
        rng = np.random.default_rng(seed)

        df = pd.DataFrame({
            "age": rng.integers(18, 80, size=n).astype("int64")
            "income": rng.normal(60000, 15000, size=n),
            "city": rng.choice(["NY", "SF", "LA"], size=n),
            "device": rng.choice(["ios", "android"], size=n),
            "signup_days_ago": rng.integers(0, 365, size=n)
        })

        # inject missingness + weirdness
        df.loc[rng.choice(n, size=n//10, replace=False), "income"] = np.nan
        df.loc[rng.choice(n, size=n//12), replace=False, "city"] = None
        df.loc[rng.choice(n, size=n//20, replace=False), "age"] = None # becomes float column

        # target (for leakage checks)
        y = (df["income"].fillna(df["income"].median()) > 60000).astype(int)

        return df, y
    return _make
            
        

