import pandas as pd 


if __name__ == "__main__":
    # 1. load the data from CSV file 
    df = pd.read_csv("data/raw/customer_segments.csv")

    # 2. Peek at the first rows
    print(f"First rows: \n{df.head()}")

    # 3. check column types
    print(df.dtypes)

    # 4. choose target column and features columns
    target_col = "cluster_id"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols]
    y = df[target_col]

    print("\nFeature columns:", feature_cols)
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from feature_transformer.transformer import FeatureTransformer, FeatureConfig

    # 5) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6) Create feature config (let numeric features be auto-inferred)
    config = FeatureConfig(
        numeric_features=None,        # let transformer detect numeric columns
        categorical_features=None,    # no categorical columns here, but we keep it general
    )

    ft = FeatureTransformer(
        config,
        task="classification",
        k_best=None,
        scale_numeric=True
    )

    # 7) Pipeline: preprocessing + model 
    model = Pipeline(
        steps=[
            ("features", ft),
            ("clf", LogisticRegression(max_iter=1000))
        ]
    )

    # 8) Fit and evaluate
    model.fit(X_train, y_train)
    print("\nTrain score:", model.score(X_train, y_train))
    print("Test score:", model.score(X_test, y_test))

    # 9) Inspect final feature names
    feature_names = model.named_steps["features"].get_feature_names_out()
    print("\nNumber of transformed features:", len(feature_names))
    print("First 10 feature names:", feature_names[:10])