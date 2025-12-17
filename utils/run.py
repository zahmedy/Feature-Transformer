import pandas as pd


def main():
    """Minimal demo that trains a classifier with the FeatureTransformer."""
    df = pd.read_csv("data/raw/customer_segments.csv")
    print("Loaded dataset with shape:", df.shape)

    # choose target column and feature columns
    target_col = "cluster_id"
    feature_cols = [col for col in df.columns if col != target_col]

    X = df[feature_cols]
    y = df[target_col]

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from src import FeatureTransformer, FeatureConfig

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # let the transformer infer numeric/categorical columns automatically
    config = FeatureConfig()
    transformer = FeatureTransformer(
        feature_config=config,
        task="classification",
        k_best=None,
        scale_numeric=True,
    )

    # Pipeline: preprocessing + classifier
    model = Pipeline(
        steps=[
            ("features", transformer),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(X_train, y_train)

    print("\nTrain score:", model.score(X_train, y_train))
    print("Test score:", model.score(X_test, y_test))

    feature_names = model.named_steps["features"].get_feature_names_out()
    print("\nNumber of transformed features:", len(feature_names))
    print("First 10 feature names:", feature_names[:10])


if __name__ == "__main__":
    main()
