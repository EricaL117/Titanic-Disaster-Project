import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def data_path(*parts):
    here = os.path.dirname(__file__)
    return os.path.join(here, "..", "data", *parts)

def load_csv(name):
    path = data_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found at {path}.")
    return pd.read_csv(path)

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]].copy()
    X["Sex"] = (X["Sex"] == "male").astype(int)
    X["Age"] = X["Age"].fillna(X["Age"].median())
    X["Fare"] = X["Fare"].fillna(X["Fare"].median())
    return X

def main():
    # Load training data
    train = load_csv("train.csv")
    print("[INFO] Train shape:", train.shape)

    # Define target (survivability) and features
    y = train["Survived"]
    X = prepare_features(train)

    # Train/validation split for accuracy check
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_tr, y_tr)
    va_pred = model.predict(X_va)
    va_acc = accuracy_score(y_va, va_pred)
    print(f"[RESULT] Accuracy on training split: {va_acc:.3f}")

    # Show how many survived vs. died in validation predictions
    va_summary = pd.Series(va_pred).value_counts().rename({0: "Predicted Died", 1: "Predicted Survived"})
    print("[RESULT] Train prediction summary:")
    print(va_summary.to_string())

    # Load test data and generate predictions
    test = load_csv("test.csv")
    print("[INFO] Test shape:", test.shape)
    Xt = prepare_features(test)
    test_pred = model.predict(Xt)

    # Display survival prediction counts for test set
    test_summary = pd.Series(test_pred).value_counts().rename({0: "Predicted Died", 1: "Predicted Survived"})
    print("[RESULT] Test prediction summary:")
    print(test_summary.to_string())

    # Save predictions to CSV
    output = pd.DataFrame({
        "PassengerId": test.get("PassengerId", range(1, len(test_pred) + 1)),
        "Survived": test_pred
    })
    out_path = data_path("prediction_Python.csv")
    output.to_csv(out_path, index=False)
    print("[INFO] Saved predictions to:", os.path.abspath(out_path))

if __name__ == "__main__":
    main()
