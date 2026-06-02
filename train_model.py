from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_and_evaluate(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        class_weight='balanced',
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\n📊 Accuracy:", acc)

    print("\n📄 Classification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Attack"]
    ))

    return {"accuracy": acc}, model