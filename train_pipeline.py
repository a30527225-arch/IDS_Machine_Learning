import os
import sys
import joblib

from utils.data_preprocessing import load_and_preprocess
from utils.train_model import train_and_evaluate

if __name__ == "__main__":
    print("=" * 60)
    print("🛡️ IDS Model Training Pipeline")
    print("=" * 60)

    dataset_path = "data/CICIDS2017_sample_kme.csv"

    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        sys.exit(1)

    print("✅ Preprocessing dataset...")

    X_train, X_test, y_train, y_test, scaler, features = load_and_preprocess(dataset_path)

    print(f"✅ Total features: {len(features)}")

    print("\n🤖 Training model...")
    results, model = train_and_evaluate(X_train, y_train, X_test, y_test)

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(features, "models/features.joblib")

    print("\n✅ Training complete!")