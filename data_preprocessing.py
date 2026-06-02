import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE


def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # 🔥 Convert to Binary (Normal vs Attack)
    df['Label'] = df['Label'].apply(lambda x: 'Normal' if x == 'BENIGN' else 'Attack')

    # Split features & label
    X = df.drop(columns=['Label'])
    y = df['Label']

    # Encode labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    print("\n📊 Before SMOTE:")
    print(pd.Series(y).value_counts())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 🔥 Fix imbalance
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print("\n📊 After SMOTE:")
    print(pd.Series(y_train).value_counts())

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    features = X.columns.tolist()

    return X_train, X_test, y_train, y_test, scaler, features