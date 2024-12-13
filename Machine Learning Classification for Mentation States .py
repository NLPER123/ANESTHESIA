from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Example of extracting features for classification (using PSD)
def extract_features_and_labels(psd, mentation_labels):
    # Reshape PSD data for classification
    psd_features = np.mean(psd, axis=1)  # Example: average PSD over the channels
    features = np.reshape(psd_features, (-1, 1))  # Convert to feature array
    labels = mentation_labels  # Assume you have mentation labels
    return features, labels

# Example: Train a Random Forest Classifier
def train_mentation_classifier(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    
    # Predictions
    y_pred = classifier.predict(X_test)
    
    # Print evaluation metrics
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

# Assume `mentation_labels` are provided (e.g., 0 for No Mentation, 1 for Active, etc.)
features, labels = extract_features_and_labels(psd, mentation_labels)
train_mentation_classifier(features, labels)
