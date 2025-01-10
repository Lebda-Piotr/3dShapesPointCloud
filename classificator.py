import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from solids import generate_sphere, generate_cube, generate_cone, generate_octahedron, generate_tetrahedron, generate_square_pyramid
from features import feature_vector

# Funkcja do generowania danych
def generate_dataset(num_samples=400, shapes=["sphere", "cube", "cone", "square_pyramid", "octahedron", "tetrahedron"]):
    """
    Generuje zestaw danych z chmur punktów dla różnych kształtów.
    Zwraca tablice feature_vectors i labels.
    """
    data = []
    labels = []

    for _ in range(num_samples):
        if "sphere" in shapes:
            sphere = generate_sphere()
            data.append(feature_vector(sphere))
            labels.append("sphere")

        if "cube" in shapes:
            cube = generate_cube()
            data.append(feature_vector(cube))
            labels.append("cube")

        if "cone" in shapes:
            cone = generate_cone()
            data.append(feature_vector(cone))
            labels.append("cone")

        if "square_pyramid" in shapes:
            square_pyramid = generate_square_pyramid()
            data.append(feature_vector(square_pyramid))
            labels.append("square_pyramid")

        if "octahedron" in shapes:
            octahedron = generate_octahedron()
            data.append(feature_vector(octahedron))
            labels.append("octahedron")

        if "tetrahedron" in shapes:
            tetrahedron = generate_tetrahedron()
            data.append(feature_vector(tetrahedron))
            labels.append("tetrahedron")

    return np.array(data), np.array(labels)

# Funkcja do trenowania modelu
def train_model(data, labels):
    # Skalowanie cech
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Kodowanie etykiet
    label_mapping = {label: idx for idx, label in enumerate(np.unique(labels))}
    labels = np.array([label_mapping[label] for label in labels])
    labels = to_categorical(labels)

    # Cross-walidacja
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for train, test in kfold.split(data, labels):
        # Inicjalizacja sieci neuronowej
        model = Sequential()
        model.add(Dense(64, input_dim=data.shape[1], activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(len(label_mapping), activation='softmax'))

        # Kompilacja modelu
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Trenowanie modelu
        model.fit(data[train], labels[train], epochs=20, batch_size=10, validation_data=(data[test], labels[test]), verbose=0)

        # Ocena modelu
        scores = model.evaluate(data[test], labels[test], verbose=0)
        cv_scores.append(scores[1] * 100)

    print(f"Cross-validation accuracy: {np.mean(cv_scores):.2f}% (+/- {np.std(cv_scores):.2f}%)")

    # Trenowanie finalnego modelu na całym zbiorze danych
    model.fit(data, labels, epochs=20, batch_size=10, validation_split=0.25)

    # Podział danych na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

    # Ocena modelu
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    report = classification_report(y_test_classes, y_pred_classes, target_names=label_mapping.keys())

    # Zapisywanie modelu, skalera i mapowania etykiet
    model_filename = "shape_classifier_nn.h5"
    scaler_filename = "scaler.pkl"
    label_mapping_filename = "label_mapping.pkl"
    model.save(model_filename)
    joblib.dump(scaler, scaler_filename)
    joblib.dump(label_mapping, label_mapping_filename)
    print(f"Model saved as {model_filename}")
    print(f"Scaler saved as {scaler_filename}")
    print(f"Label mapping saved as {label_mapping_filename}")

    return model, scaler, label_mapping

# Funkcja do klasyfikacji nowej chmury punktów
def classify_new_point_cloud(points, model_path="shape_classifier_nn.h5", scaler_path="scaler.pkl", label_mapping_path="label_mapping.pkl"):
    """
    Klasyfikuje nową chmurę punktów, używając zapisanego modelu.
    """
    # Wczytanie modelu, skalera i mapowania etykiet
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    label_mapping = joblib.load(label_mapping_path)
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}

    # Przetwarzanie cech
    features = np.array(feature_vector(points)).reshape(1, -1)
    features = scaler.transform(features)

    # Predykcja klasy
    prediction = model.predict(features)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    predicted_class = reverse_label_mapping[predicted_class_idx]

    return predicted_class

# Przykładowa klasyfikacja nowej chmury punktów
if __name__ == "__main__":
    accuracy, report, model_filename = train_model()
    print("Accuracy:", accuracy)
    print("Raport:", report)

    new_shape = generate_tetrahedron()
    prediction = classify_new_point_cloud(new_shape)
    print("Prediction for new point cloud: ", prediction)