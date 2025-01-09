import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib  
from solids import generate_sphere, generate_cube, generate_cone, generate_octahedron, generate_tetrahedron, generate_square_pyramid
from features import feature_vector

# Funkcja do generowania danych
def generate_dataset(num_samples=100, shapes=["sphere", "cube", "cone", "square_pyramid", "octahedron", "tetrahedron"]):
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
def train_model(num_samples=400, shapes=["sphere", "cube", "cone", "square_pyramid", "octahedron", "tetrahedron"]):
    # Generowanie danych
    data, labels = generate_dataset(num_samples=num_samples, shapes=shapes)

    # Podział danych na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Inicjalizacja i trenowanie klasyfikatora Random Forest
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Predykcja na zbiorze testowym
    y_pred = rf_classifier.predict(X_test)

    # Wyniki
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Zapisywanie modelu
    model_filename = "shape_classifier_rf.pkl"
    joblib.dump(rf_classifier, model_filename)
    print(f"Model zapisany jako {model_filename}")

    return accuracy, report, model_filename

# Funkcja do klasyfikacji nowej chmury punktów
def classify_new_point_cloud(points, model_path="shape_classifier_rf.pkl"):
    """
    Klasyfikuje nową chmurę punktów, używając zapisanego modelu.
    """
    # Wczytanie modelu
    classifier = joblib.load(model_path)
    features = np.array(feature_vector(points)).reshape(1, -1)
    return classifier.predict(features)[0]

# Przykładowa klasyfikacja nowej chmury punktów
if __name__ == "__main__":
    new_shape = generate_tetrahedron()
    prediction = classify_new_point_cloud(new_shape)
    print("Predykcja dla nowej chmury punktów: ", prediction)