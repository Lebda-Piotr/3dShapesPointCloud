import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from solids import generate_sphere, generate_cube, generate_cone, generate_tetrahedron, generate_square_pyramid, generate_octahedron
from features import feature_vector
from sklearn.model_selection import cross_val_score

def load_model(model_path):
    return joblib.load(model_path)

def evaluate_model(model, X_test, y_test):
    """
    Ocenia model.
    """
    y_pred = model.predict(X_test)

    # Rezultaty
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return accuracy, report, conf_matrix

def generate_test_data(classes, num_samples=100):
    """
    Generuje dane testowa bazując na tych w modelu.
    """
    data = []
    labels = []

    for _ in range(num_samples):
        for cls in classes:
            if cls == "sphere":
                shape = generate_sphere()
            elif cls == "cube":
                shape = generate_cube()
            elif cls == "cone":
                shape = generate_cone()
            elif cls == "tetrahedron":
                shape = generate_tetrahedron()
            elif cls == "square_pyramid":
                shape = generate_square_pyramid()
            elif cls == "octahedron":
                shape = generate_octahedron()
            else:
                continue

            data.append(feature_vector(shape))
            labels.append(cls)

    return np.array(data), np.array(labels)

def cross_validation(model, X, y, cv=5):
    # Cross-validation
    scores = cross_val_score(model, X, y, cv=cv)
    print(f"Mean accuracy (CV={cv}): {np.mean(scores):.2f}")
    print(f"Standard deviation: {np.std(scores):.2f}")

def feature_importance_analysis(model, feature_names):
    """
    stare
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]

        print("Feature importances:")
        for i in sorted_indices:
            print(f"{feature_names[i]}: {importances[i]:.4f}")

        # Visualization of feature importances
        plt.bar(range(len(importances)), importances[sorted_indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in sorted_indices], rotation=45)
        plt.title('Feature Importances')
        plt.show()
    else:
        print("Model does not support feature importance analysis.")

# Przykład
if __name__ == "__main__":
    # Path to the model
    model_path = "shape_classifier_rf.pkl"
    
    # Load the model
    model = load_model(model_path)

    # Test data (example - should match the actual data format)
    X_test = np.random.rand(20, 6)  # 20 examples, 6 features
    y_test = np.random.choice(["sphere", "cube", "cone", "square_pyramid", "octahedron", "tetrahedron"], 20)

    # Evaluate the model
    accuracy, report, conf_matrix = evaluate_model(model, X_test, y_test)
    print("Accuracy:", accuracy)
    print("Classification Report:", report)
    print("Confusion Matrix:", conf_matrix)

    # Cross-validation
    X = np.random.rand(100, 6)  # 100 examples, 6 features
    y = np.random.choice(["sphere", "cube", "cone", "square_pyramid", "octahedron", "tetrahedron"], 100)
    cross_validation(model, X, y, cv=5)

    # Feature importance analysis
    feature_names = ["Mean Distance", "Std Distance", "BBox X", "BBox Y", "BBox Z", "Convex Hull Volume"]
    feature_importance_analysis(model, feature_names)