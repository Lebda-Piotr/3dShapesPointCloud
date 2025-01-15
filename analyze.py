import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from solids import generate_sphere, generate_cube, generate_cone, generate_tetrahedron, generate_square_pyramid, generate_octahedron
from features import feature_vector
from sklearn.model_selection import cross_val_score

def load_model(model_path):
    return joblib.load(model_path)

def evaluate_model(model, X_test, y_test, label_mapping):
    """
    Ocenia model.
    """
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_labels = [list(label_mapping.keys())[list(label_mapping.values()).index(cls)] for cls in predicted_classes]

    # Rezultaty
    accuracy = accuracy_score(y_test, predicted_labels)
    report = classification_report(y_test, predicted_labels, output_dict=True)
    conf_matrix = confusion_matrix(y_test, predicted_labels)

    return accuracy, report, conf_matrix, predicted_labels

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

def generate_confusion_matrix(y_true, y_pred, classes):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    matrix_file_path = "confusion_matrix.png"
    plt.savefig(matrix_file_path)
    plt.close()
    return matrix_file_path

def plot_classification_report(report, classes):
    report_df = pd.DataFrame(report).transpose()
    plt.figure(figsize=(10, 7))
    sns.heatmap(report_df.iloc[:-1, :].T, annot=True, cmap='Blues')
    plt.title('Classification Report')
    report_file_path = "classification_report.png"
    plt.savefig(report_file_path)
    plt.close()
    return report_file_path

def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 7))
        plt.bar(range(len(importances)), importances[sorted_indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in sorted_indices], rotation=45)
        plt.title('Feature Importances')
        feature_importance_file_path = "feature_importances.png"
        plt.savefig(feature_importance_file_path)
        plt.close()
        return feature_importance_file_path
    else:
        print("Model does not support feature importance analysis.")
        return None

def plot_roc_curve(y_true, y_pred, classes):
    y_true_bin = label_binarize(y_true, classes=classes)
    y_pred_bin = label_binarize(y_pred, classes=classes)
    n_classes = len(classes)

    plt.figure(figsize=(10, 7))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f}) for class {classes[i]}')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    roc_curve_file_path = "roc_curve.png"
    plt.savefig(roc_curve_file_path)
    plt.close()
    return roc_curve_file_path

def plot_loss_curve(history):
    plt.figure(figsize=(10, 7))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    loss_curve_file_path = "loss_curve.png"
    plt.savefig(loss_curve_file_path)
    plt.close()
    return loss_curve_file_path

def plot_accuracy_curve(history):
    plt.figure(figsize=(10, 7))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    accuracy_curve_file_path = "accuracy_curve.png"
    plt.savefig(accuracy_curve_file_path)
    plt.close()
    return accuracy_curve_file_path

def generate_normalized_confusion_matrix(y_true, y_pred, classes):
    conf_matrix = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    normalized_matrix_file_path = "normalized_confusion_matrix.png"
    plt.savefig(normalized_matrix_file_path)
    plt.close()
    return normalized_matrix_file_path