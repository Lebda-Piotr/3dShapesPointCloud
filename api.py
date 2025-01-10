from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model, save_model, Sequential
import numpy as np
import os
from analyze import accuracy_score, classification_report, confusion_matrix
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
from features import feature_vector
import joblib
from plyfile import PlyData
import pickle
import json
from enum import Enum
from solids import generate_cone, generate_cube, generate_octahedron, generate_sphere, generate_square_pyramid, generate_tetrahedron, visualize_point_cloud
from save_solids import save_point_cloud
from classificator import generate_dataset, train_model

app = FastAPI()

class Shape(str, Enum):
    sphere = "sphere"
    cube = "cube"
    cone = "cone"
    tetrahedron = "tetrahedron"
    square_pyramid = "square_pyramid"
    octahedron = "octahedron"

def save_model_files(model, scaler, label_mapping, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "model.h5")
    save_model(model, model_path)

    scaler_path = os.path.join(output_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    label_mapping_path = os.path.join(output_dir, "label_mapping.json")
    with open(label_mapping_path, "w") as f:
        json.dump(label_mapping, f)

    return model_path, scaler_path, label_mapping_path

def create_zip(model_path, scaler_path, label_mapping_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(model_path, os.path.basename(model_path))
        zf.write(scaler_path, os.path.basename(scaler_path))
        zf.write(label_mapping_path, os.path.basename(label_mapping_path))

@app.post("/train/")
async def train_model_endpoint(
    num_samples: int = Query(100, ge=1, description="Number of samples to generate (default is 100)"),
    shapes: str = Query("", description="Comma-separated list of shapes to include in the dataset, available solids: sphere, cube, cone, tetrahedron, square_pyramid, octahedron. Default is all available shapes.")
):
    try:
        if not shapes:
            shape_list = [shape.value for shape in Shape]  
        else:
            shape_list = [shape.strip() for shape in shapes.split(',')]

            # Walidacja kształtów
            for shape in shape_list:
                if shape not in Shape.__members__:
                    raise HTTPException(status_code=400, detail=f"Invalid shape: {shape}")

        # Generowanie danych
        data, labels = generate_dataset(num_samples=num_samples, shapes=shape_list)

        # Trenowanie
        model, scaler, label_mapping = train_model(data, labels)

        # Zapisywanie modelu, skalera i mapowania 
        temp_dir = "temp_model_files"
        model_path, scaler_path, label_mapping_path = save_model_files(model, scaler, label_mapping, temp_dir)

        # Tworzenie pliku 
        zip_path = os.path.join(temp_dir, "model_package.zip")
        create_zip(model_path, scaler_path, label_mapping_path, zip_path)

        return FileResponse(zip_path, media_type='application/zip', filename="model_package.zip")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/")
def generate_point_cloud(
    shape: Shape,
    num_points: int = Query(1000, ge=1, description="Number of points to generate"),
    radius: float = Query(None, description="Radius (only used for sphere or cone)"),
    edge_length: float = Query(None, description="Edge length (only used for cube, tetrahedron, etc.)"),
    height: float = Query(None, description="Height (only used for cone or pyramid)"),
    filename: str = Query("output.ply", description="Filename to save the generated point cloud")
):
    try:
        # Generowanie odpowiedniej chmury 
        if shape == Shape.sphere:
            points = generate_sphere(num_points, radius=radius)
        elif shape == Shape.cube:
            points = generate_cube(num_points, edge_length=edge_length)
        elif shape == Shape.cone:
            points = generate_cone(num_points, radius=radius, height=height)
        elif shape == Shape.tetrahedron:
            points = generate_tetrahedron(num_points, edge_length=edge_length)
        elif shape == Shape.square_pyramid:
            points = generate_square_pyramid(num_points, base_length=edge_length, height=height)
        elif shape == Shape.octahedron:
            points = generate_octahedron(num_points, edge_length=edge_length)
        else:
            raise HTTPException(status_code=400, detail="Invalid shape")

        # Wyświetlanie chmury 
        visualize_point_cloud(points)

        # Zapis chmury punktów
        save_point_cloud(points, filename)

        # Sprawdzenie
        if not os.path.exists(filename):
            raise HTTPException(status_code=500, detail="File not found after saving")

        return FileResponse(path=filename, filename=filename, media_type='application/octet-stream')
    except Exception as e:
        print(f"Error during point cloud generation: {e}")
        raise HTTPException(status_code=500, detail="Point cloud generation failed")

@app.post("/classify/")
async def classify(
    point_cloud: UploadFile = File(...),
    model_package: UploadFile = File(...)
):
    try:
        # Zapisanie pliku ZIP 
        model_package_path = "temp_model_package.zip"
        with open(model_package_path, "wb") as buffer:
            shutil.copyfileobj(model_package.file, buffer)

        # Rozpakowanie pliku 
        with zipfile.ZipFile(model_package_path, 'r') as zf:
            temp_dir = "temp_extract"
            zf.extractall(temp_dir)

        model = load_model(os.path.join(temp_dir, "model.h5"))

        with open(os.path.join(temp_dir, "scaler.pkl"), "rb") as f:
            scaler = joblib.load(f)

        with open(os.path.join(temp_dir, "label_mapping.json"), "r", encoding='utf-8') as f:
            label_mapping = json.load(f)

        # Zapisanie pliku chmury punktów 
        point_cloud_path = "temp_point_cloud.ply"
        with open(point_cloud_path, "wb") as buffer:
            shutil.copyfileobj(point_cloud.file, buffer)

        # Wczytanie danych chmury punktów z pliku 
        plydata = PlyData.read(point_cloud_path)
        point_cloud_np = np.vstack([plydata['vertex'][axis] for axis in ['x', 'y', 'z']]).T

        # Generowanie wektora 
        features = np.array(feature_vector(point_cloud_np)).reshape(1, -1)

        # Skalowanie danych i klasyfikacja
        features_scaled = scaler.transform(features)
        predictions = model.predict(features_scaled)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        predicted_class = [k for k, v in label_mapping.items() if v == predicted_class_idx][0]

        return {"predictions": predicted_class}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/analyze/")
async def analyze(
    model_package: UploadFile = File(...),
    num_samples: int = Query(100, ge=1, description="Number of test samples to generate")
):
    try:
        # Zapisanie pliku ZIP 
        model_package_path = "temp_model_package.zip"
        with open(model_package_path, "wb") as buffer:
            shutil.copyfileobj(model_package.file, buffer)

        # Rozpakowanie pliku ZIP
        with zipfile.ZipFile(model_package_path, 'r') as zf:
            temp_dir = "temp_extract"
            zf.extractall(temp_dir)

        model = load_model(os.path.join(temp_dir, "model.h5"))

        with open(os.path.join(temp_dir, "scaler.pkl"), "rb") as f:
            scaler = joblib.load(f)

        with open(os.path.join(temp_dir, "label_mapping.json"), "r", encoding='utf-8') as f:
            label_mapping = json.load(f)

        # Pobranie listy kategorii
        classes = list(label_mapping.keys())
        num_classes = len(classes)

        # Generowanie danych
        samples_per_class = num_samples // num_classes
        data = []
        labels = []

        for _ in range(samples_per_class):
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

        data = np.array(data)
        labels = np.array(labels)

        # Skalowanie
        data_scaled = scaler.transform(data)

        # Predykcja 
        predictions = model.predict(data_scaled)
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_labels = [list(label_mapping.keys())[list(label_mapping.values()).index(cls)] for cls in predicted_classes]

        # Ocena 
        accuracy = accuracy_score(labels, predicted_labels)
        report = classification_report(labels, predicted_labels, target_names=classes)
        conf_matrix = confusion_matrix(labels, predicted_labels)

        # Wizualizacja macierzy konfuzji
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')

        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": "confusion_matrix.png"
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))