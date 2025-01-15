from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model, save_model, Sequential
import numpy as np
import os
from analyze import generate_test_data, evaluate_model, plot_classification_report, generate_confusion_matrix, generate_normalized_confusion_matrix, plot_roc_curve
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

def create_zip2(files, zip_path):
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for file in files:
            zf.write(file, os.path.basename(file))

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
        # Zapisanie pliku ZIP z modelem
        model_package_path = "temp_model_package.zip"
        with open(model_package_path, "wb") as buffer:
            shutil.copyfileobj(model_package.file, buffer)

        # Rozpakowanie modelu, skalera i mapowania etykiet
        with zipfile.ZipFile(model_package_path, 'r') as zf:
            temp_dir = "temp_extract"
            zf.extractall(temp_dir)

        model = load_model(os.path.join(temp_dir, "model.h5"))

        with open(os.path.join(temp_dir, "scaler.pkl"), "rb") as f:
            scaler = joblib.load(f)

        with open(os.path.join(temp_dir, "label_mapping.json"), "r", encoding='utf-8') as f:
            label_mapping = json.load(f)

        # Generowanie danych testowych
        classes = list(label_mapping.keys())
        data, labels = generate_test_data(classes, num_samples)

        # Skalowanie danych testowych
        data_scaled = scaler.transform(data)

        # Obliczenia metryk
        accuracy, report, conf_matrix, predicted_labels = evaluate_model(model, data_scaled, labels, label_mapping)

        # Generowanie wizualizacji
        matrix_file_path = generate_confusion_matrix(labels, predicted_labels, classes)
        normalized_matrix_file_path = generate_normalized_confusion_matrix(labels, predicted_labels, classes)
        report_file_path = plot_classification_report(report, classes)
        roc_curve_file_path = plot_roc_curve(labels, predicted_labels, classes)

        # Sprawdzenie, czy pliki istnieją
        if not os.path.exists(matrix_file_path):
            raise HTTPException(status_code=500, detail="Confusion matrix file not found")
        if not os.path.exists(normalized_matrix_file_path):
            raise HTTPException(status_code=500, detail="Normalized confusion matrix file not found")
        if not os.path.exists(report_file_path):
            raise HTTPException(status_code=500, detail="Classification report file not found")
        if not os.path.exists(roc_curve_file_path):
            raise HTTPException(status_code=500, detail="ROC curve file not found")

        # Tworzenie pliku ZIP
        zip_path = "analysis_results.zip"
        create_zip2([matrix_file_path, normalized_matrix_file_path, report_file_path, roc_curve_file_path], zip_path)

        # Zwrot pliku ZIP
        return FileResponse(path=zip_path, filename=zip_path, media_type='application/zip')

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))