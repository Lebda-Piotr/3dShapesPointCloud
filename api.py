from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends
from enum import Enum
import joblib
import numpy as np
from features import feature_vector
from plyfile import PlyData
import io
from solids import generate_sphere, generate_cube, generate_cone, generate_tetrahedron, generate_square_pyramid, generate_octahedron, visualize_point_cloud
import open3d as o3d
from fastapi.responses import FileResponse
import os
from typing import List
from classificator import train_model

app = FastAPI()

# Wczytanie wcześniej wytrenowanego modelu
model_path = "shape_classifier_rf.pkl"
try:
    classifier = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    raise HTTPException(status_code=500, detail="Model loading failed")

class Shape(str, Enum):
    sphere = "sphere"
    cube = "cube"
    cone = "cone"
    tetrahedron = "tetrahedron"
    square_pyramid = "square_pyramid"
    octahedron = "octahedron"

def read_ply(file_contents):
    try:
        plydata = PlyData.read(io.BytesIO(file_contents))
    except Exception:
        try:
            plydata = PlyData.read(io.StringIO(file_contents.decode('utf-8')))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read PLY file: {e}")
    return plydata

def save_point_cloud(points, filename):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, cloud)

@app.post("/classify/")
async def classify_point_cloud(file: UploadFile = File(...)):
    try:
        # Odczytanie przesłanego pliku
        contents = await file.read()
        
        # Wczytanie danych z pliku PLY
        plydata = read_ply(contents)
        points = np.vstack([plydata['vertex'][axis] for axis in ['x', 'y', 'z']]).T
        
        # Ekstrakcja cech z chmury punktów
        features = np.array(feature_vector(points)).reshape(1, -1)
        
        # Predykcja klasy chmury punktów
        prediction = classifier.predict(features)[0]
        
        return {"prediction": prediction}
    except Exception as e:
        print(f"Error during classification: {e}")
        raise HTTPException(status_code=500, detail="Classification failed")

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
        # Generowanie odpowiedniej chmury punktów
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

        # Wyświetlanie chmury punktów
        visualize_point_cloud(points)

        # Zapis chmury punktów do pliku
        save_point_cloud(points, filename)

        # Sprawdzenie, czy plik istnieje
        if not os.path.exists(filename):
            raise HTTPException(status_code=500, detail="File not found after saving")

        return FileResponse(path=filename, filename=filename, media_type='application/octet-stream')
    except Exception as e:
        print(f"Error during point cloud generation: {e}")
        raise HTTPException(status_code=500, detail="Point cloud generation failed")

@app.post("/train/")
def train_model_endpoint(
    num_samples: int = Query(100, ge=1, description="Number of samples to generate (default is 100)"),
    shapes: str = Query("", description="Comma-separated list of shapes to include in the dataset, available solids: sphere, cube, cone, tetrahedron, square_pyramid, octahedron. Default is all available shapes.")
):
    try:
        # Jeśli shapes jest puste, używamy domyślnej listy wszystkich kształtów
        if not shapes:
            shape_list = [shape.value for shape in Shape]  # Użycie wszystkich dostępnych kształtów
        else:
            # Podzielenie ciągu znaków na kształty
            shape_list = [shape.strip() for shape in shapes.split(',')]

            # Walidacja poprawności kształtów
            for shape in shape_list:
                if shape not in Shape.__members__:
                    raise HTTPException(status_code=400, detail=f"Invalid shape: {shape}")

        # Wywołanie funkcji train_model z pliku classificator.py
        accuracy, report, model_filename = train_model(num_samples=num_samples, shapes=shape_list)

        return {
            "message": f"Model trained and saved as {model_filename}",
            "accuracy": accuracy,
            "classification_report": report,
            "model_file": model_filename
        }
    except Exception as e:
        print(f"Error during model training: {e}")
        raise HTTPException(status_code=500, detail="Model training failed")
# e:; cd 'e:\\Nowy folder (2)\\bryly'; & 'e:\\Nowy folder (2)\\python.exe' -m uvicorn api:app --reload
