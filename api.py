from fastapi import FastAPI, File, UploadFile, HTTPException
import joblib
import numpy as np
from features import feature_vector
from plyfile import PlyData
import io

app = FastAPI()

# Wczytanie wcześniej wytrenowanego modelu
model_path = "shape_classifier_rf.pkl"
try:
    classifier = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    raise HTTPException(status_code=500, detail="Model loading failed")

def read_ply(file_contents):
    try:
        # Próbujemy odczytać plik PLY jako binarny
        plydata = PlyData.read(io.BytesIO(file_contents))
    except Exception:
        try:
            plydata = PlyData.read(io.StringIO(file_contents.decode('utf-8')))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read PLY file: {e}")
    return plydata

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

# e:; cd 'e:\\Nowy folder (2)\\bryly'; & 'e:\\Nowy folder (2)\\python.exe' -m uvicorn api:app --reload
