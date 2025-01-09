# Użyj oficjalnego obrazu Pythona jako bazowego
FROM python:3.8.9

# Zainstaluj brakujące zależności
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Ustaw katalog roboczy w kontenerze
WORKDIR /app

# Skopiuj pliki projektu do katalogu roboczego
COPY . /app

# Zaktualizuj pip i zainstaluj zależności
RUN pip install --upgrade pip && pip install --no-cache-dir --upgrade -r requirements.txt

# Uruchom aplikację
CMD ["uvicorn", "api:app", "--reload", "--host", "0.0.0.0", "--port", "8080"]