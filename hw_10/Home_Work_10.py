from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional
from sklearn.linear_model import LinearRegression
import pandas as pd
import joblib
import os
import logging
import uvicorn
from contextlib import asynccontextmanager

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальная переменная для модели
ml_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    global ml_model
    try:
        DATA_PATH = os.path.join("data", "realty_data.csv")
        MODEL_FILE = "realty_model.joblib"

        if os.path.exists(DATA_PATH):
            if os.path.exists(MODEL_FILE):
                ml_model = joblib.load(MODEL_FILE)
                logger.info("Модель загружена из файла")
            else:
                logger.info("Обучение новой модели...")
                data = pd.read_csv(DATA_PATH)
                data_clean = data.dropna(subset=['total_square', 'rooms', 'price'])
                X = data_clean[['total_square', 'rooms']]
                y = data_clean['price']

                ml_model = LinearRegression()
                ml_model.fit(X, y)
                joblib.dump(ml_model, MODEL_FILE)
                logger.info("Модель обучена и сохранена")
        else:
            logger.error(f"Файл данных не найден по пути: {DATA_PATH}")
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
    yield
    ml_model = None


app = FastAPI(
    title="API прогнозирования стоимости недвижимости",
    version="1.0.0",
    lifespan=lifespan
)


class PropertyFeatures(BaseModel):
    total_square: float = Field(..., gt=0, description="Общая площадь в кв.м")
    rooms: int = Field(..., gt=0, description="Количество комнат")


class PredictionResponse(BaseModel):
    predicted_price: float
    model_type: str


@app.get("/health")
async def health_check():
    return {
        "status": "OK",
        "model_loaded": ml_model is not None,
        "data_available": os.path.exists(os.path.join("data", "realty_data.csv"))
    }


@app.get("/predict_get", response_model=PredictionResponse)
async def predict_get(
        total_square: float = Query(..., gt=0, description="Общая площадь в кв.м"),
        rooms: int = Query(..., gt=0, description="Количество комнат")
):
    """Прогноз через GET-запрос"""
    if ml_model is None:
        raise HTTPException(503, detail="Модель не загружена")
    try:
        prediction = ml_model.predict([[total_square, rooms]])[0]
        return PredictionResponse(
            predicted_price=round(prediction, 2),
            model_type="LinearRegression"
        )
    except Exception as e:
        raise HTTPException(400, detail=str(e))


@app.post("/predict_post", response_model=PredictionResponse)
async def predict_post(data: PropertyFeatures):
    """Прогноз через POST-запрос"""
    if ml_model is None:
        raise HTTPException(503, detail="Модель не загружена")
    try:
        prediction = ml_model.predict([[data.total_square, data.rooms]])[0]
        return PredictionResponse(
            predicted_price=round(prediction, 2),
            model_type="LinearRegression"
        )
    except Exception as e:
        raise HTTPException(400, detail=str(e))


def start_app():
    uvicorn.run(
        "Home_Work_10:app",
        host="127.0.0.1",
        port=8080,
        reload=True
    )


if __name__ == "__main__":
    start_app()