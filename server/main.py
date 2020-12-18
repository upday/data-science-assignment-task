import fasttext
import json
import os
import uvicorn
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from config.config import Config
from model.preprocessing import tokenize, tokenize_url
from model.predict_model import predict

app = FastAPI()
Config.init_config()

class Article(BaseModel):
    url: Optional[str] = ''
    title: Optional[str] = ''
    text: Optional[str] = ''

@app.post("/predict_label")
async def predict_label(article: Article):
    label, prob = predict(model, article.url, article.title, article.text)
    json_compatible_item_data = jsonable_encoder({"label": label, "probability": prob})
    return JSONResponse(content=json_compatible_item_data)


def load_model():
    model_path = os.path.join(Config.get_value("model", "output", "path"), Config.get_value("model", "output", "name"))
    global model
    model = fasttext.load_model(model_path)


if __name__ == "__main__":
    global model
    model = None
    # load model
    load_model()

    host = Config.get_value("application", "host")
    port = Config.get_value("application", "port")
    print(f"Navigate the url: http://{host}:{port}/docs for Swagger docs")
    
    uvicorn.run(app, host=host, port=port)