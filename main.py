from fastapi import FastAPI, Request
from pydantic import BaseModel

from model import get_prediction 

app = FastAPI()

class TextParameter(BaseModel):
    text: str


@app.post('/predict') # revisar en postman
def predict(text_parameter: TextParameter):
    text = text_parameter.text
    sentimiento, sentimiento_proba = get_prediction(text)
    return { 
        "sentimiento": sentimiento,
        "proba_negativo": sentimiento_proba[0],
        "proba_neutro": sentimiento_proba[1],
        "proba_positivo": sentimiento_proba[2] 
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app="main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

