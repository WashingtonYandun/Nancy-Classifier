from fastapi import FastAPI
from routers import router
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import uvicorn
import numpy as np
from os import environ


app = FastAPI()


# Aquí puedes agregar configuraciones adicionales, como CORS
app.include_router(router)


if __name__ == "__main__":
    port = int(environ.get("PORT", 8000))
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)