from fastapi import APIRouter, HTTPException
from models.models import TextClassificationInput
from classifier.classifier_factory import ClassifierFactory, categories

router = APIRouter()

@router.post("/class")
async def classify_text(item: TextClassificationInput):
    if not item.title:
        raise HTTPException(status_code=400, detail="Title must not be empty.")
    elif len(item.title) > 64:
        raise HTTPException(status_code=400, detail="Title exceeds 64 characters.")

    classifier = ClassifierFactory.create_classifier()
    response = classifier.classify(item.title)
    return response

@router.get("/")
async def read_root():
    return categories