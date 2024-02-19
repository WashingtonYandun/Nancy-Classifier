from pydantic import BaseModel

class TextClassificationInput(BaseModel):
    title: str
