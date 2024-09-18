from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from pydantic import BaseModel, Field, Extra
from haystack.schema import Answer, Document

class CustomConfig:
    arbitrary_types_allowed = True
    json_encoders = {
        # if any of the documents contains an embedding field as an ndarray the latter needs to be converted to list of float
        np.ndarray: lambda x: x.tolist(),
        pd.DataFrame: lambda x: [x.columns.tolist()] + x.values.tolist(),
    }

class RequestBaseModel(BaseModel):
    class Config(CustomConfig):
        # Forbid any extra fields in the request to avoid silent failures
        extra = Extra.forbid

class QueryRequest(RequestBaseModel):
    query: str
    params: Optional[dict] = None
    debug: Optional[bool] = False

class QueryResponse(BaseModel):
    query: str
    answers: List[Answer] = []
    documents: List[Document] = []
    results: Optional[List[str]] = None
    debug: Optional[Dict] = Field(None, alias="_debug")
    
    class Config(CustomConfig):
        pass