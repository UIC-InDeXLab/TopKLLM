import concurrent
import csv
import io
import logging
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator
from tqdm import tqdm

from models import get_available_models, get_model_llm

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    model: str

    @field_validator("model")
    def validate_model(cls, model: str) -> str:
        if model not in get_available_models():
            raise ValueError("Model not available")
        return model


class BatchQueryRequest(BaseModel):
    queries: List[str]
    model: str

    @field_validator("model")
    def validate_model(cls, model: str) -> str:
        if model not in get_available_models():
            raise ValueError("Model not available")
        return model


@app.post("/v1/query/")
def query_model(request: QueryRequest):
    model_llm = get_model_llm(request.model)
    with model_llm() as llm:
        res = llm.query(request.query, model=request.model)

    return {"response": res.response}


@app.post("/v1/batch_query/")
async def batch_query(request: BatchQueryRequest):
    model_llm = get_model_llm(request.model)
    responses = []

    with model_llm() as llm:
        for query in request.queries:
            res = llm.query(query, model=request.model)
            responses.append((query, res.response))

    csv_filename = f"{request.model}.csv"
    with open(csv_filename, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["query", "response"])
        csv_writer.writerows(responses)

    return FileResponse(csv_filename, media_type='text/csv', filename=csv_filename)


@app.post("/v1/upload_query_file/")
async def upload_query_file(model: str, file: UploadFile = File(...)):
    if model not in get_available_models():
        raise HTTPException(status_code=400, detail="Model not available")

    # Read CSV file
    contents = await file.read()
    file_stream = io.StringIO(contents.decode('utf-8'))
    csv_reader = csv.reader(file_stream)
    queries = list(csv_reader)

    if not queries:
        raise HTTPException(status_code=400, detail="No queries found in file")

    # Process queries
    model_llm = get_model_llm(model)
    responses = []

    def process_query(llm, query):
        try:
            query_text = query[0]  # Assuming the CSV has one column with queries
            res = llm.query(query_text, model=model)
            return query_text, res.response
        except Exception as e:
            return (query_text, f"Error: {str(e)}")

    with model_llm() as llm:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_query = {executor.submit(process_query, llm, query): query for query in queries}
            for future in tqdm(concurrent.futures.as_completed(future_to_query), total=len(future_to_query),
                               desc="Processing queries"):
                query = future_to_query[future]
                try:
                    result = future.result()
                    responses.append(result)
                except Exception as e:
                    responses.append((query, f"Error: {str(e)}"))

    # Save responses to CSV file
    csv_filename = f"{model}.csv"
    with open(csv_filename, mode="w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["query", "response"])
        csv_writer.writerows(responses)

    return FileResponse(csv_filename, media_type='text/csv', filename=csv_filename)


@app.get("/v1/models/")
def get_models():
    return {"models": get_available_models()}
