from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import pandas as pd
from io import StringIO
from openai import OpenAI
from fastapi.responses import StreamingResponse
from datetime import datetime
from tqdm import tqdm

load_dotenv()

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

os.makedirs('dist', exist_ok=True)


class QuestionResponse(BaseModel):
    question: str
    response: str


@app.post("/score/")
async def score_answer(qr: QuestionResponse):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            max_tokens=10,
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a response evaluator."},
                {"role": "user",
                 "content":
                     f"Score the following answer to the question on a scale from 0 to 1:\n\nQuestion: {qr.question}\nAnswer: {qr.response}"}
            ]
        )

        # Extract and print response
        score = float(response.choices[0].message.content.strip())
        print(score)

        if 0 <= score <= 1:
            return {"score": score}
        else:
            raise HTTPException(status_code=400, detail="Score out of range")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-csv/")
async def process_csv(file: UploadFile = File(...), num_asks: int = 1):
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))

        if 'query' not in df.columns or 'response' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'query' and 'response' columns")

        score_lists = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            question = row['query']
            response = row['response']

            scores = []
            for _ in range(num_asks):
                try:
                    response_gpt = client.chat.completions.create(
                        model="gpt-4o-mini",
                        max_tokens=10,
                        temperature=0,
                        messages=[
                            {"role": "system", "content": "You are a response evaluator."},
                            {"role": "user",
                             "content":
                                 f"Score the following answer to the question on a scale from 0 to 1, 0 should be for completely incorrect response and 1 for completely correct, do not explain the score, just return the socre:\n\nQuestion: {question}\nAnswer: {response}"}
                        ]
                    )

                    score = float(response_gpt.choices[0].message.content.strip())
                    if 0 <= score <= 1:
                        scores.append(score)
                except Exception as e:
                    print(f"exception occurred, {e}")
                    continue

            score_lists.append(scores)

        df['scores'] = score_lists

        output = StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        file_path = os.path.join('dist', f"scored_{file.filename}_{datetime.now().isoformat()}")
        with open(file_path, 'w') as f:
            f.write(output.getvalue())

        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=scored_{file.filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
