from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
import os
import pdfplumber
from sentence_transformers import SentenceTransformer, util
from typing import List
import uvicorn

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#  BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(resume_path):
    """Extract text from a PDF resume."""
    text = ""
    try:
        with pdfplumber.open(resume_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading {resume_path}: {e}")  
        return ""

    print(f"Extracted text from {resume_path}:\n{text[:500]}")  
    return text.strip()


def generate_feedback(resume_text, job_description):
    """Generate feedback based on missing skills/keywords."""
    job_keywords = set(job_description.lower().split())
    resume_keywords = set(resume_text.lower().split())
    missing_keywords = job_keywords - resume_keywords

    if missing_keywords:
        feedback = f"Your resume is missing the following important keywords: {', '.join(missing_keywords)}"
    else:
        feedback = "Your resume is well-matched with the job description."

    return feedback


def rank_resumes(resume_list, job_description):
    """Rank only the selected resumes based on similarity to the job description."""
    job_embedding = model.encode(job_description, convert_to_tensor=True)
    resume_scores = {}
    resume_feedback = {}

    for resume_path in resume_list:
        resume_text = extract_text_from_pdf(resume_path)
        if not resume_text:
            print(f"Skipping {resume_path} (No text extracted)")
            continue

        resume_embedding = model.encode(resume_text, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(job_embedding, resume_embedding)

        print(f"Resume: {os.path.basename(resume_path)} | Score: {similarity.item()}")  

        resume_scores[resume_path] = similarity.item()
        resume_feedback[resume_path] = generate_feedback(resume_text, job_description)

    ranked_resumes = sorted(resume_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_resumes, resume_feedback


@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serve the HTML page."""
    with open("index.html", "r") as file:
        return HTMLResponse(content=file.read())


@app.post("/upload/")
async def upload_files(job_description: str = Form(...), resumes: List[UploadFile] = File(...)):
    """Handle file uploads and return ranking results only for selected resumes."""
    saved_files = []
    for file in resumes:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())
        saved_files.append(file_path)

    print(f"Saved files: {saved_files}")  

    # Rank only the newly uploaded resumes
    ranked_results, feedback_results = rank_resumes(saved_files, job_description)

    if not ranked_results:
        return JSONResponse(content={"message": "No resumes analyzed. Please try again!"})

    response = [{
        "resume": os.path.basename(resume),
        "score": round(score*100, 2),
        "feedback": feedback_results[resume]
    } for resume, score in ranked_results]

    return JSONResponse(content=response)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
