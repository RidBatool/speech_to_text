########################################
# 🚀 1. Import Required Libraries
########################################

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

import os
import io
import whisper
import uuid
from typing import Annotated
from PIL import Image

os.makedirs("uploads", exist_ok=True)


########################################
# 🔐 2. Load Environment Variables & Initialize Whisper
########################################
load_dotenv()

#You can store it in a .env file later if needed
MODEL_SIZE= os.getenv("WHISPER_MODEL_SIZE", "base") #base, small, medium , large

#Initialize the Whisper model(once, at app start)
asr_model= whisper.load_model(MODEL_SIZE)


########################################
# 🌐 3. Initialize FastAPI App & Enable CORS
########################################

app=FastAPI()

#Allow all origin (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origin=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

########################################
# 📦 4. Define Input Endpoint for Audio Upload
########################################
@app.post("/transcribe-audio")
async def transcribe_audio(file: UploadFile = File(...)):
    #Validate file Type
    if file.content_type not in ["audio/mpeg", "audio/wav", "audio/x-wav", "audio/mp4", "audio/x-m4a"]:
        return {'error':'Unsupported audio format.'}
    
    #Save uploaded file temporarily
    temp_filename = f'uploads{uuid.uuid4()}.mp3'
    with open(temp_filename, "wb") as f:
        f.write(await file.read())

    #Transcribe using whisper
    result = asr_model.transcribe(temp_filename)

    #Cleanup(optional) : delete file  after transcription
    #os.remove(temp_filename)

    ##Retun Result
    return {"transcript": result["text"]}


########################################
# 🏠 5. Serve Static Frontend HTML (UI)
########################################

@app.get("/")
async def server_html():
    return FileResponse("speech_ui.html")