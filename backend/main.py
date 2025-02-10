from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import refactor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(refactor.router)

@app.get("/")
def home():
    return {"message": "CodeAether Backend Running!"}
