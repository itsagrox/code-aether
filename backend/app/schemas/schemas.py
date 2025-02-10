from pydantic import BaseModel

class CodeRefactorRequest(BaseModel):
    originalCode: str
    language: str
    modelType: str

class CodeRefactorResponse(BaseModel):
    originalCode: str
    refactoredCode: str
    message: str