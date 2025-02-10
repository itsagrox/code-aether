from fastapi import APIRouter, HTTPException
from app.schemas.schemas import CodeRefactorRequest, CodeRefactorResponse
from app.services.ai_service import call_huggingface_api

router = APIRouter()

@router.post("/refactor", response_model=CodeRefactorResponse)
async def refactor_code(request: CodeRefactorRequest):
    if not request.originalCode.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty")

    # Call the local model to get refactored code
    try:
        refactored_code = call_huggingface_api(request.originalCode, request.language, request.modelType)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return CodeRefactorResponse(
        originalCode=request.originalCode, 
        refactoredCode=refactored_code,     
        message="Refactored code generated successfully"
    )
