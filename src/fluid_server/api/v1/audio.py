"""
OpenAI-compatible audio transcription endpoint
"""
import logging
from fastapi import APIRouter, HTTPException, File, Form, UploadFile
from typing import Optional
from ...models.openai import TranscriptionResponse
from ...managers.runtime_manager import RuntimeManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/audio")


@router.post("/transcriptions", response_model=TranscriptionResponse)
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0)
) -> TranscriptionResponse:
    """
    OpenAI-compatible audio transcription endpoint
    
    Args:
        file: Audio file to transcribe
        model: Model to use (whisper-1 compatible)
        language: Optional language hint
        prompt: Optional prompt to guide transcription
        response_format: Response format (json, text, srt, vtt)
        temperature: Sampling temperature
        runtime_manager: Runtime manager instance (injected)
        
    Returns:
        Transcription response
    """
    try:
        # Read audio data
        audio_data = await file.read()
        
        # TODO: Get Whisper runtime via dependency injection
        # For now, return placeholder response
        return TranscriptionResponse(
            text="Transcription not yet implemented",
            language=language or "en"
        )
        
    except ValueError as e:
        logger.error(f"Invalid transcription request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))