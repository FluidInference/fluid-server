"""
OpenAI-compatible audio transcription endpoint
"""

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from ...managers.runtime_manager import RuntimeManager
from ...models.openai import TranscriptionResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/audio")


def get_runtime_manager(request: Request) -> RuntimeManager:
    """Dependency to get runtime manager"""
    return request.app.state.runtime_manager


@router.post("/transcriptions", response_model=TranscriptionResponse)
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str | None = Form("json"),
    temperature: float | None = Form(0),
    runtime_manager: RuntimeManager = Depends(get_runtime_manager),
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
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Read audio data
        audio_data = await file.read()
        logger.info(f"Processing audio file: {file.filename}, size: {len(audio_data)} bytes")

        # Get Whisper runtime with error recovery
        logger.info("Getting Whisper runtime...")
        try:
            whisper_runtime = await runtime_manager.get_whisper()
        except RuntimeError as e:
            logger.warning(f"Whisper model not available: {e}")
            raise HTTPException(
                status_code=503,
                detail="Transcription service is temporarily unavailable. Please try again later.",
            ) from e

        # Perform transcription
        logger.info(f"Starting transcription with model: {model}, language: {language}")
        result = await whisper_runtime.transcribe(
            audio_data, language=language, return_timestamps=(response_format == "json")
        )

        logger.info(f"Transcription completed: {result['text'][:100]}...")

        # Return response based on format
        if response_format == "json":
            return TranscriptionResponse(
                text=result["text"],
                language=result["language"],
                duration=result.get("duration"),
                segments=result.get("segments"),
            )
        else:
            # For other formats, return plain text
            return result["text"]

    except ValueError as e:
        logger.error(f"Invalid transcription request: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
