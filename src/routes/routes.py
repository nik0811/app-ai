from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from src.stream.stream import recognize_from_microphone
import logging


router = APIRouter()

@router.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        await recognize_from_microphone(websocket)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logging.error(f"Error in websocket: {str(e)}")