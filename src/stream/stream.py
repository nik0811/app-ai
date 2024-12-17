from src.controller.llm import get_llm_response
from src.controller.tts import TextToSpeechConverter
from src.controller.stt import TranscriptionService
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

async def recognize_from_microphone(websocket: WebSocket):
    try:
        while True:
            data = await websocket.receive_bytes()
            transcription_service = TranscriptionService(model_path="large-v3")
            transcription = await transcription_service.process_audio(data)
            print(transcription)
    except WebSocketDisconnect:
        print("Client disconnected")
