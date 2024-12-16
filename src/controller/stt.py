import argparse
import logging
import numpy as np
from faster_whisper import WhisperModel
import asyncio
import websockets
import signal
import torch
import json
import difflib
from typing import List, Dict
from dataclasses import dataclass
import queue
from concurrent.futures import ThreadPoolExecutor
import re

@dataclass
class AudioConfig:
    """Audio processing configuration"""
    SAMPLE_RATE: int = 16000
    MIN_DURATION: float = 2.0  # Reduced from 30s for faster processing
    CHUNK_DURATION: float = 1.0
    OVERLAP_DURATION: float = 0.2

    @property
    def chunk_size(self) -> int:
        return int(self.SAMPLE_RATE * self.CHUNK_DURATION)

    @property
    def overlap_size(self) -> int:
        return int(self.SAMPLE_RATE * self.OVERLAP_DURATION)

class AudioProcessor:
    """Handles audio preprocessing and buffering"""
    def __init__(self, config: AudioConfig):
        self.config = config
        self.buffer = np.array([], dtype=np.float32)
        
    def process_chunk(self, audio_chunk: bytes) -> np.ndarray:
        chunk_data = np.frombuffer(audio_chunk, dtype=np.float32)
        
        # Normalize audio
        if len(chunk_data) > 0:
            chunk_data = chunk_data / np.max(np.abs(chunk_data))
        
        self.buffer = np.append(self.buffer, chunk_data)
        
        # Process when we have enough data
        if len(self.buffer) >= self.config.chunk_size:
            # Extract the chunk to process
            chunk_to_process = self.buffer[:self.config.chunk_size]
            # Keep overlap for continuity
            self.buffer = self.buffer[-self.config.overlap_size:]
            
            # Ensure 4-byte alignment
            if len(chunk_to_process) % 4 != 0:
                padding = 4 - (len(chunk_to_process) % 4)
                chunk_to_process = np.pad(chunk_to_process, (0, padding), mode='constant')
            
            return chunk_to_process
        return None

class TranscriptionWorker:
    """Manages transcription processing"""
    def __init__(self, model: WhisperModel, num_workers: int = 4):
        self.model = model
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.queue = queue.Queue()
        self._setup_processing()
        self.pending_tasks = []  # Track pending transcription tasks

    def _setup_processing(self):
        self.transcription_params = {
            # Beam Search - Further optimized for accuracy
            'beam_size': 5,
            'best_of': 5,
            'patience': 1.5,  # Reduced for faster processing
            
            # Enhanced repetition prevention
            'length_penalty': 1.2,  # Increased to favor longer, more complete transcriptions
            'repetition_penalty': 1.3,  # Increased to better prevent repetition
            'no_repeat_ngram_size': 3,  # Increased to catch longer repeated phrases
            
            # Refined confidence thresholds
            'temperature': 0.0,  # Keep deterministic
            'compression_ratio_threshold': 2.4,  # Increased to better handle natural speech
            'log_prob_threshold': -0.8,  # Adjusted for better filtering
            'no_speech_threshold': 0.4,  # Increased to better filter non-speech
            
            # Enhanced context handling
            'condition_on_previous_text': True,
            'initial_prompt': None,
            'prefix': None,
            
            # Improved text processing
            'suppress_blank': True,
            'suppress_tokens': [-1],
            'without_timestamps': False,
            'max_initial_timestamp': 0.8,  # Reduced for more accurate start times
            
            # Word timestamps for detailed output
            'word_timestamps': True,
            
            # Optimized VAD parameters
            'vad_filter': True,
            'vad_parameters': {
                'min_silence_duration_ms': 300,  # Reduced for more natural breaks
                'speech_pad_ms': 150,  # Increased for better context
                'min_speech_duration_ms': 150,  # Reduced to catch shorter utterances
                'max_speech_duration_s': 6.0  # Increased for longer phrases
            }
        }

    async def transcribe(self, audio_data: np.ndarray) -> Dict:
        # Submit transcription task to thread pool
        future = self.executor.submit(self._transcribe_sync, audio_data)
        self.pending_tasks.append(future)
        return await asyncio.get_event_loop().run_in_executor(None, future.result)

    def _transcribe_sync(self, audio_data: np.ndarray) -> Dict:
        try:
            segments, info = self.model.transcribe(audio_data, **self.transcription_params)
            transcription = self._process_segments(segments)
            
            if transcription:
                return self._create_response(transcription, segments)
            return {"results": {"channels": []}}
            
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            return {"results": {"channels": []}}

    def _process_segments(self, segments: List) -> str:
        transcription = []
        prev_text = ""
        
        for segment in segments:
            text = segment.text.strip()
            # Enhanced similarity check with context
            if text and not self._is_similar(text, prev_text):
                # Clean up common transcription artifacts
                text = self._clean_text(text)
                if text:  # Only add if text remains after cleaning
                    transcription.append(text)
                    prev_text = text
        
        return " ".join(transcription)

    def _create_response(self, transcription: str, segments) -> Dict:
        return {
            "results": {
                "channels": [{
                    "alternatives": [{
                        "transcript": transcription,
                        "words": [
                            {
                                "word": word.word,
                                "start": word.start,
                                "end": word.end,
                                "confidence": getattr(word, 'probability', 0.0)
                            } for segment in segments for word in segment.words
                        ]
                    }]
                }]
            }
        }

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean up common transcription artifacts"""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove standalone punctuation
        text = re.sub(r'\s*([,.!?])\s*', r'\1 ', text)
        # Remove repeated punctuation
        text = re.sub(r'([,.!?])\1+', r'\1', text)
        return text.strip()

    @staticmethod
    def _is_similar(text1: str, text2: str, threshold: float = 0.85) -> bool:
        """Enhanced similarity check with better threshold"""
        # Ignore case and extra whitespace for comparison
        text1 = re.sub(r'\s+', ' ', text1.lower().strip())
        text2 = re.sub(r'\s+', ' ', text2.lower().strip())
        return difflib.SequenceMatcher(None, text1, text2).ratio() > threshold

async def handle_client(websocket, worker: TranscriptionWorker):
    """Handle individual client connection"""
    audio_config = AudioConfig()
    processor = AudioProcessor(audio_config)
    final_buffer = None
    pending_chunks = []  # Track chunks being processed
    
    try:
        async for message in websocket:
            if not isinstance(message, bytes):
                continue
            
            if message == b"EOS":
                # Wait for all pending transcriptions to complete
                for task in pending_chunks:
                    response = await task
                    await websocket.send(json.dumps(response))
                
                if len(processor.buffer) > 0:
                    final_buffer = processor.buffer
                    response = await worker.transcribe(final_buffer)
                    await websocket.send(json.dumps(response))
                
                logging.info("Received EOS signal, closing connection")
                await websocket.close(code=1000, reason="Transcription completed")
                break
                
            chunk = processor.process_chunk(message)
            if chunk is not None:
                # Process chunk asynchronously
                transcription_task = asyncio.create_task(worker.transcribe(chunk))
                pending_chunks.append(transcription_task)
                
                # Process completed transcriptions
                completed = []
                for task in pending_chunks:
                    if task.done():
                        response = await task
                        await websocket.send(json.dumps(response))
                        completed.append(task)
                
                # Remove completed tasks
                for task in completed:
                    pending_chunks.remove(task)
    except websockets.exceptions.ConnectionClosedOK:
        logging.info("Client disconnected normally")
    except websockets.exceptions.ConnectionClosedError as e:
        logging.warning(f"Connection closed unexpectedly: {e}")
    except Exception as e:
        logging.error(f"Error handling client: {e}")
        try:
            await websocket.close(code=1011, reason=f"Internal server error: {str(e)}")
        except:
            pass
    finally:
        # Clean up resources
        processor.buffer = np.array([], dtype=np.float32)

def initialize_model(model_path: str, device: str, num_workers: int) -> WhisperModel:
    """Initialize and optimize the Whisper model"""
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TensorFloat-32
        torch.cuda.set_per_process_memory_fraction(0.85, device=0)
    
    return WhisperModel(
        model_path,
        device=device,
        compute_type="float16" if device == "cuda" else "int8",
        cpu_threads=num_workers if device == "cpu" else 4,
        num_workers=num_workers
    )

async def start_server(model_path: str, port: int, num_workers: int):
    """Start the WebSocket server"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    model = initialize_model(model_path, device, num_workers)
    worker = TranscriptionWorker(model, num_workers)
    
    stop = asyncio.Event()
    server = await websockets.serve(
        lambda ws: handle_client(ws, worker),
        "localhost",
        port
    )
    
    logging.info(f"Server started on ws://localhost:{port}")
    
    try:
        await stop.wait()  # Wait until stop event is set
    finally:
        server.close()
        await server.wait_closed()
        worker.executor.shutdown(wait=True)
        logging.info("Server shutdown complete")

def start_websocket_server(model_path: str, port: int, num_workers: int):
    """Initialize and start the WebSocket server"""
    loop = asyncio.get_event_loop()
    main_task = asyncio.ensure_future(start_server(model_path, port, num_workers))
    
    def signal_handler():
        logging.info("Shutting down server...")
        main_task.cancel()
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        loop.run_until_complete(main_task)
    except asyncio.CancelledError:
        pass
    finally:
        loop.close()
        logging.info("Server shutdown complete")

# Main server class and startup code remains similar, but uses these optimized components

def _main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="WebSocket transcription service")

    parser.add_argument('-m', '--model', default='large-v3', type=str, help="Whisper model path, default to %(default)s")
    parser.add_argument('-ws', '--websocket_port', type=int, default=8765, help="WebSocket server port, default to %(default)s")
    parser.add_argument('-w', '--workers', type=int, default=11, help="Number of threads for parallel processing (default: 11)")
    parser.add_argument(
        '-l', '--log_level', default='INFO', type=str, help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    start_websocket_server(model_path=args.model, port=args.websocket_port, num_workers=args.workers)

if __name__ == '__main__':
    _main()

