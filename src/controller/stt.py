import argparse
import logging
import numpy as np
from faster_whisper import WhisperModel
import asyncio
import torch
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
        # Add parameters for audio enhancement
        self.noise_reduce_strength = 0.15
        self.noise_floor = 0.001
        self.gain = 1.2
        
    def process_chunk(self, audio_chunk: bytes) -> np.ndarray:
        # Ensure audio_chunk length is multiple of 4 before converting to float32
        if len(audio_chunk) % 4 != 0:
            padding = 4 - (len(audio_chunk) % 4)
            audio_chunk = audio_chunk + b'\x00' * padding
            
        chunk_data = np.frombuffer(audio_chunk, dtype=np.float32)
        
        if len(chunk_data) > 0:
            # Apply noise reduction
            chunk_data = self._reduce_noise(chunk_data)
            # Apply audio enhancement
            chunk_data = self._enhance_audio(chunk_data)
            # Normalize after enhancement
            chunk_data = chunk_data / np.max(np.abs(chunk_data))
        
        self.buffer = np.append(self.buffer, chunk_data)
        
        # Process when we have enough data
        if len(self.buffer) >= self.config.chunk_size:
            # Extract the chunk to process
            chunk_to_process = self.buffer[:self.config.chunk_size]
            # Keep overlap for continuity
            self.buffer = self.buffer[-self.config.overlap_size:]
            return chunk_to_process
        return None

    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """Apply simple noise reduction using spectral gating"""
        # Estimate noise floor from quiet parts
        noise_mask = np.abs(audio) < self.noise_floor
        if np.any(noise_mask):
            noise_estimate = np.mean(np.abs(audio[noise_mask]))
            # Apply soft threshold
            reduction = np.maximum(0, np.abs(audio) - noise_estimate * self.noise_reduce_strength)
            return np.sign(audio) * reduction
        return audio

    def _enhance_audio(self, audio: np.ndarray) -> np.ndarray:
        """Enhance audio clarity"""
        # Apply subtle compression
        threshold = 0.3
        ratio = 0.6
        makeup_gain = self.gain

        # Calculate amplitude envelope
        amplitude = np.abs(audio)
        mask = amplitude > threshold
        
        if np.any(mask):
            # Apply compression only to samples above threshold
            compressed = np.copy(audio)
            compressed[mask] *= (1 + (threshold - amplitude[mask]) * (1 - ratio))
            # Apply makeup gain
            compressed *= makeup_gain
            return compressed
        
        return audio * makeup_gain

class TranscriptionWorker:
    """Manages transcription processing"""
    def __init__(self, model: WhisperModel, num_workers: int = 4):
        self.model = model
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.queue = queue.Queue()
        self._setup_processing()
        self.pending_tasks = []  # Track pending transcription tasks
        self.previous_text = ""  # Add buffer for previous text
        self.text_buffer = []    # Add buffer for collecting text segments

    def _setup_processing(self):
        self.transcription_params = {
            # Beam Search - Further optimized for accuracy
            'beam_size': 5,
            'best_of': 5,
            'patience': 1.5,  # Reduced for faster processing
            
            # Enhanced repetition prevention
            'length_penalty': 0.8,        # Reduced to avoid over-penalizing longer segments
            'repetition_penalty': 1.1,    # Reduced to avoid over-aggressive repetition prevention
            'no_repeat_ngram_size': 3,  # Increased to catch longer repeated phrases
            
            # Refined confidence thresholds
            'temperature': 0.0,  # Keep deterministic
            'compression_ratio_threshold': 1.8,  # Reduced to be more permissive
            'log_prob_threshold': -1.0,   # More permissive threshold
            'no_speech_threshold': 0.5,  # More strict non-speech filtering
            
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
                'min_silence_duration_ms': 500,  # Increased for better sentence grouping
                'speech_pad_ms': 300,           # Increased padding
                'min_speech_duration_ms': 250,  # Increased minimum duration
                'max_speech_duration_s': 10.0   # Increased for longer phrases
            },
            
            # Add language forcing
            'language': 'en',  # Force English language
            'task': 'transcribe',  # Force transcription task
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
        current_segments = []
        
        for segment in segments:
            text = segment.text.strip()
            if text:
                # Clean up the text
                text = self._clean_text(text)
                
                # Only add if it's not too similar to previous text
                if not self._is_similar(text, self.previous_text):
                    current_segments.append(text)
                    self.previous_text = text
        
        # Combine segments and add to buffer
        if current_segments:
            combined_text = " ".join(current_segments)
            self.text_buffer.append(combined_text)
            
            # Keep only last few segments in buffer to maintain context
            if len(self.text_buffer) > 3:
                self.text_buffer.pop(0)
            
            # Return combined recent segments for more continuous output
            return " ".join(self.text_buffer)
        
        return ""

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

    def _clean_text(self, text: str) -> str:
        """Clean up common transcription artifacts"""
        # Remove non-English characters
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove standalone punctuation
        text = re.sub(r'\s*([,.!?])\s*', r'\1 ', text)
        # Remove repeated punctuation
        text = re.sub(r'([,.!?])\1+', r'\1', text)
        # Remove single-character words except 'a' and 'i'
        text = re.sub(r'\b[b-hj-z]\b', '', text, flags=re.IGNORECASE)
        return text.strip()

    @staticmethod
    def _is_similar(text1: str, text2: str, threshold: float = 0.85) -> bool:
        """Enhanced similarity check with better threshold"""
        # Ignore case and extra whitespace for comparison
        text1 = re.sub(r'\s+', ' ', text1.lower().strip())
        text2 = re.sub(r'\s+', ' ', text2.lower().strip())
        return difflib.SequenceMatcher(None, text1, text2).ratio() > threshold

class TranscriptionService:
    """Handles audio transcription"""
    def __init__(self, model_path: str, num_workers: int = 4):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._initialize_model(model_path, num_workers)
        self.worker = TranscriptionWorker(self.model, num_workers)
        self.audio_config = AudioConfig()
        self.processor = AudioProcessor(self.audio_config)
        
    def _initialize_model(self, model_path: str, num_workers: int) -> WhisperModel:
        """Initialize and optimize the Whisper model"""
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.cuda.set_per_process_memory_fraction(0.85, device=0)
        
        return WhisperModel(
            model_path,
            device=self.device,
            compute_type="float16" if self.device == "cuda" else "int8",
            cpu_threads=num_workers if self.device == "cpu" else 4,
            num_workers=num_workers
        )

    async def process_audio(self, audio_chunk: bytes) -> Dict:
        """Process a chunk of audio data"""
        chunk = self.processor.process_chunk(audio_chunk)
        if chunk is not None:
            return await self.worker.transcribe(chunk)
        return {"results": {"channels": []}}

    async def finalize(self) -> Dict:
        """Process any remaining audio in the buffer"""
        if len(self.processor.buffer) > 0:
            return await self.worker.transcribe(self.processor.buffer)
        return {"results": {"channels": []}}

    def cleanup(self):
        """Clean up resources"""
        self.worker.executor.shutdown(wait=True)
        self.processor.buffer = np.array([], dtype=np.float32)

def _main():
    """Example usage"""
    parser = argparse.ArgumentParser(description="Audio transcription service")
    parser.add_argument('-m', '--model', default='large-v3', type=str, help="Whisper model path")
    parser.add_argument('-w', '--workers', type=int, default=4, help="Number of worker threads")
    parser.add_argument('-l', '--log_level', default='INFO', type=str, 
                       help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    
    # Example usage
    service = TranscriptionService(model_path=args.model, num_workers=args.workers)
    # Use service.process_audio() and service.finalize() as needed

if __name__ == '__main__':
    _main()

