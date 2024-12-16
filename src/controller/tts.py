import os
import torch
import nltk
import json
import asyncio
import websockets
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS
from transformers import BertTokenizer, BertForMaskedLM
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from functools import partial


class TextToSpeechConverter:
    def __init__(self, host='0.0.0.0', port=8888):
        print("Initializing Text-to-Speech Converter...")
        
        # Download required NLTK resources
        print("Downloading NLTK resources...")
        try:
            nltk.download('averaged_perceptron_tagger_eng')
        except Exception as e:
            print(f"Error downloading NLTK resources: {e}")
            raise
            
        # Check for CUDA availability
        print("Checking CUDA availability...")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but not available. Please ensure you have a GPU and CUDA installed.")
            
        # Initialize paths and device
        print("Loading tone color converter...")
        self.ckpt_converter = 'OpenVoiceV2/converter'
        self.device = "cuda:0"
        self.output_dir = 'outputs_v2'
        
        # Add error handling for model loading
        try:
            self.tone_color_converter = ToneColorConverter(f'{self.ckpt_converter}/config.json', device=self.device)
            self.tone_color_converter.load_ckpt(f'{self.ckpt_converter}/checkpoint.pth')
        except Exception as e:
            print(f"Error loading tone color converter: {e}")
            raise
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.host = host
        self.port = port

        # Initialize TTS model and load speaker embedding
        print("Loading TTS model and speaker embedding...")
        self.tts_model = TTS(language='EN', device=self.device)
        self.speaker_key = 'en-us'
        self.speaker_id = self.tts_model.hps.data.spk2id[self.speaker_key.upper()]
        self.source_se = torch.load(
            f'OpenVoiceV2/base_speakers/ses/{self.speaker_key}.pth',
            map_location=self.device,
            weights_only=True
        )

        # Pre-load BERT model
        print("Loading BERT model...")
        try:
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
            self.bert_model.to(self.device)
        except Exception as e:
            print(f"Error loading BERT model: {e}")
            raise
            
        print("Initialization complete! Starting server...")

        # Add timeout configuration
        self.connection_timeout = 300  # 5 minutes
        self.chunk_timeout = 30  # 30 seconds per chunk

        # Configure multiprocessing settings
        self.num_processes = multiprocessing.cpu_count()
        self.process_pool = multiprocessing.Pool(processes=self.num_processes)
        
        # Configure thread pool for I/O operations
        self.thread_pool = ThreadPoolExecutor(max_workers=32)
        
        # Enhanced request queue with maxsize
        self.request_queue = asyncio.Queue(maxsize=10000)
        
        # Add worker pools
        self.worker_tasks = set()
        self.max_workers = 4  # Number of concurrent speech generation workers
        
        # Initialize processing flag
        self.processing = False

        # Add shutdown flag
        self.shutdown_flag = False

    async def handle_websocket(self, websocket, path):
        """Handle WebSocket connections"""
        try:
            print(f"New connection from client")
            self.active_connections.add(websocket)
            websocket.ping_timeout = self.connection_timeout
            
            try:
                async for message in websocket:
                    if self.shutdown_flag:
                        await websocket.send(json.dumps({
                            'status': 'shutdown',
                            'message': 'Server is shutting down'
                        }))
                        break
                    
                    # Parse the incoming JSON message
                    data = json.loads(message)
                    
                    # Add request to queue
                    await self.request_queue.put((websocket, data))
                    
                    # Start processing if not already running
                    if not self.processing:
                        self.processing = True
                        asyncio.create_task(self.process_queue())
                    
                    # Send acknowledgment
                    await websocket.send(json.dumps({
                        'status': 'queued',
                        'position': self.request_queue.qsize()
                    }))
                    
            finally:
                self.active_connections.remove(websocket)
                
        except Exception as e:
            print(f"WebSocket handler error: {e}")
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def process_queue(self):
        """Process requests from the queue using multiple workers"""
        while not self.request_queue.empty():
            # Clean up completed workers
            self.worker_tasks = {task for task in self.worker_tasks if not task.done()}
            
            # If we have capacity, start new workers
            while len(self.worker_tasks) < self.max_workers and not self.request_queue.empty():
                worker = asyncio.create_task(self.process_single_request())
                self.worker_tasks.add(worker)
            
            await asyncio.sleep(0.1)  # Prevent CPU spinning
        
        # Wait for remaining tasks
        if self.worker_tasks:
            await asyncio.wait(self.worker_tasks)
        self.processing = False

    async def process_single_request(self):
        """Process a single request from the queue"""
        try:
            websocket, data = await self.request_queue.get()
            
            # Check if websocket is still open
            if websocket.closed:
                self.request_queue.task_done()
                return

            # Extract parameters
            text = data.get('text')
            language = data.get('language', 'EN')
            speed = float(data.get('speed', 1.0))
            
            # Generate audio in thread pool to prevent blocking
            try:
                audio_data = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool,
                    partial(
                        self.generate_speech_stream,
                        text,
                        language,
                        'resources/example_reference.mp3',
                        speed
                    )
                )
                
                # Verify audio data is not None
                if audio_data is None:
                    raise ValueError("Failed to generate audio data")

                # Stream audio data in larger chunks
                chunk_size = 32768
                for i in range(0, len(audio_data), chunk_size):
                    if websocket.closed:
                        break
                    chunk = audio_data[i:i + chunk_size]
                    await websocket.send(chunk)
                
                if not websocket.closed:
                    await websocket.send(json.dumps({'status': 'finished'}))
            except Exception as e:
                print(f"Error generating speech: {e}")
                if not websocket.closed:
                    await self.send_error(websocket, str(e))
            
            self.request_queue.task_done()
            
        except Exception as e:
            print(f"Error processing request: {e}")
            if not websocket.closed:
                await self.send_error(websocket, str(e))
            self.request_queue.task_done()

    async def send_error(self, websocket, error_message):
        """Helper method to send error messages"""
        try:
            await websocket.send(json.dumps({'error': error_message}))
        except:
            print("Failed to send error message")

    async def start_server(self):
        """Start the WebSocket server"""
        try:
            # Track active connections
            self.active_connections = set()
            
            server = await websockets.serve(
                self.handle_websocket,
                self.host,
                self.port,
                ping_interval=None,
                max_size=2**23,
                max_queue=2**10
            )
            print(f"WebSocket server listening on ws://{self.host}:{self.port}")
            
            # Keep server running until shutdown flag is set
            while not self.shutdown_flag:
                await asyncio.sleep(1)
            
            # Graceful shutdown sequence
            print("\nInitiating graceful shutdown...")
            
            # Close server to prevent new connections
            server.close()
            await server.wait_closed()
            
            # Close all active websocket connections
            if self.active_connections:
                print(f"Closing {len(self.active_connections)} active connections...")
                close_tasks = [ws.close() for ws in self.active_connections]
                await asyncio.gather(*close_tasks, return_exceptions=True)
            
            # Wait for ongoing tasks to complete
            if self.worker_tasks:
                print(f"Waiting for {len(self.worker_tasks)} tasks to complete...")
                await asyncio.wait(self.worker_tasks)
            
            # Cleanup resources
            await self.cleanup()
            print("Server shutdown complete")
            
        except Exception as e:
            print(f"Server error: {e}")
            raise

    def extract_voice_embedding(self, reference_speaker_path):
        """Extract tone color embedding from reference voice"""
        return se_extractor.get_se(reference_speaker_path, self.tone_color_converter, vad=False)

    def generate_speech_stream(self, text, language, reference_speaker_path, speed=1.0):
        """Generate and stream speech data"""
        try:
            # Validate inputs
            if not text or not reference_speaker_path:
                raise ValueError("Text and reference_speaker_path are required")

            # Initialize cache attributes if they don't exist
            if not hasattr(self, '_cached_reference_path'):
                self._cached_reference_path = None
                self._cached_target_se = None
            
            # Cache target_se if not already cached or if reference_speaker_path changed
            if self._cached_reference_path != reference_speaker_path:
                print(f"Extracting voice embedding for {reference_speaker_path}")
                self._cached_reference_path = reference_speaker_path
                target_se, _ = self.extract_voice_embedding(reference_speaker_path)
                if target_se is None:
                    raise ValueError("Failed to extract voice embedding")
                self._cached_target_se = target_se
            
            # Use cached target_se
            target_se = self._cached_target_se
            
            if target_se is None:
                raise ValueError("Target voice embedding is None")

            # Create temporary files for intermediate and final audio
            temp_src_file = os.path.join(self.output_dir, 'temp_audio_src.wav')
            temp_converted_file = os.path.join(self.output_dir, 'temp_audio_converted.wav')
            
            # Ensure TTS model is on GPU
            self.tts_model.to(self.device)

            # Generate initial speech using cached model and speaker
            self.tts_model.tts_to_file(
                text=text,
                speaker_id=self.speaker_id,
                speed=speed,
                output_path=temp_src_file
            )
            
            # Run the tone color converter
            self.tone_color_converter.convert(
                audio_src_path=temp_src_file,
                src_se=self.source_se,
                tgt_se=target_se,
                output_path=temp_converted_file,
                message="@MyShell"
            )
            
            # Read and return the converted audio
            with open(temp_converted_file, 'rb') as f:
                audio_data = f.read()
            
            # Clean up temporary files
            os.remove(temp_src_file)
            os.remove(temp_converted_file)
            
            return audio_data
                
        except Exception as e:
            print(f"Error in generate_speech_stream: {e}")
            raise

    async def cleanup(self):
        """Cleanup resources on shutdown"""
        print("Cleaning up resources...")
        
        # Cancel any pending tasks
        remaining_tasks = [task for task in self.worker_tasks if not task.done()]
        for task in remaining_tasks:
            task.cancel()
        
        if remaining_tasks:
            await asyncio.wait(remaining_tasks)
        
        # Shutdown process pool gracefully
        self.process_pool.close()
        self.process_pool.join()
        
        # Shutdown thread pool with timeout
        self.thread_pool.shutdown(wait=True, cancel_futures=True)

if __name__ == "__main__":
    tts = TextToSpeechConverter()
    print(f"Starting WebSocket server on ws://0.0.0.0:8888")
    try:
        asyncio.run(tts.start_server())
    except KeyboardInterrupt:
        print("\nReceived shutdown signal (Ctrl+C)")
        tts.shutdown_flag = True

