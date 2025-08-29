#!/usr/bin/env python3
"""
Comprehensive test script for Fluid Server Whisper transcription
Tests the compiled executable with various audio files
"""

import sys
import time
import json
import requests
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import argparse

class FluidServerTester:
    def __init__(self, server_executable: str, host: str = "127.0.0.1", port: int = 8088):
        self.server_executable = Path(server_executable)
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.server_process = None
        
    def start_server(self, model_path: str) -> bool:
        """Start the Fluid Server executable"""
        try:
            print(f"Starting server: {self.server_executable}")
            cmd = [
                str(self.server_executable),
                "--host", self.host,
                "--port", str(self.port),
                "--llm-model", "qwen3-4b-int8-ov",
                "--whisper-model", "whisper-large-v3-turbo-fp16-ov-npu",
                "--model-path", model_path
            ]
            
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            print("Waiting for server to start...")
            for i in range(60):  # Wait up to 60 seconds
                try:
                    response = requests.get(f"{self.base_url}/health", timeout=1)
                    if response.status_code == 200:
                        print("Server started successfully")
                        time.sleep(5)  # Wait for model warmup
                        return True
                except requests.RequestException:
                    pass
                time.sleep(1)
                print(f"  Waiting... ({i+1}/60)")
            
            print("Server failed to start within 60 seconds")
            return False
            
        except Exception as e:
            print(f"Error starting server: {e}")
            return False
    
    def stop_server(self):
        """Stop the server process"""
        if self.server_process:
            print("Stopping server...")
            self.server_process.terminate()
            self.server_process.wait()
            self.server_process = None
    
    def test_health(self) -> bool:
        """Test server health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                print("Health check passed")
                return True
            else:
                print(f"Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"Health check error: {e}")
            return False
    
    def test_models_endpoint(self) -> bool:
        """Test models listing endpoint"""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                print(f"Models endpoint working: {len(models.get('data', []))} models")
                return True
            else:
                print(f"Models endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"Models endpoint error: {e}")
            return False
    
    def transcribe_file(self, audio_file: str, timeout: int = 30) -> Optional[Dict]:
        """Transcribe an audio file"""
        try:
            audio_path = Path(audio_file)
            if not audio_path.exists():
                print(f"Audio file not found: {audio_file}")
                return None
            
            print(f"Transcribing: {audio_path.name}")
            
            with open(audio_path, 'rb') as f:
                files = {
                    'file': (audio_path.name, f, 'audio/wav'),
                    'model': (None, 'whisper-large-v3-turbo-fp16-ov-npu')
                }
                
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/v1/audio/transcriptions",
                    files=files,
                    timeout=timeout
                )
                elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"Transcription successful ({elapsed:.1f}s)")
                print(f"  Text: '{result.get('text', '')}'")
                print(f"  Duration: {result.get('duration', 0):.2f}s")
                return result
            else:
                print(f"Transcription failed: {response.status_code}")
                print(f"  Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"Transcription error: {e}")
            return None
    
    def run_tests(self, audio_files: List[str], model_path: str) -> Dict[str, bool]:
        """Run comprehensive tests"""
        results = {}
        
        print("=" * 50)
        print("FLUID SERVER WHISPER TESTING")
        print("=" * 50)
        
        # Start server
        if not self.start_server(model_path):
            return {"server_start": False}
        
        results["server_start"] = True
        
        try:
            # Test health endpoint
            results["health"] = self.test_health()
            
            # Test models endpoint
            results["models"] = self.test_models_endpoint()
            
            # Test transcriptions
            transcription_results = []
            for audio_file in audio_files:
                result = self.transcribe_file(audio_file)
                transcription_results.append(result is not None)
            
            results["transcriptions"] = all(transcription_results)
            results["transcription_count"] = sum(transcription_results)
            results["total_files"] = len(audio_files)
            
        finally:
            self.stop_server()
        
        return results

def convert_audio_file(input_file: str) -> Optional[str]:
    """Convert audio file to 16kHz mono WAV"""
    try:
        script_path = Path(__file__).parent / "convert_audio.py"
        cmd = [sys.executable, str(script_path), input_file]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            # Extract output filename from the conversion script output
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.startswith("Successfully converted to:"):
                    return line.split(": ", 1)[1]
        else:
            print(f"Conversion failed: {result.stderr}")
        return None
    except Exception as e:
        print(f"Error converting audio: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Test Fluid Server Whisper transcription")
    parser.add_argument("--server", default="../windows/fluid-server.exe", 
                       help="Path to fluid-server.exe")
    parser.add_argument("--model-path", default=r"Models/",
                       help="Path to models directory")
    parser.add_argument("--audio-files", nargs="+", 
                       default=["test_tone.wav", "startup.wav"],
                       help="Audio files to test")
    parser.add_argument("--convert", action="store_true",
                       help="Convert audio files to 16kHz mono before testing")
    parser.add_argument("--port", type=int, default=8088,
                       help="Server port")
    
    args = parser.parse_args()
    
    # Convert audio files if requested
    test_files = []
    for audio_file in args.audio_files:
        if args.convert:
            converted = convert_audio_file(audio_file)
            if converted:
                test_files.append(converted)
            else:
                print(f"Skipping {audio_file} - conversion failed")
        else:
            test_files.append(audio_file)
    
    if not test_files:
        print("No audio files to test!")
        sys.exit(1)
    
    # Run tests
    tester = FluidServerTester(args.server, port=args.port)
    results = tester.run_tests(test_files, args.model_path)
    
    # Print results
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20}: {status}")
    
    # Overall result
    success = all(v for k, v in results.items() if k not in ["transcription_count", "total_files"])
    print(f"\nOverall: {'ALL TESTS PASSED' if success else 'SOME TESTS FAILED'}")
    
    if "transcription_count" in results:
        print(f"Transcriptions: {results['transcription_count']}/{results['total_files']} successful")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()