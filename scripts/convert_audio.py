#!/usr/bin/env python3
"""
Audio conversion script for Fluid Server testing
Converts audio files to 16kHz mono WAV format for optimal compatibility
"""

import sys
import argparse
from pathlib import Path

def convert_audio(input_file: str, output_file: str = None) -> str:
    """Convert audio file to 16kHz mono WAV format"""
    try:
        import librosa
        import soundfile as sf
        
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Generate output filename if not provided
        if output_file is None:
            output_file = input_path.stem + "_16khz_mono.wav"
        
        output_path = Path(output_file)
        
        print(f"Converting: {input_path} -> {output_path}")
        
        # Load audio file and convert to 16kHz mono
        audio_data, sample_rate = librosa.load(
            str(input_path),
            sr=16000,  # Resample to 16kHz
            mono=True  # Convert to mono
        )
        
        print(f"Original: {librosa.get_duration(path=str(input_path)):.2f}s")
        print(f"Converted: {len(audio_data)/16000:.2f}s at 16kHz mono")
        
        # Save as 16-bit WAV file
        sf.write(
            str(output_path),
            audio_data,
            16000,
            subtype='PCM_16'
        )
        
        print(f"Successfully converted to: {output_path}")
        return str(output_path)
        
    except ImportError:
        print("Error: librosa and soundfile are required for audio conversion")
        print("Install with: pip install librosa soundfile")
        return None
    except Exception as e:
        print(f"Error converting audio: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Convert audio to 16kHz mono WAV")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("-o", "--output", help="Output WAV file (optional)")
    
    args = parser.parse_args()
    
    result = convert_audio(args.input, args.output)
    if result:
        print(f"\nConversion successful: {result}")
        sys.exit(0)
    else:
        print("\nConversion failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()