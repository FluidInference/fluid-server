# NPU Support Guide

Fluid Server supports multiple NPU (Neural Processing Unit) runtimes for optimal performance on different hardware architectures. This guide covers the supported NPU backends and their specific configurations.

## Supported NPU Runtimes

### Intel NPU (OpenVINO)

Intel NPU support is provided through the OpenVINO runtime, optimized for Intel NPU and integrated graphics.

#### Model Format
- **Format**: OpenVINO IR format (.xml/.bin files)
- **Location**: `models/whisper/whisper-large-v3-turbo-fp16-ov-npu/`
- **Optimization**: Optimized for Intel NPU and integrated graphics

#### Performance Characteristics
- Excellent performance on Intel Arc graphics and NPU
- Low power consumption
- Optimized for Intel's AI acceleration hardware

#### Model Directory Structure
```
models/whisper/whisper-large-v3-turbo-fp16-ov-npu/
├── openvino_model.xml
├── openvino_model.bin
└── config.json
```

### Qualcomm NPU (QNN)

Qualcomm NPU support uses the Qualcomm Neural Network (QNN) SDK with device-specific compilation for Snapdragon processors.

#### Model Format
- **Format**: ONNX format with device-specific compilation
- **Location**: `models/whisper/whisper-large-v3-turbo-qnn/snapdragon-x-elite/`
- **Performance**: 16× real-time transcription on Snapdragon X Elite
- **Hardware**: Snapdragon X Elite devices with HTP (Hexagon Tensor Processor)

#### Performance Characteristics
- Exceptional performance on Snapdragon X Elite devices
- Leverages Hexagon Tensor Processor (HTP) for AI acceleration
- 16× real-time transcription performance
- Optimized for ARM64 architecture

#### Model Directory Structure
```
models/whisper/whisper-large-v3-turbo-qnn/snapdragon-x-elite/
├── whisper_encoder.onnx
├── whisper_decoder.onnx
└── config.json
```

## Runtime Selection

The server automatically detects your hardware and selects the appropriate NPU runtime:

### Automatic Detection
- **ARM64 Architecture**: QNN backend is automatically preferred
- **Intel x64 Architecture**: OpenVINO backend is automatically preferred
- **Fallback**: CPU-based inference if NPU is unavailable

### Manual Runtime Selection
You can explicitly specify the runtime through command-line arguments:

```powershell
# Force OpenVINO runtime
.\dist\fluid-server.exe --whisper-model whisper-large-v3-turbo-ov-npu

# Force QNN runtime (ARM64 only)
.\dist\fluid-server.exe --whisper-model whisper-large-v3-turbo-qnn
```

## Hardware Requirements

### Intel NPU Requirements
- **OS**: Windows 10/11
- **Hardware**: Intel Arc graphics or Intel NPU
- **Runtime**: OpenVINO 2025.2.0+ runtime
- **Memory**: 8GB+ RAM recommended

### Qualcomm NPU Requirements
- **OS**: Windows 11 (ARM64)
- **Hardware**: Snapdragon X Elite with HTP
- **Runtime**: ONNX Runtime QNN (bundled with dependencies)
- **Memory**: 8GB+ RAM recommended

## Performance Optimization

### Intel NPU Optimization
1. **Driver Updates**: Ensure latest Intel graphics drivers
2. **OpenVINO Version**: Use OpenVINO 2025.2.0 or later
3. **Model Precision**: FP16 models provide best performance/accuracy balance

### Qualcomm NPU Optimization
1. **Device Compatibility**: Verify Snapdragon X Elite compatibility
2. **Power Settings**: Use high-performance power profile
3. **Memory Management**: Close unnecessary applications for optimal memory usage

## Troubleshooting

### Common Issues

#### Intel NPU Issues
- **Driver Problems**: Update Intel graphics drivers
- **OpenVINO Installation**: Verify OpenVINO runtime is properly installed
- **Model Loading Errors**: Check model file integrity and paths

#### Qualcomm NPU Issues
- **Architecture Mismatch**: Verify ARM64 Windows environment
- **QNN Availability**: Ensure Snapdragon X Elite with HTP support
- **Model Compilation**: Check ONNX Runtime QNN installation

### Debug Commands

```powershell
# Check NPU availability
.\dist\fluid-server.exe --log-level DEBUG

# Test specific runtime
curl -X POST http://localhost:8080/v1/test -H "Content-Type: application/json"

# Verify model loading
curl http://localhost:8080/v1/models
```

### Performance Monitoring

```powershell
# Monitor transcription performance
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@test_audio.wav" \
  -F "model=whisper-large-v3-turbo-qnn" \
  -F "response_format=verbose_json"
```

## Best Practices

1. **Hardware Matching**: Use the NPU runtime that matches your hardware
2. **Model Selection**: Choose the appropriate model size for your use case
3. **Memory Management**: Monitor memory usage with larger models
4. **Performance Testing**: Benchmark different runtimes on your specific hardware
5. **Regular Updates**: Keep NPU drivers and runtimes updated