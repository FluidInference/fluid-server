# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
import platform

datas = [('src\\fluid_server', 'fluid_server')]
binaries = []
hiddenimports = [
    'uvicorn.logging', 
    'uvicorn.loops', 
    'uvicorn.loops.auto', 
    'uvicorn.protocols', 
    'uvicorn.protocols.http', 
    'uvicorn.protocols.http.auto', 
    'uvicorn.protocols.websockets', 
    'uvicorn.protocols.websockets.auto', 
    'uvicorn.lifespan', 
    'uvicorn.lifespan.on', 
    'uvicorn.lifespan.off', 
    'fluid_server.app', 
    'fluid_server.managers', 
    'fluid_server.runtimes', 
    'fluid_server.runtimes.llamacpp_llm',
    'fluid_server.runtimes.qnn_whisper',
    'fluid_server.runtimes.onnx_llm',
    'fluid_server.api', 
    'fluid_server.models', 
    'fluid_server.utils',
    'fluid_server.utils.platform_utils',
    'fluid_server.utils.model_converter',
    'librosa', 
    'scipy', 
    'scipy.signal', 
    'scipy.stats', 
    'scipy.stats._distn_infrastructure', 
    'scipy.stats._stats', 
    'scipy.stats.distributions', 
    'scipy.io', 
    'scipy.fft', 
    'numpy', 
    'soundfile', 
    '_soundfile_data', 
    'multiprocessing', 
    'asyncio',
    'llama_cpp',
    'llama_cpp.llama_cpp',
    'llama_cpp._internals',
]

# Architecture-specific package collection
if platform.machine().lower() in ['x86_64', 'amd64']:
    # x64: Only include basic packages, skip OpenVINO collection to avoid torch issues
    collect_packages = ['librosa', 'scipy', 'soundfile', 'llama_cpp']
    hiddenimports.extend([
        'openvino', 
        'openvino_genai', 
        'openvino_tokenizers', 
        'openvino.runtime', 
        'openvino.properties',
    ])
    
    # Explicitly collect openvino_tokenizers to include DLLs
    tmp_ret = collect_all('openvino_tokenizers')
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]
elif platform.machine().lower() in ['arm64', 'aarch64']:
    # ARM64: Include ARM-specific packages
    collect_packages = ['librosa', 'scipy', 'soundfile', 'llama_cpp', 'whisper', 'onnxruntime']
    hiddenimports.extend([
        'onnxruntime',
        'onnxruntime.capi',
        'onnxruntime.providers',
        'whisper',
        'whisper.decoding',
        'whisper.audio',
        'whisper.tokenizer',
    ])
else:
    # Default: basic packages only
    collect_packages = ['librosa', 'scipy', 'soundfile', 'llama_cpp']

# Collect packages normally
for pkg in collect_packages:
    tmp_ret = collect_all(pkg)
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]

a = Analysis(
    ['src\\fluid_server\\__main__.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['openvino.torch', 'openvino.frontend.pytorch'] + (['onnxruntime', 'whisper'] if platform.machine().lower() in ['x86_64', 'amd64'] else []),
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='fluid-server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Disable UPX for ARM64 compatibility
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    # target_arch can be set to 'arm64' or 'x86_64' if needed
    # By default, PyInstaller builds for the host architecture
    codesign_identity=None,
    entitlements_file=None,
)