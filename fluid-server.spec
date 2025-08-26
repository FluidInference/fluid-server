# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('src\\fluid_server', 'fluid_server')]
binaries = []
hiddenimports = [
    'openvino', 
    'openvino_genai', 
    'openvino_tokenizers', 
    'openvino.runtime', 
    'openvino.properties', 
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
    'fluid_server.api', 
    'fluid_server.models', 
    'fluid_server.utils', 
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
    'onnxruntime',
    'onnxruntime.capi',
    'onnxruntime.providers',
    'whisper',
    'whisper.decoding',
    'whisper.audio',
    'whisper.tokenizer',
    'torch',
    'fluid_server.runtimes.qnn_whisper',
]

for pkg in ['openvino', 'openvino_genai', 'openvino_tokenizers', 'librosa', 'scipy', 'soundfile']:
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
    excludes=[],
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
