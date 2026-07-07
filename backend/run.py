"""
run.py — SentinelAI launcher
Downloads the face_landmarker model on first run, then starts Flask.
Compatible with Python 3.10+ on Windows, macOS, Linux.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

MODEL_PATH = Path(__file__).parent / 'face_landmarker.task'
MODEL_URL  = (
    'https://storage.googleapis.com/mediapipe-models/'
    'face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
)


def download_model() -> None:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100_000:
        print(f'[SentinelAI] Model ready: {MODEL_PATH}')
        return

    print('[SentinelAI] Downloading face_landmarker model (~6 MB)...')
    try:
        req = urllib.request.Request(MODEL_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as resp, \
             MODEL_PATH.open('wb') as f:
            total      = int(resp.headers.get('Content-Length', 0))
            downloaded = 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f'\r  {pct:.0f}%  ({downloaded // 1024} KB)', end='', flush=True)
        print(f'\n[SentinelAI] Model saved.')
    except Exception as exc:
        print(f'\n[SentinelAI] Download failed: {exc}')
        print(f'  Manual download: {MODEL_URL}')
        print(f'  Save to:         {MODEL_PATH}')
        sys.exit(1)


if __name__ == '__main__':
    download_model()
    from app import app
    print('\n🚗  SentinelAI — http://localhost:5000')
    print('    Open frontend/index.html in your browser.')
    print('    Press Ctrl+C to stop.\n')
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
