import os
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from predict import analyze_audio, load_model

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / 'uploads'
STATIC_FOLDER = BASE_DIR / 'static' / 'dist'
UPLOAD_FOLDER.mkdir(exist_ok=True)

model_state = {'model': None, 'scaler': None, 'error': None}
ALLOWED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.aiff', '.aif', '.webm'}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once when the backend starts."""
    try:
        model_state['model'], model_state['scaler'] = load_model(BASE_DIR)
        print('Model loaded successfully. Backend is ready for audio analysis.')
    except (FileNotFoundError, ValueError) as error:
        model_state['error'] = str(error)
        print(f'Model is not ready: {error}. Run: python train.py')
    yield


app = FastAPI(title='AI Voice Detection API', version='1.0.0', lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

if (STATIC_FOLDER / 'assets').exists():
    app.mount('/assets', StaticFiles(directory=STATIC_FOLDER / 'assets'), name='assets')


@app.get('/api/health')
async def health():
    return {
        'status': 'ok' if model_state['model'] is not None else 'degraded',
        'model_loaded': model_state['model'] is not None,
        'model_error': model_state['error'],
    }


@app.post('/api/analyze')
async def analyze(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail='No file selected')
    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        formats = ', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f'Invalid audio format. Supported formats: {formats}')
    if model_state['model'] is None:
        raise HTTPException(status_code=503, detail='Model is not loaded. Run python train.py, then restart the backend.')

    file_path = UPLOAD_FOLDER / f'{uuid4().hex}{extension}'
    try:
        file_path.write_bytes(await file.read())
        prediction, confidence = analyze_audio(
            file_path,
            model=model_state['model'],
            scaler=model_state['scaler'],
        )
        return {'prediction': prediction, 'confidence': float(confidence)}
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        print(f'Error processing file: {error}')
        raise HTTPException(status_code=500, detail='An error occurred while processing the audio file') from error
    finally:
        file_path.unlink(missing_ok=True)


@app.get('/{path:path}')
async def serve_react(path: str):
    requested_file = STATIC_FOLDER / path
    if path and requested_file.is_file():
        return FileResponse(requested_file)
    index_file = STATIC_FOLDER / 'index.html'
    if index_file.exists():
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail='React build not found. Run npm run build in frontend/.')


if __name__ == '__main__':
    import uvicorn

    uvicorn.run('app:app', host='0.0.0.0', port=5000, reload=True)