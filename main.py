from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import subprocess
import os
import json
import requests
import time
import asyncio
from roop.processors.frame.face_swapper import get_face_swapper, clear_face_swapper
import roop.globals
import yaml
import logging

logger = logging.getLogger(__name__)
app = FastAPI(
    title="Hệ thống Hoán đổi Khuôn mặt và Giọng nói",
    description="""
Hệ thống API tích hợp nhiều tính năng xử lý AI:

- 🎭 Hoán đổi khuôn mặt giữa hai ảnh
- 🗣️ Chuyển văn bản thành giọng nói
- 🖼️ Tải ảnh và tệp âm thanh
- 🎬 Tạo video hoạt hình từ ảnh và âm thanh
- 🧠 Chuyển đổi và quản lý mô hình AI động

Công nghệ sử dụng: vietTTS, Wav2Lip, InsightFace, OpenAI...
""",
    version="1.0.0",
    contact={
        "name": "Nhóm phát triển AI",
        "email": "support@example.com"
    }
)

# Tích hợp tài liệu OpenAPI tùy chỉnh
def custom_openapi():
    openapi_path = Path(__file__).parent / "openapi.yaml"
    if openapi_path.exists():
        with open(openapi_path, "r") as f:
            return yaml.safe_load(f)
    return app.openapi_schema  # Fallback to default schema if file not found

app.openapi = custom_openapi

UPLOAD_DIR = Path("uploads")
STATIC_DIR = Path("static")
MODELS_DIR = Path("models")
UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
VIDEO_DIR = STATIC_DIR / "Videos"
CONFIG_PATH = Path(__file__).parent / "model_config.json"

# Mount thư mục chứa file tĩnh
app.mount("/static", StaticFiles(directory="./static"), name="static")
app.mount("/models", StaticFiles(directory="./models"), name="models")

@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = Path.home() / "Dev/roop/Web/main.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(), status_code=200)
    return HTMLResponse(content="<h1>Index file not found</h1>", status_code=404)

@app.post("/face_swap/")
async def face_swap(source: UploadFile = File(...), target: UploadFile = File(...)):
    source_path = UPLOAD_DIR / source.filename
    target_path = UPLOAD_DIR / target.filename
    output_path = STATIC_DIR / "output.jpg"
    
    with source_path.open("wb") as buffer:
        shutil.copyfileobj(source.file, buffer)
    with target_path.open("wb") as buffer:
        shutil.copyfileobj(target.file, buffer)
    
    # Kiểm tra và thiết lập model_path nếu cần
    if not hasattr(roop.globals, 'model_path') or not roop.globals.model_path or not os.path.exists(roop.globals.model_path):
        roop.globals.model_path = str(Path(__file__).parent / "models" / "inswapper_128.onnx")
        print(f"[FACE_SWAP] Thiết lập model_path mặc định: {roop.globals.model_path}")
    
    print(f"[FACE_SWAP] Thực hiện face swap với mô hình: {roop.globals.model_path}")
    command = [
        "python", "run.py",
        "-s", str(source_path),
        "-t", str(target_path),
        "-o", str(output_path),
        "--frame-processor", "face_swapper",
        "--model-path", roop.globals.model_path  # Truyền model_path qua tham số
    ]
    try:
        subprocess.run(command, check=True)
        return JSONResponse(content={"message": "Face swap completed!", "model_used": roop.globals.model_path})
    except subprocess.CalledProcessError as e:
        print(f"[FACE_SWAP] Lỗi khi chạy lệnh face swap: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/text_to_speech/")
async def text_to_speech(text: str = Form(...), voice: str = Form(...)):
    output_audio = STATIC_DIR / "output.wav"
    
    # Đảm bảo lệnh khớp với terminal
    command = [
        "viettts", "synthesis",
        "--text", text,
        "--voice", voice,
        "--output", str(output_audio)
    ]
    
    try:
        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        return JSONResponse(content={"output_audio": "/static/output.wav"})
    except subprocess.CalledProcessError as e:
        error_message = f"Command failed with exit code {e.returncode}: {e.stderr}"
        print(error_message)
        return JSONResponse(content={"error": error_message}, status_code=500)
        
@app.post("/animate_image/")
async def animate_image():
    # Tạo thư mục Videos nếu chưa tồn tại
    VIDEO_DIR.mkdir(exist_ok=True)
    
    image_path = STATIC_DIR / "output.jpg"
    audio_path = STATIC_DIR / "output.wav"
    output_video = STATIC_DIR / "output.mp4"
    
    if not image_path.exists() or not audio_path.exists():
        return JSONResponse(content={"error": "Missing input files (image or audio)"}, status_code=400)

    # Xóa tất cả video cũ trước khi chạy
    for old_video in STATIC_DIR.glob("*.mp4"):
        os.remove(old_video)

    command = [
        "python", "inference.py",
        "--driven_audio", str(audio_path),
        "--source_image", str(image_path),
        "--still",
        "--preprocess", "full",
        "--cpu",
        "--result_dir", str(STATIC_DIR)
    ]
    
    try:
        subprocess.run(command, check=True)
        
        # Lấy video mới nhất theo tên file
        video_files = sorted(STATIC_DIR.glob("*.mp4"), reverse=True)
        latest_video = video_files[0] if video_files else None
        
        if not latest_video:
            return JSONResponse(content={"error": "No video was generated by inference.py"}, status_code=500)
        
        latest_video.rename(output_video)
        
        # Đảm bảo không ghi đè file cũ khi tạo output_web.mp4 trong thư mục Videos
        output_web_video = VIDEO_DIR / "output_web.mp4"
        count = 1
        while output_web_video.exists():
            output_web_video = VIDEO_DIR / f"output_web_{count}.mp4"
            count += 1
        
        ffmpeg_command = [
            "ffmpeg", "-i", str(output_video),
            "-y", "-vcodec", "libx264", "-acodec", "aac",
            str(output_web_video)
        ]
        subprocess.run(ffmpeg_command, check=True)
        
        return JSONResponse(content={"output_video": f"/static/Videos/{output_web_video.name}"})
    except subprocess.CalledProcessError as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/upload_image")
async def upload_image(image: UploadFile = File(...)):
    try:
        output_path = STATIC_DIR / "output.jpg"
        with output_path.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        return JSONResponse(content={"message": "Tải ảnh thành công"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
        
@app.post("/upload_voice")
async def upload_voice(voice: UploadFile = File(...)):
    try:
        output_path = STATIC_DIR / "output.wav"
        with output_path.open("wb") as buffer:
            shutil.copyfileobj(voice.file, buffer)
        return JSONResponse(content={"message": "Tải giọng nói thành công"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/generate_script/")
async def generate_script(
    text: str = Form(...),
    ai_provider: str = Form("gemini"),
    api_key: str = Form(...),
    model_name: str = Form("gemini-1.5-flash")
):
    try:
        # Validate model name
        if ai_provider == "gemini" and not model_name.startswith("gemini-"):
            model_name = "gemini-1.5-flash"
            logger.warning(f"Model name không hợp lệ, sử dụng mặc định: {model_name}")
        
        command = ["python", "chatbox.py", text]
        
        env = os.environ.copy()
        env["AI_PROVIDER"] = ai_provider
        env["API_KEY"] = api_key
        env["MODEL_NAME"] = model_name
        
        logger.info(f"Chuẩn bị chạy lệnh: {' '.join(command)}")
        logger.info(f"AI Provider: {ai_provider}, Model: {model_name}")
        
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=env
        )
        
        # Kiểm tra output
        if not result.stdout.strip():
            error_msg = "Không nhận được kết quả từ AI"
            logger.error(error_msg)
            raise HTTPException(
                status_code=500,
                detail=error_msg
            )
            
        logger.info("Tạo kịch bản thành công")
        return JSONResponse(content={"script_output": result.stdout})
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Lỗi khi tạo kịch bản: {e.stderr}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )
    except Exception as e:
        error_msg = f"Lỗi không xác định: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )

@app.post("/change_model/")
async def change_model(model_file: UploadFile = File(...)):
    print(f"[CHANGE_MODEL] Nhận yêu cầu change_model, tên tệp: {model_file.filename}, kích thước: {model_file.size}")
    try:
        if not model_file.filename:
            print("[CHANGE_MODEL] Lỗi: Không có tệp được gửi")
            raise HTTPException(status_code=400, detail="Phải cung cấp tệp")

        model_dir = Path(__file__).parent / "models"
        model_dir.mkdir(exist_ok=True)

        model_name = Path(model_file.filename).stem
        file_extension = Path(model_file.filename).suffix or '.bin'
        new_model_path = model_dir / model_file.filename

        counter = 1
        while new_model_path.exists():
            new_model_path = model_dir / f"{model_name}_{counter}{file_extension}"
            counter += 1

        print(f"[CHANGE_MODEL] Lưu tệp vào: {new_model_path}")
        try:
            with new_model_path.open("wb") as buffer:
                content = await model_file.read()
                print(f"[CHANGE_MODEL] Kích thước nội dung tệp: {len(content)} bytes")
                buffer.write(content)
        except Exception as e:
            print(f"[CHANGE_MODEL] Lỗi lưu tệp: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Lỗi khi lưu tệp: {str(e)}")

        print("[CHANGE_MODEL] Tải tệp vào face swapper")
        try:
            clear_face_swapper()
            roop.globals.model_path = str(new_model_path)
            get_face_swapper()
            print(f"[CHANGE_MODEL] Mô hình đã được tải: {roop.globals.model_path}")
            
            # Kiểm tra quyền ghi file
            config_dir = CONFIG_PATH.parent
            if not os.access(config_dir, os.W_OK):
                print(f"[CHANGE_MODEL] Không có quyền ghi vào thư mục: {config_dir}")
                raise HTTPException(status_code=500, detail=f"Không có quyền ghi vào thư mục: {config_dir}")
            
            # Lưu model_path vào file cấu hình
            with CONFIG_PATH.open("w") as f:
                json.dump({"model_path": str(new_model_path)}, f)
            print(f"[CHANGE_MODEL] Lưu model_path vào config: {new_model_path}")
            if CONFIG_PATH.exists():
                print(f"[CHANGE_MODEL] Xác nhận file config tồn tại: {CONFIG_PATH}")
            else:
                print(f"[CHANGE_MODEL] Lỗi: File config không được tạo: {CONFIG_PATH}")
        except Exception as e:
            print(f"[CHANGE_MODEL] Lỗi load tệp hoặc lưu config: {str(e)}")
            if new_model_path.exists():
                os.remove(new_model_path)
            raise HTTPException(status_code=500, detail=f"Lỗi khi load tệp hoặc lưu config: {str(e)}")

        print("[CHANGE_MODEL] Hoàn tất xử lý tệp")
        return JSONResponse(content={
            "success": True,
            "model_name": model_name,
            "model_path": str(new_model_path),
            "relative_path": f"models/{new_model_path.name}"
        })

    except HTTPException as e:
        print(f"[CHANGE_MODEL] HTTPException: {str(e)}")
        raise e
    except Exception as e:
        print(f"[CHANGE_MODEL] Lỗi trong change_model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý tệp: {str(e)}")

@app.get("/models/{model_path:path}")
async def serve_model(model_path: str):
    model_file = Path(__file__).parent / "models" / model_path
    if not model_file.exists():
        print(f"[SERVE_MODEL] Tệp không tồn tại: {model_file}")
        raise HTTPException(status_code=404, detail="Model file not found")
    print(f"[SERVE_MODEL] Phục vụ tệp: {model_file}")
    return FileResponse(model_file)

@app.post("/activate_model/")
async def activate_model(request: dict):
    try:
        model_path = request.get('model_path')
        if not model_path:
            print("[ACTIVATE_MODEL] Lỗi: Không cung cấp model_path")
            raise HTTPException(status_code=400, detail="Phải cung cấp model_path")

        full_path = Path(__file__).parent / "models" / model_path
        if not full_path.exists():
            print(f"[ACTIVATE_MODEL] Tệp không tồn tại: {full_path}")
            raise HTTPException(status_code=404, detail=f"Model file not found at: {full_path}")

        print(f"[ACTIVATE_MODEL] Kích hoạt mô hình: {full_path}")
        try:
            clear_face_swapper()
            roop.globals.model_path = str(full_path)
            get_face_swapper()
            print(f"[ACTIVATE_MODEL] Mô hình đã được tải: {roop.globals.model_path}")
            
            # Kiểm tra quyền ghi file
            config_dir = CONFIG_PATH.parent
            if not os.access(config_dir, os.W_OK):
                print(f"[ACTIVATE_MODEL] Không có quyền ghi vào thư mục: {config_dir}")
                raise HTTPException(status_code=500, detail=f"Không có quyền ghi vào thư mục: {config_dir}")
            
            # Lưu model_path vào file cấu hình
            with CONFIG_PATH.open("w") as f:
                json.dump({"model_path": str(full_path)}, f)
            print(f"[ACTIVATE_MODEL] Lưu model_path vào config: {full_path}")
            if CONFIG_PATH.exists():
                print(f"[ACTIVATE_MODEL] Xác nhận file config tồn tại: {CONFIG_PATH}")
            else:
                print(f"[ACTIVATE_MODEL] Lỗi: File config không được tạo: {CONFIG_PATH}")
        except Exception as e:
            print(f"[ACTIVATE_MODEL] Lỗi kích hoạt mô hình hoặc lưu config: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Lỗi khi kích hoạt mô hình hoặc lưu config: {str(e)}")

        print("[ACTIVATE_MODEL] Kích hoạt mô hình thành công")
        return JSONResponse(content={
            "success": True,
            "message": "Kích hoạt mô hình thành công",
            "model_path": str(full_path)
        })

    except HTTPException as e:
        print(f"[ACTIVATE_MODEL] HTTPException: {str(e)}")
        raise e
    except Exception as e:
        print(f"[ACTIVATE_MODEL] Lỗi trong activate_model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi kích hoạt mô hình: {str(e)}")
        
@app.post("/delete_model/")
async def delete_model(request: dict):
    try:
        model_path = request.get('model_path')
        if not model_path:
            print("[DELETE_MODEL] Lỗi: Không cung cấp model_path")
            raise HTTPException(status_code=400, detail="Phải cung cấp model_path")

        full_path = Path(__file__).parent / "models" / model_path
        if not full_path.exists():
            print(f"[DELETE_MODEL] Tệp không tồn tại: {full_path}")
            raise HTTPException(status_code=404, detail=f"Model file not found at: {full_path}")

        print(f"[DELETE_MODEL] Xóa mô hình: {full_path}")
        try:
            # Xóa file mô hình
            os.remove(full_path)
            print(f"[DELETE_MODEL] Đã xóa file: {full_path}")

            # Kiểm tra nếu mô hình đang được kích hoạt
            current_model_path = load_model_path()
            if str(full_path) == current_model_path:
                # Thiết lập về mô hình mặc định
                default_model_path = str(Path(__file__).parent / "models" / "inswapper_128.onnx")
                roop.globals.model_path = default_model_path
                try:
                    with CONFIG_PATH.open("w") as f:
                        json.dump({"model_path": default_model_path}, f)
                    print(f"[DELETE_MODEL] Cập nhật model_config.json về mặc định: {default_model_path}")
                except Exception as e:
                    print(f"[DELETE_MODEL] Lỗi lưu config: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Lỗi khi lưu config: {str(e)}")
                clear_face_swapper()
                get_face_swapper()
                print(f"[DELETE_MODEL] Mô hình mặc định đã được tải: {roop.globals.model_path}")

        except Exception as e:
            print(f"[DELETE_MODEL] Lỗi xóa mô hình: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Lỗi khi xóa mô hình: {str(e)}")

        print("[DELETE_MODEL] Xóa mô hình thành công")
        return JSONResponse(content={
            "success": True,
            "message": "Xóa mô hình thành công",
            "model_path": str(full_path)
        })

    except HTTPException as e:
        print(f"[DELETE_MODEL] HTTPException: {str(e)}")
        raise e
    except Exception as e:
        print(f"[DELETE_MODEL] Lỗi trong delete_model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi xóa mô hình: {str(e)}")

@app.post("/face_swap_ai/")
async def face_swap_ai(source: UploadFile = File(...), target: UploadFile = File(...)):
    # Định nghĩa thư mục lưu trữ
    source_path = UPLOAD_DIR / source.filename
    target_path = UPLOAD_DIR / target.filename
    images_dir = STATIC_DIR / "Images"
    images_dir.mkdir(exist_ok=True)  # Tạo thư mục Images nếu chưa tồn tại
    
    # Tạo tên file đầu ra với số thứ tự để tránh ghi đè
    output_base = "output"
    file_extension = ".jpg"
    output_path = images_dir / f"{output_base}{file_extension}"
    counter = 1
    while output_path.exists():
        output_path = images_dir / f"{output_base}_{counter}{file_extension}"
        counter += 1
    
    # Lưu file nguồn và mục tiêu
    with source_path.open("wb") as buffer:
        shutil.copyfileobj(source.file, buffer)
    with target_path.open("wb") as buffer:
        shutil.copyfileobj(target.file, buffer)
    
    # Kiểm tra và thiết lập model_path nếu cần
    if not hasattr(roop.globals, 'model_path') or not roop.globals.model_path or not os.path.exists(roop.globals.model_path):
        roop.globals.model_path = str(Path(__file__).parent / "models" / "inswapper_128.onnx")
        print(f"[FACE_SWAP_AI] Thiết lập model_path mặc định: {roop.globals.model_path}")
    
    print(f"[FACE_SWAP_AI] Thực hiện face swap với mô hình: {roop.globals.model_path}")
    command = [
        "python", "run.py",
        "-s", str(source_path),
        "-t", str(target_path),
        "-o", str(output_path),
        "--frame-processor", "face_swapper",
        "--model-path", roop.globals.model_path
    ]
    
    try:
        subprocess.run(command, check=True)
        return JSONResponse(content={
            "message": "Face swap completed!",
            "model_used": roop.globals.model_path,
            "output_image": f"/static/Images/{output_path.name}"
        })
    except subprocess.CalledProcessError as e:
        print(f"[FACE_SWAP_AI] Lỗi khi chạy lệnh face swap: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        # Dọn dẹp file tạm
        if source_path.exists():
            os.remove(source_path)
        if target_path.exists():
            os.remove(target_path)

def load_model_path():
    default_model_path = str(Path(__file__).parent / "models" / "inswapper_128.onnx")
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r") as f:
                config = json.load(f)
                model_path = config.get("model_path")
                if model_path and os.path.exists(model_path):
                    print(f"[INIT] Loaded model path from config: {model_path}")
                    return model_path
                else:
                    print(f"[INIT] Invalid model path in config: {model_path}, using default: {default_model_path}")
        except Exception as e:
            print(f"[INIT] Error reading model config: {str(e)}, using default: {default_model_path}")
    else:
        print(f"[INIT] Model config not found at {CONFIG_PATH}, using default: {default_model_path}")
    return default_model_path

# Thiết lập model_path khi khởi động
roop.globals.model_path = load_model_path()
print(f"[INIT] Initial model path set: {roop.globals.model_path}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=55021)	
