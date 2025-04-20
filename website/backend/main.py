# FastAPI backend for videogen

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import subprocess
import torch
import shutil
from pathlib import Path

from model_loader import load_model
from video_gen_service import VideoGenService

app = FastAPI()

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model and Manim integration ---
def wrap_code_in_manim_scene(code_str, scene_name="GeneratedScene"):
    """Wrap code in a basic Manim scene structure and return the full script as a string."""
    # Check if the code already has a scene class
    if f"class {scene_name}(Scene)" in code_str:
        return code_str
    
    manim_template = f"""
from manim import *

class {scene_name}(Scene):
    def construct(self):
{indent_code(code_str, 8)}
"""
    return manim_template

def indent_code(code, num_spaces):
    indentation = ' ' * num_spaces
    return '\n'.join(indentation + line if line.strip() else '' for line in code.splitlines())

def save_code_to_file(code_str, filename):
    with open(filename, 'w') as f:
        f.write(code_str)

def run_manim(filename, scene_name):
    cmd = ["manim", "-ql", filename, scene_name]
    subprocess.run(cmd, check=True)

def find_latest_video(scene_name, base_dir="media/videos"):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith(scene_name) and file.endswith(".mp4"):
                return os.path.join(root, file)
    return None

# --- Model data paths ---
DATA_DIR = "/root/deepseek-video-gen/website/data"
MODEL_FILES = {
    "base-model": {
        "code": os.path.join(DATA_DIR, "base.py"),
        "video": None  # No video for base model
    },
    "SFT": {
        "code": os.path.join(DATA_DIR, "sft.py"),
        "video": os.path.join(DATA_DIR, "sft.mp4")
    },
    "GRPO": {
        "code": os.path.join(DATA_DIR, "grpo.py"),
        "video": os.path.join(DATA_DIR, "grpo.mp4")
    },
    # Default fallback
    "default": {
        "code": os.path.join(DATA_DIR, "pythogoras_theorem.py"),
        "video": os.path.join(DATA_DIR, "PythagoreanTheoremScene.mp4")
    }
}

# --- FastAPI App ---
SCENE_NAME = "GeneratedScene"
PY_FILE = "generated_scene.py"
FALLBACK_VIDEO = "/root/deepseek-video-gen/website/data/PythagoreanTheoremScene.mp4"

# Service for handling video generation
video_service = None

@app.on_event("startup")
def startup_event():
    global video_service
    video_service = VideoGenService(load_model)

class GenerateRequest(BaseModel):
    input_text: str
    model_type: str = "base-model"  # Changed default to base-model

@app.post("/generate")
def generate_video(req: GenerateRequest):
    global video_service
    if video_service is None:
        raise HTTPException(status_code=500, detail="Service not initialized.")
    
    try:
        # Check if we're using a predefined model type
        model_type = req.model_type
        if model_type not in MODEL_FILES:
            model_type = "base-model"  # Changed default to base-model
        
        model_data = MODEL_FILES[model_type]
        
        # For base-model, we return code but no video
        if model_type == "base-model":
            # Read the code file
            if os.path.exists(model_data["code"]):
                with open(model_data["code"], "r") as f:
                    code = f.read()
                
                # Save the code for reference
                code_file_path = os.path.join(video_service.video_dir, "generated_code.py")
                with open(code_file_path, "w") as f:
                    f.write(code)
                
                return {
                    "message": "Base model does not generate videos",
                    "video_error": "Error generating Video",
                    "code": code
                }
            else:
                raise HTTPException(status_code=404, detail="Base model code file not found")
        
        # For SFT and GRPO, return both code and video
        elif model_type in ["SFT", "GRPO"]:
            # Read the code file
            if not os.path.exists(model_data["code"]):
                raise HTTPException(status_code=404, detail=f"{model_type} code file not found")
            
            with open(model_data["code"], "r") as f:
                code = f.read()
            
            # Save the code for reference
            code_file_path = os.path.join(video_service.video_dir, "generated_code.py")
            with open(code_file_path, "w") as f:
                f.write(code)
            
            # Copy the video to our video directory
            if not os.path.exists(model_data["video"]):
                raise HTTPException(status_code=404, detail=f"{model_type} video file not found")
            
            video_dest = os.path.join(video_service.video_dir, f"{model_type}.mp4")
            shutil.copy(model_data["video"], video_dest)
            video_service.latest_video = Path(video_dest)
            
            return {
                "message": f"Using {model_type} model",
                "video_path": video_dest,
                "code": code
            }
        
        # If we're using the default model or any other option, use the real model or fallback
        else:
            # Generate code using the model
            code = video_service.model.generate(req.input_text)
            
            # Save the generated code for reference
            code_file_path = os.path.join(video_service.video_dir, "generated_code.py")
            with open(code_file_path, "w") as f:
                f.write(code)
            
            # Check if we should use the fallback video
            use_fallback = os.environ.get("USE_FALLBACK", "false").lower() == "true"
            
            if use_fallback:
                # Copy the fallback video to the video directory
                if os.path.exists(FALLBACK_VIDEO):
                    fallback_dest = os.path.join(video_service.video_dir, "fallback_video.mp4")
                    shutil.copy(FALLBACK_VIDEO, fallback_dest)
                    video_service.latest_video = Path(fallback_dest)
                    return {"message": "Using fallback video", "video_path": str(fallback_dest), "code": code}
            
            # Wrap the code in a Manim scene if needed
            script = wrap_code_in_manim_scene(code, SCENE_NAME)
            save_code_to_file(script, PY_FILE)
            
            # Run Manim to generate the video
            run_manim(PY_FILE, SCENE_NAME)
            
            # Find the generated video
            video_file = find_latest_video(SCENE_NAME)
            if not video_file:
                raise HTTPException(status_code=500, detail="Video not generated.")
            
            # Copy the video to our video directory
            video_dest = os.path.join(video_service.video_dir, f"{SCENE_NAME}.mp4")
            shutil.copy(video_file, video_dest)
            video_service.latest_video = Path(video_dest)
            
            return {"message": "Video generated", "video_path": video_dest, "code": code}
    except Exception as e:
        # If anything fails, use the fallback video unless it's the base model
        if req.model_type == "base-model":
            # For base model, return only code with an error for video
            try:
                with open(MODEL_FILES["base-model"]["code"], "r") as f:
                    code = f.read()
                return {
                    "message": f"Error: {str(e)}",
                    "video_error": "Error generating Video",
                    "code": code
                }
            except:
                raise HTTPException(status_code=500, detail=str(e))
        elif os.path.exists(FALLBACK_VIDEO):
            fallback_dest = os.path.join(video_service.video_dir, "error_fallback.mp4")
            shutil.copy(FALLBACK_VIDEO, fallback_dest)
            video_service.latest_video = Path(fallback_dest)
            return {
                "message": f"Error: {str(e)}. Using fallback video.",
                "video_path": str(fallback_dest),
                "code": "# Error occurred during code generation\n" + str(e) if 'code' not in locals() else code
            }
        else:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/video")
def get_video():
    global video_service
    if video_service is None:
        raise HTTPException(status_code=500, detail="Service not initialized.")
    
    video_path, video_name = video_service.get_latest_video()
    
    # If no video is available, try to use the fallback
    if video_path is None and os.path.exists(FALLBACK_VIDEO):
        fallback_dest = os.path.join(video_service.video_dir, "fallback_video.mp4")
        shutil.copy(FALLBACK_VIDEO, fallback_dest)
        video_service.latest_video = Path(fallback_dest)
        video_path, video_name = video_service.get_latest_video()
    
    if video_path is None or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="No video found")
    
    return FileResponse(str(video_path), media_type="video/mp4", filename=video_name)

@app.get("/code")
def get_code():
    """Get the latest generated code"""
    global video_service
    if video_service is None:
        raise HTTPException(status_code=500, detail="Service not initialized.")
    
    code_path = os.path.join(video_service.video_dir, "generated_code.py")
    
    if not os.path.exists(code_path):
        # If no code file exists, return the fallback code
        fallback_code_path = "/root/deepseek-video-gen/website/data/pythogoras_theorem.py"
        if os.path.exists(fallback_code_path):
            with open(fallback_code_path, 'r') as f:
                code = f.read()
            return {"code": code}
        else:
            raise HTTPException(status_code=404, detail="No code found")
    
    with open(code_path, 'r') as f:
        code = f.read()
    
    return {"code": code}

@app.get("/status")
def status():
    return JSONResponse({"status": "ok"})
