import base64
import os
from typing import Callable
from threading import Lock
from secrets import compare_digest
from io import BytesIO

from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
from PIL import Image
import tempfile

# Import WebUI related modules and add try-except to be compatible with non-WebUI environments
try:
    from modules import shared
    from modules.call_queue import queue_lock as webui_queue_lock
    IN_WEBUI = True
except ImportError:
    IN_WEBUI = False
    shared = type('Shared', (), {'cmd_opts': type('CmdOpts', (), {'api_auth': None})()})()
    webui_queue_lock = None

from backend_wanvideo.inferrence import generate_t2v, generate_i2v, generate_v2v, get_model_files

# Pydantic model defines request and response formats and adds Field descriptions
class Text2VideoRequest(BaseModel):
    dit_models: list[str] = Field(..., description="DIT model file list (multiple selections possible, merged into one model)")
    t5_model: str = Field(..., description="T5 model file name")
    vae_model: str = Field(..., description="VAE model file name")
    prompt: str = Field("", description="Positive prompt words, describing the video content, can contain <lora:model file name:weight>")
    negative_prompt: str = Field("", description="Negative prompt words, used to exclude unnecessary elements")
    num_inferrence_steps: int = Field(15, ge=1, le=100, description="Number of inferrence steps, affects generation quality and speed")
    seed: int = Field(-1, description="Random seed, -1 means random")
    height: int = Field(480, ge=256, le=1080, multiple_of=8, description="Video height (pixels)")
    width: int = Field(832, ge=256, le=1920, multiple_of=8, description="Video width (pixels)")
    num_frames: int = Field(81, ge=1, description="Total number of video frames")
    cfg_scale: float = Field(5.0, ge=0.0, le=20.0, description="CFG Scale, controls the degree of compliance of prompt words")
    sigma_shift: float = Field(5.0, ge=0.0, description="Sigma Shift, controls the diffusion process")
    tea_cache_l1_thresh: float = Field(0.07, ge=0.0, le=1.0, description="TeaCache L1 threshold, the larger the faster but the lower the quality")
    tea_cache_model_id: str = Field("Wan2.1-T2V-1.3B", description="TeaCache model ID", example="Wan2.1-T2V-1.3B")
    image_encoder_model: str | None = Field(None, description="Image Encoder model file name (optional)")
    fps: int = Field(15, ge=1, le=60, description="Output frame rate (FPS)")
    denoising_strength: float = Field(1.0, ge=0.0, le=1.0, description="Noise reduction strength")
    rand_device: str = Field("cpu", description="Random device: cpu or cuda")
    tiled: bool = Field(True, description="Whether to use Tiled processing")
    tile_size_x: int = Field(30, ge=1, description="Tile Size X")
    tile_size_y: int = Field(52, ge=1, description="Tile Size Y")
    tile_stride_x: int = Field(15, ge=1, description="Tile Stride X")
    tile_stride_y: int = Field(26, ge=1, description="Tile Stride Y")
    torch_dtype: str = Field("bfloat16", description="DIT/T5/VAE data types: float16, bfloat16, float8_e4m3fn")
    image_encoder_torch_dtype: str = Field("float32", description="Image Encoder data type: float16, float32, bfloat16")
    use_usp: bool = Field(False, description="Whether to use USP (Unified Sequence Parallel)")
    enable_num_persistent: bool = Field(False, description="Whether to enable video memory optimization parameters")
    num_persistent_param_in_dit: int | None = Field(None, description="Video memory management parameter value, the smaller the value, the less video memory required")

class Image2VideoRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded string of the first frame")
    dit_models: list[str] = Field(..., description="DIT model file list (multiple selections possible, merged into one model)")
    t5_model: str = Field(..., description="T5 model file name")
    vae_model: str = Field(..., description="VAE model file name")
    image_encoder_model: str = Field(..., description="Image Encoder model file name")
    end_image: str | None = Field(None, description="Base64 encoded string of the last frame image (optional)")
    prompt: str = Field("", description="Positive prompt words, describing the video content, can contain <lora:model file name:weight>")
    negative_prompt: str = Field("", description="Negative prompt words, used to exclude unnecessary elements")
    num_inferrence_steps: int = Field(15, ge=1, le=100, description="Number of inferrence steps, affects generation quality and speed")
    seed: int = Field(-1, description="Random seed, -1 means random")
    height: int = Field(480, ge=256, le=1080, multiple_of=8, description="Video height (pixels)")
    width: int = Field(832, ge=256, le=1920, multiple_of=8, description="Video width (pixels)")
    num_frames: int = Field(81, ge=1, description="Total number of video frames")
    cfg_scale: float = Field(5.0, ge=0.0, le=20.0, description="CFG Scale, controls the degree of compliance of prompt words")
    sigma_shift: float = Field(5.0, ge=0.0, description="Sigma Shift, controls the diffusion process")
    tea_cache_l1_thresh: float = Field(0.19, ge=0.0, le=1.0, description="TeaCache L1 threshold, the larger the faster but the lower the quality")
    tea_cache_model_id: str = Field("Wan2.1-I2V-14B-480P", description="TeaCache model ID", example="Wan2.1-I2V-14B-480P")
    fps: int = Field(15, ge=1, le=60, description="Output frame rate (FPS)")
    denoising_strength: float = Field(1.0, ge=0.0, le=1.0, description="Noise reduction strength")
    rand_device: str = Field("cpu", description="Random device: cpu or cuda")
    tiled: bool = Field(True, description="Whether to use Tiled processing")
    tile_size_x: int = Field(30, ge=1, description="Tile Size X")
    tile_size_y: int = Field(52, ge=1, description="Tile Size Y")
    tile_stride_x: int = Field(15, ge=1, description="Tile Stride X")
    tile_stride_y: int = Field(26, ge=1, description="Tile Stride Y")
    torch_dtype: str = Field("bfloat16", description="DIT/T5/VAE data types: float16, bfloat16, float8_e4m3fn")
    image_encoder_torch_dtype: str = Field("float32", description="Image Encoder data type: float16, float32, bfloat16")
    use_usp: bool = Field(False, description="Whether to use USP (Unified Sequence Parallel)")
    enable_num_persistent: bool = Field(False, description="Whether to enable video memory optimization parameters")
    num_persistent_param_in_dit: int | None = Field(None, description="Video memory management parameter value, the smaller the value, the less video memory required")

class Video2VideoRequest(BaseModel):
    video: str = Field(..., description="Base64 encoded string of the original video")
    dit_models: list[str] = Field(..., description="DIT model file list (multiple selections possible, merged into one model)")
    t5_model: str = Field(..., description="T5 model file name")
    vae_model: str = Field(..., description="VAE model file name")
    prompt: str = Field("", description="Positive prompt words, describing the video content, can contain <lora:model file name:weight>")
    negative_prompt: str = Field("", description="Negative prompt words, used to exclude unnecessary elements")
    control_video: str | None = Field(None, description="Base64 encoded string of control video (optional)")
    num_inferrence_steps: int = Field(15, ge=1, le=100, description="Number of inferrence steps, affects generation quality and speed")
    seed: int = Field(-1, description="Random seed, -1 means random")
    height: int = Field(480, ge=256, le=1080, multiple_of=8, description="Video height (pixels)")
    width: int = Field(832, ge=256, le=1920, multiple_of=8, description="Video width (pixels)")
    num_frames: int = Field(81, ge=1, description="Total number of video frames")
    cfg_scale: float = Field(5.0, ge=0.0, le=20.0, description="CFG Scale, controls the degree of compliance of prompt words")
    sigma_shift: float = Field(5.0, ge=0.0, description="Sigma Shift, controls the diffusion process")
    image_encoder_model: str | None = Field(None, description="Image Encoder model file name (optional)")
    fps: int = Field(15, ge=1, le=60, description="Output frame rate (FPS)")
    denoising_strength: float = Field(0.7, ge=0.0, le=1.0, description="Noise reduction strength")
    rand_device: str = Field("cpu", description="Random device: cpu or cuda")
    tiled: bool = Field(True, description="Whether to use Tiled processing")
    tile_size_x: int = Field(30, ge=1, description="Tile Size X")
    tile_size_y: int = Field(52, ge=1, description="Tile Size Y")
    tile_stride_x: int = Field(15, ge=1, description="Tile Stride X")
    tile_stride_y: int = Field(26, ge=1, description="Tile Stride Y")
    torch_dtype: str = Field("bfloat16", description="DIT/T5/VAE data types: float16, bfloat16, float8_e4m3fn")
    image_encoder_torch_dtype: str = Field("float32", description="Image Encoder data type: float16, float32, bfloat16")
    use_usp: bool = Field(False, description="Whether to use USP (Unified Sequence Parallel)")
    enable_num_persistent: bool = Field(False, description="Whether to enable video memory optimization parameters")
    num_persistent_param_in_dit: int | None = Field(None, description="Video memory management parameter value, the smaller the value, the less video memory required")

class VideoResponse(BaseModel):
    video: str = Field(..., description="Generated video, Base64-encoded MP4 file")
    info: str = Field(..., description="Generation information, including hardware information, time consumption, resolution, etc.")

class ModelsResponse(BaseModel):
    dit_models: list[str] = Field(..., description="Available DIT model file list")
    t5_models: list[str] = Field(..., description="A list of available T5 model files")
    vae_models: list[str] = Field(..., description="Available VAE model file list")
    image_encoder_models: list[str] = Field(..., description="Available Image Encoder model file list")
    lora_models: list[str] = Field(..., description="Available LoRA model file list")

class Api:
    def __init__(self, app: FastAPI, queue_lock: Lock = None, prefix: str = "/wanvideo/v1"):
        self.app = app
        self.queue_lock = queue_lock or Lock() # Use the passed in queue_lock or default Lock
        self.prefix = prefix
        self.credentials = {}

        # Load the WebUI API authentication configuration (WebUI environment only)
        if IN_WEBUI and shared.cmd_opts.api_auth:
            for auth in shared.cmd_opts.api_auth.split(","):
                user, password = auth.split(":")
                self.credentials[user] = password

        # Registering API Routes
        self.add_api_route(
            "t2v",
            self.endpoint_text2video,
            methods=["POST"],
            response_model=VideoResponse,
            summary="Generate video from text prompt",
            description="Creates a video based on a text prompt using the WanVideo model."
        )
        self.add_api_route(
            "i2v",
            self.endpoint_image2video,
            methods=["POST"],
            response_model=VideoResponse,
            summary="Generate video from an initial image",
            description="Creates a video starting from an initial image (and optional end image) with a text prompt."
        )
        self.add_api_route(
            "v2v",
            self.endpoint_video2video,
            methods=["POST"],
            response_model=VideoResponse,
            summary="Generate video from an input video",
            description="Creates a video by transforming an input video (with optional control video) using a text prompt."
        )
        self.add_api_route(
            "models",
            self.endpoint_models,
            methods=["GET"],
            response_model=ModelsResponse,
            summary="List available models",
            description="Returns a list of available DIT, T5, VAE, Image Encoder, and LoRA model files."
        )

    def auth(self, creds: HTTPBasicCredentials = Depends(HTTPBasic())):
        if not self.credentials:
            return True
        if creds.username in self.credentials:
            if compare_digest(creds.password, self.credentials[creds.username]):
                return True
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"}
        )

    def add_api_route(self, path: str, endpoint: Callable, **kwargs):
        path = f"{self.prefix}/{path}" if self.prefix else path
        if self.credentials:
            return self.app.add_api_route(path, endpoint, dependencies=[Depends(self.auth)], **kwargs)
        return self.app.add_api_route(path, endpoint, **kwargs)

    def decode_base64_image(self, base64_str: str) -> str:
        """Decode the Base64 image and save it as a temporary file, returning the file path"""
        try:
            img_data = base64.b64decode(base64_str)
            img = Image.open(BytesIO(img_data)).convert("RGB")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                img.save(tmp.name)
                return tmp.name
        except Exception as e:
            raise HTTPException(400, f"Invalid image Base64 data: {str(e)}")

    def decode_base64_video(self, base64_str: str) -> str:
        """Decode the Base64 video and save it as a temporary file, returning the file path"""
        try:
            video_data = base64.b64decode(base64_str)
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(video_data)
                return tmp.name
        except Exception as e:
            raise HTTPException(400, f"Invalid video Base64 data: {str(e)}")

    def encode_video_to_base64(self, video_path: str) -> str:
        """Encode the video file into a Base64 string"""
        try:
            with open(video_path, "rb") as f:
                video_data = f.read()
            return base64.b64encode(video_data).decode("utf-8")
        except Exception as e:
            raise HTTPException(500, f"Failed to encode video to Base64: {str(e)}")

    def endpoint_text2video(self, req: Text2VideoRequest):
        with self.queue_lock:
            output_path, info = generate_t2v(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                num_inferrence_steps=req.num_inferrence_steps,
                seed=req.seed,
                height=req.height,
                width=req.width,
                num_frames=req.num_frames,
                cfg_scale=req.cfg_scale,
                sigma_shift=req.sigma_shift,
                tea_cache_l1_thresh=req.tea_cache_l1_thresh,
                tea_cache_model_id=req.tea_cache_model_id,
                dit_models=req.dit_models,
                t5_model=req.t5_model,
                vae_model=req.vae_model,
                image_encoder_model=req.image_encoder_model,
                fps=req.fps,
                denoising_strength=req.denoising_strength,
                rand_device=req.rand_device,
                tiled=req.tiled,
                tile_size_x=req.tile_size_x,
                tile_size_y=req.tile_size_y,
                tile_stride_x=req.tile_stride_x,
                tile_stride_y=req.tile_stride_y,
                torch_dtype=req.torch_dtype,
                image_encoder_torch_dtype=req.image_encoder_torch_dtype,
                use_usp=req.use_usp,
                enable_num_persistent=req.enable_num_persistent,
                num_persistent_param_in_dit=req.num_persistent_param_in_dit
            )
            if output_path is None:
                raise HTTPException(500, f"Video generation failed: {info}")
            video_base64 = self.encode_video_to_base64(output_path)
            os.remove(output_path) # Clean up temporary files
            return VideoResponse(video=video_base64, info=info)

    def endpoint_image2video(self, req: Image2VideoRequest):
        image_path = self.decode_base64_image(req.image)
        end_image_path = self.decode_base64_image(req.end_image) if req.end_image else None
        with self.queue_lock:
            output_path, info = generate_i2v(
                image=image_path,
                end_image=end_image_path,
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                num_inferrence_steps=req.num_inferrence_steps,
                seed=req.seed,
                height=req.height,
                width=req.width,
                num_frames=req.num_frames,
                cfg_scale=req.cfg_scale,
                sigma_shift=req.sigma_shift,
                tea_cache_l1_thresh=req.tea_cache_l1_thresh,
                tea_cache_model_id=req.tea_cache_model_id,
                dit_models=req.dit_models,
                t5_model=req.t5_model,
                vae_model=req.vae_model,
                image_encoder_model=req.image_encoder_model,
                fps=req.fps,
                denoising_strength=req.denoising_strength,
                rand_device=req.rand_device,
                tiled=req.tiled,
                tile_size_x=req.tile_size_x,
                tile_size_y=req.tile_size_y,
                tile_stride_x=req.tile_stride_x,
                tile_stride_y=req.tile_stride_y,
                torch_dtype=req.torch_dtype,
                image_encoder_torch_dtype=req.image_encoder_torch_dtype,
                use_usp=req.use_usp,
                enable_num_persistent=req.enable_num_persistent,
                num_persistent_param_in_dit=req.num_persistent_param_in_dit
            )
            os.remove(image_path)
            if end_image_path:
                os.remove(end_image_path)
            if output_path is None:
                raise HTTPException(500, f"Video generation failed: {info}")
            video_base64 = self.encode_video_to_base64(output_path)
            os.remove(output_path)
            return VideoResponse(video=video_base64, info=info)

    def endpoint_video2video(self, req: Video2VideoRequest):
        video_path = self.decode_base64_video(req.video)
        control_video_path = self.decode_base64_video(req.control_video) if req.control_video else None
        with self.queue_lock:
            output_path, info = generate_v2v(
                video=video_path,
                control_video=control_video_path,
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                num_inferrence_steps=req.num_inferrence_steps,
                seed=req.seed,
                height=req.height,
                width=req.width,
                num_frames=req.num_frames,
                cfg_scale=req.cfg_scale,
                sigma_shift=req.sigma_shift,
                dit_models=req.dit_models,
                t5_model=req.t5_model,
                vae_model=req.vae_model,
                image_encoder_model=req.image_encoder_model,
                fps=req.fps,
                denoising_strength=req.denoising_strength,
                rand_device=req.rand_device,
                tiled=req.tiled,
                tile_size_x=req.tile_size_x,
                tile_size_y=req.tile_size_y,
                tile_stride_x=req.tile_stride_x,
                tile_stride_y=req.tile_stride_y,
                torch_dtype=req.torch_dtype,
                image_encoder_torch_dtype=req.image_encoder_torch_dtype,
                use_usp=req.use_usp,
                enable_num_persistent=req.enable_num_persistent,
                num_persistent_param_in_dit=req.num_persistent_param_in_dit
            )
            os.remove(video_path)
            if control_video_path:
                os.remove(control_video_path)
            if output_path is None:
                raise HTTPException(500, f"Video generation failed: {info}")
            video_base64 = self.encode_video_to_base64(output_path)
            os.remove(output_path)
            return VideoResponse(video=video_base64, info=info)

    def endpoint_models(self):
        base_dir = "models/wan2.1"
        return ModelsResponse(
            dit_models=get_model_files(os.path.join(base_dir, "dit")),
            t5_models=get_model_files(os.path.join(base_dir, "t5")),
            vae_models=get_model_files(os.path.join(base_dir, "vae")),
            image_encoder_models=get_model_files(os.path.join(base_dir, "image_encoder")),
            lora_models=get_model_files(os.path.join(base_dir, "lora"))
        )

def on_app_started(_: None, app: FastAPI):
    # Using webui_queue_lock in WebUI environment
    queue_lock = webui_queue_lock if IN_WEBUI else Lock()
    Api(app, queue_lock, "/wanvideo/v1")
