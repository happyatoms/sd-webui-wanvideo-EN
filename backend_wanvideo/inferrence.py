# F:\sd-reforge\webui\extensions\sd-webui-wanvideo\scripts\generation.py
import gradio as gr
import torch
import time
import psutil
import os
import numpy as np
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from PIL import Image
from tqdm import tqdm
import random
import logging
import re
try:
    from modules import script_callbacks, shared
    IN_WEBUI = True
except ImportError:
    IN_WEBUI = False
    shared = type('Shared', (), {'opts': type('Opts', (), {'outdir_samples': '', 'outdir_txt2img_samples': ''})})()

# Get hardware information
def get_hardware_info():
    info = ""
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
            info += f"GPU: {gpu_name}\nVRAM: {total_vram}GB\n"
        else:
            info += "GPU: not available\n"
        info += f"CPU: {psutil.cpu_count(logical=False)} physical cores / {psutil.cpu_count(logical=True)} logical cores\n"
        info += f"Memory: {psutil.virtual_memory().total // (1024 ** 3)}GB\n"
    except Exception as e:
        info += f"Failed to obtain hardware information: {str(e)}"
    return info

# Get a list of model files in the specified directory
def get_model_files(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return ["No model file"]
    files = [
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and not f.endswith('.txt') and not f.endswith('.json')
    ]
    return files if files else ["No model file"]

# Extract LoRA information from the prompt word
def extract_lora_from_prompt(prompt):
    lora_pattern = r"<lora:([^:]+):([\d\.]+)>"
    matches = re.findall(lora_pattern, prompt)
    loras = [(name, float(weight)) for name, weight in matches]
    cleaned_prompt = re.sub(lora_pattern, "", prompt).strip()
    return loras, cleaned_prompt

# Load the model and LoRA
def load_models(dit_models, t5_model, vae_model, image_encoder_model=None, lora_prompt="",
                torch_dtype="bfloat16", image_encoder_torch_dtype="float32", use_usp=False,
                num_persistent_param_in_dit=None):
    # Define model directory
    base_dir = "models/wan2.1"
    dit_dir = os.path.join(base_dir, "dit")
    t5_dir = os.path.join(base_dir, "t5")
    vae_dir = os.path.join(base_dir, "vae")
    lora_dir = os.path.join(base_dir, "lora")
    image_encoder_dir = os.path.join(base_dir, "image_encoder") if image_encoder_model else None
    
    # Automatically create directories
    os.makedirs(dit_dir, exist_ok=True)
    os.makedirs(t5_dir, exist_ok=True)
    os.makedirs(vae_dir, exist_ok=True)
    os.makedirs(lora_dir, exist_ok=True)
    if image_encoder_dir:
        os.makedirs(image_encoder_dir, exist_ok=True)
    
    # Log directory information for debugging
    logging.info(f"DIT model directory: {os.path.abspath(dit_dir)}")
    logging.info(f"T5 model directory: {os.path.abspath(t5_dir)}")
    logging.info(f"VAE model directory: {os.path.abspath(vae_dir)}")
    logging.info(f"LoRA model directory: {os.path.abspath(lora_dir)}")
    if image_encoder_dir:
        logging.info(f"Image Encoder model directory: {os.path.abspath(image_encoder_dir)}")
    
    # Check if the model file exists
    if not dit_models or "No model file" in dit_models or t5_model == "No model file" or vae_model == "No model file":
        raise Exception("Please make sure all model folders have valid model files: DIT, T5 and VAE models cannot be empty")
    if image_encoder_model in ["No model file", "None"]:
        image_encoder_model = None
    
    # Treat multiple DIT model files as a whole
    dit_model_paths = [os.path.join(dit_dir, dit_model) for dit_model in dit_models if dit_model != "no model file"]
    if not dit_model_paths:
        raise Exception("No valid DIT model file selected")
    
    # Organize model_list, DIT model as a nested list
    model_list = [
        dit_model_paths, # Multiple DIT files merged and loaded
        os.path.join(t5_dir, t5_model),
        os.path.join(vae_dir, vae_model)
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Support FP16, BF16 and FP8
    if torch_dtype == "float16":
        torch_dtype = torch.float16
    elif torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float8_e4m3fn
    
    # Data types supported by Image Encoder
    if image_encoder_torch_dtype == "float16":
        image_encoder_torch_dtype = torch.float16
    elif image_encoder_torch_dtype == "float32":
        image_encoder_torch_dtype = torch.float32
    else:
        image_encoder_torch_dtype = torch.bfloat16
    
    model_manager = ModelManager(device="cpu", torch_dtype=torch_dtype)
    
    # Check file path
    for item in model_list:
        if isinstance(item, list):
            for path in item:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"DIT model file {path} does not exist, please check the path")
        elif not os.path.exists(item):
            raise FileNotFoundError(f"Model file {item} does not exist, please check the path")
    
    # Load Image Encoder (if it exists)
    if image_encoder_model:
        image_encoder_path = os.path.join(image_encoder_dir, image_encoder_model)
        if not os.path.exists(image_encoder_path):
            raise FileNotFoundError(f"Image Encoder file {image_encoder_path} does not exist, please check the path")
        logging.info(f"Load Image Encoder: {image_encoder_path} (using {image_encoder_torch_dtype})")
        model_manager.load_models([image_encoder_path], torch_dtype=image_encoder_torch_dtype)
        model_list.insert(0, image_encoder_path)
    
    # Load the base model
    logging.info(f"Start loading base model: {model_list} (using {torch_dtype})")
    model_manager.load_models(model_list, torch_dtype=torch_dtype)
    logging.info(f"Basic model loading completed: {model_manager.model_name if model_manager.model_name else 'No model was identified'}")
    
    # Extract LoRA information from the prompt word and load it
    loras, _ = extract_lora_from_prompt(lora_prompt)
    loaded_loras = {}
    if loras:
        for lora_name, lora_weight in loras:
            lora_path = os.path.join(lora_dir, lora_name)
            if not os.path.exists(lora_path):
                logging.warning(f"LoRA file {lora_path} does not exist, skip loading")
                continue
            logging.info(f"Load LoRA: {lora_path} (alpha={lora_weight})")
            model_manager.load_lora(lora_path, lora_alpha=lora_weight)
            loaded_loras[lora_name] = lora_weight
    
    # Check USP environment
    if use_usp and not torch.distributed.is_initialized():
        logging.warning("USP enablement failed: distributed environment is not initialized, USP will be disabled")
        use_usp = False
    
    # Create a pipeline
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch_dtype, device=device, use_usp=use_usp)
    if device == "cuda":
        pipe.enable_vram_management(num_persistent_param_in_dit=num_persistent_param_in_dit)
    
    # Set pipeline information
    pipe.hardware_info = get_hardware_info()
    pipe.model_name = f"DIT: {', '.join(dit_models)}, T5: {t5_model}, VAE: {vae_model}" + (f", Image Encoder: {image_encoder_model}" if image_encoder_model else "")
    pipe.lora_info = ", ".join([f"{name} ({weight})" for name, weight in loaded_loras.items()]) if loaded_loras else "None"
    pipe.torch_dtype_info = f"DIT/T5/VAE: {torch_dtype}, Image Encoder: {image_encoder_torch_dtype if image_encoder_model else 'not used'}"
    pipe.num_persistent_param_in_dit = num_persistent_param_in_dit # Add new attributes for display
    return pipe

# Adaptive image resolution
def adaptive_resolution(image):
    if image is None:
        return 512, 512
    try:
        img = Image.open(image)
        width, height = img.size
        return width, height
    except Exception as e:
        return 512, 512

# Generate Vincent video
def generate_t2v(prompt, negative_prompt, num_inferrence_steps, seed, height, width,
                 num_frames, cfg_scale, sigma_shift, tea_cache_l1_thresh, tea_cache_model_id,
                 dit_models, t5_model, vae_model, image_encoder_model, fps, denoising_strength,
                 rand_device, tiled, tile_size_x, tile_size_y, tile_stride_x, tile_stride_y,
                 torch_dtype, image_encoder_torch_dtype, use_usp, enable_num_persistent=None,
                 num_persistent_param_in_dit=None, progress_bar_cmd=tqdm, progress_bar_st=None):
    # Handle the switch logic of num_persistent_param_in_dit
    if not enable_num_persistent:
        num_persistent_param_in_dit = None
    
    # Create a new pipe for each generation
    pipe = load_models(dit_models, t5_model, vae_model, image_encoder_model, prompt, torch_dtype,
                      image_encoder_torch_dtype, use_usp, num_persistent_param_in_dit)
    
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        # Handling random seeds
        actual_seed = int(seed)
        if actual_seed == -1:
            actual_seed = random.randint(0, 2**32 - 1)

        # Extract LoRA from the prompt word and clean the prompt word
        _, cleaned_prompt = extract_lora_from_prompt(prompt)

        # Call WanVideoPipeline
        frames = pipe(
            prompt=cleaned_prompt or "Default prompt word",
            negative_prompt=negative_prompt or "",
            input_image=None,
            input_video=None,
            denoising_strength=float(denoising_strength),
            seed=actual_seed,
            rand_device=rand_device,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            cfg_scale=float(cfg_scale),
            num_inferrence_steps=int(num_inferrence_steps),
            sigma_shift=float(sigma_shift),
            tiled = bool(tiled),
            tile_size=(int(tile_size_x), int(tile_size_y)),
            tile_stride=(int(tile_stride_x), int(tile_stride_y)),
            tea_cache_l1_thresh=float(tea_cache_l1_thresh) if tea_cache_l1_thresh is not None else None,
            tea_cache_model_id=tea_cache_model_id,
            progress_bar_cmd=progress_bar_cmd,
            progress_bar_st=progress_bar_st
        )

        output_dir = shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples or "outputs"
        os.makedirs(output_dir, exist_ok=True)
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"Output directory {output_dir} is not writable, please check permissions")
        output_path = os.path.join(output_dir, f"wan_video_t2v_{int(time.time())}.mp4")
        
        disk_space = psutil.disk_usage(output_dir).free // (1024 ** 3)
        if disk_space < 1:
            raise Exception("Insufficient disk space, please clean up and try again")
        
        save_video(frames, output_path, fps=int(fps), quality=5)

        mem_info = ""
        if torch.cuda.is_available():
            mem_used = torch.cuda.max_memory_allocated() // (1024 ** 3)
            mem_reserved = torch.cuda.max_memory_reserved() // (1024 ** 3)
            mem_info = f"Video memory usage: {mem_used}GB / Peak reserved: {mem_reserved}GB\n"

        time_cost = time.time() - start_time
        info = f"""{pipe.hardware_info}
Generate information:
- Resolution: {width}x{height}
- Total number of frames: {num_frames}
- Number of inferrence steps: {num_inferrence_steps}
- Random seed: {actual_seed} {'(randomly generated)' if seed == -1 else ''}
- Total time: {time_cost:.2f} seconds
- Frame rate: {fps} FPS
- Video duration: {num_frames / int(fps):.1f} seconds
{mem_info}
- Model version: DIT: {', '.join(dit_models)}, T5: {t5_model}, VAE: {vae_model}{', Image Encoder: ' + image_encoder_model if image_encoder_model else ''}
- Using Tiled: {'yes' if tiled else 'no'}
- Tile Size: ({tile_size_x}, {tile_size_y})
- Tile Stride: ({tile_stride_x}, {tile_stride_y})
- TeaCache L1 threshold: {tea_cache_l1_thresh if tea_cache_l1_thresh is not None else 'Not used'}
- TeaCache Model ID: {tea_cache_model_id}
- Torch data types: {pipe.torch_dtype_info}
- Use USP: {'yes' if use_usp else 'no'}
- Video memory management parameters (num_persistent_param_in_dit): {num_persistent_param_in_dit if num_persistent_param_in_dit is not None else 'unlimited'}
- Loaded LoRA: {pipe.lora_info}
"""
        return output_path, info
    except Exception as e:
        return None, f"Generation failed: {str(e)}"
    finally:
        del pipe # Ensure that the pipe is cleaned up after each generation

# Generate image-generated video (new end_image parameter)
def generate_i2v(image, end_image, prompt, negative_prompt, num_inferrence_steps, seed, height, width,
                num_frames, cfg_scale, sigma_shift, tea_cache_l1_thresh, tea_cache_model_id,
                dit_models, t5_model, vae_model, image_encoder_model, fps, denoising_strength,
                rand_device, tiled, tile_size_x, tile_size_y, tile_stride_x, tile_stride_y,
                torch_dtype, image_encoder_torch_dtype, use_usp, enable_num_persistent=None,
                num_persistent_param_in_dit=None, progress_bar_cmd=tqdm, progress_bar_st=None):
    # Handle the switch logic of num_persistent_param_in_dit
    if not enable_num_persistent:
        num_persistent_param_in_dit = None
    
    # Create a new pipe for each generation
    pipe = load_models(dit_models, t5_model, vae_model, image_encoder_model, prompt, torch_dtype,
                      image_encoder_torch_dtype, use_usp, num_persistent_param_in_dit)
    
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        if image is None:
            raise ValueError("Please upload the first frame")
        img = Image.open(image).convert("RGB")
        # If the tail frame is provided, load it
        end_img = Image.open(end_image).convert("RGB") if end_image else None

        # Handling random seeds
        actual_seed = int(seed)
        if actual_seed == -1:
            actual_seed = random.randint(0, 2**32 - 1)

        # Extract LoRA from the prompt word and clean the prompt word
        _, cleaned_prompt = extract_lora_from_prompt(prompt)

        #Call WanVideoPipeline and add the end_image parameter
        frames = pipe(
            prompt=cleaned_prompt or "Default prompt word",
            negative_prompt=negative_prompt or "",
            input_image=img,
            end_image=end_img, # New optional end frame parameter
            input_video=None,
            denoising_strength=float(denoising_strength),
            seed=actual_seed,
            rand_device=rand_device,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            cfg_scale=float(cfg_scale),
            num_inferrence_steps=int(num_inferrence_steps),
            sigma_shift=float(sigma_shift),
            tiled = bool(tiled),
            tile_size=(int(tile_size_x), int(tile_size_y)),
            tile_stride=(int(tile_stride_x), int(tile_stride_y)),
            tea_cache_l1_thresh=float(tea_cache_l1_thresh) if tea_cache_l1_thresh is not None else None,
            tea_cache_model_id=tea_cache_model_id,
            progress_bar_cmd=progress_bar_cmd,
            progress_bar_st=progress_bar_st
        )

        output_dir = shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples or "outputs"
        os.makedirs(output_dir, exist_ok=True)
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"Output directory {output_dir} is not writable, please check permissions")
        output_path = os.path.join(output_dir, f"wan_video_i2v_{int(time.time())}.mp4")
        
        disk_space = psutil.disk_usage(output_dir).free // (1024 ** 3)
        if disk_space < 1:
            raise Exception("Insufficient disk space, please clean up and try again")
        
        save_video(frames, output_path, fps=int(fps), quality=5)

        mem_info = ""
        if torch.cuda.is_available():
            mem_used = torch.cuda.max_memory_allocated() // (1024 ** 3)
            mem_reserved = torch.cuda.max_memory_reserved() // (1024 ** 3)
            mem_info = f"Video memory usage: {mem_used}GB / Peak reserved: {mem_reserved}GB\n"

        time_cost = time.time() - start_time
        info = f"""{pipe.hardware_info}
Generate information:
- Resolution: {width}x{height}
- Total number of frames: {num_frames}
- Number of inferrence steps: {num_inferrence_steps}
- Random seed: {actual_seed} {'(randomly generated)' if seed == -1 else ''}
- Total time: {time_cost:.2f} seconds
- Frame rate: {fps} FPS
- Video duration: {num_frames / int(fps):.1f} seconds
{mem_info}
- Model version: DIT: {', '.join(dit_models)}, T5: {t5_model}, VAE: {vae_model}{', Image Encoder: ' + image_encoder_model if image_encoder_model else ''}
- Using Tiled: {'yes' if tiled else 'no'}
- Tile Size: ({tile_size_x}, {tile_size_y})
- Tile Stride: ({tile_stride_x}, {tile_stride_y})
- TeaCache L1 threshold: {tea_cache_l1_thresh if tea_cache_l1_thresh is not None else 'Not used'}
- TeaCache Model ID: {tea_cache_model_id}
- Torch data types: {pipe.torch_dtype_info}
- Use USP: {'yes' if use_usp else 'no'}
- Video memory management parameters (num_persistent_param_in_dit): {num_persistent_param_in_dit if num_persistent_param_in_dit is not None else 'unlimited'}
- Loaded LoRA: {pipe.lora_info}
- Whether to use the end frame: {'yes' if end_image else 'no'}
"""
        return output_path, info
    except Exception as e:
        return None, f"Generation failed: {str(e)}"
    finally:
        del pipe # Ensure that the pipe is cleaned up after each generation

# Generate video (new control_video parameter)
def generate_v2v(video, control_video, prompt, negative_prompt, num_inferrence_steps, seed, height, width,
                num_frames, cfg_scale, sigma_shift, dit_models, t5_model, vae_model,
                image_encoder_model, fps, denoising_strength, rand_device, tiled,
                tile_size_x, tile_size_y, tile_stride_x, tile_stride_y, torch_dtype,
                image_encoder_torch_dtype, use_usp, enable_num_persistent=None,
                num_persistent_param_in_dit=None, progress_bar_cmd=tqdm, progress_bar_st=None):
    # Handle the switch logic of num_persistent_param_in_dit
    if not enable_num_persistent:
        num_persistent_param_in_dit = None
    
    # Create a new pipe for each generation
    pipe = load_models(dit_models, t5_model, vae_model, image_encoder_model, prompt, torch_dtype,
                      image_encoder_torch_dtype, use_usp, num_persistent_param_in_dit)
    
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        if video is None and control_video is None:
            raise ValueError("Please upload at least the initial video or the control video")
        video_data = VideoData(video, height=int(height), width=int(width)) if video else None
        control_video_data = VideoData(control_video, height=int(height), width=int(width)) if control_video else None

        # Handling random seeds
        actual_seed = int(seed)
        if actual_seed == -1:
            actual_seed = random.randint(0, 2**32 - 1)

        # Extract LoRA from the prompt word and clean the prompt word
        _, cleaned_prompt = extract_lora_from_prompt(prompt)

        #Call WanVideoPipeline and add the control_video parameter
        frames = pipe(
            prompt=cleaned_prompt or "Default prompt word",
            negative_prompt=negative_prompt or "",
            input_image=None,
            input_video=video_data,
            control_video=control_video_data, # New optional control video parameters
            denoising_strength=float(denoising_strength),
            seed=actual_seed,
            rand_device=rand_device,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            cfg_scale=float(cfg_scale),
            num_inferrence_steps=int(num_inferrence_steps),
            sigma_shift=float(sigma_shift),
            tiled = bool(tiled),
            tile_size=(int(tile_size_x), int(tile_size_y)),
            tile_stride=(int(tile_stride_x), int(tile_stride_y)),
            tea_cache_l1_thresh=None, # TeaCache does not support video generation
            tea_cache_model_id="",
            progress_bar_cmd=progress_bar_cmd,
            progress_bar_st=progress_bar_st
        )

        output_dir = shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples or "outputs"
        os.makedirs(output_dir, exist_ok=True)
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"Output directory {output_dir} is not writable, please check permissions")
        output_path = os.path.join(output_dir, f"wan_video_v2v_{int(time.time())}.mp4")
        
        disk_space = psutil.disk_usage(output_dir).free // (1024 ** 3)
        if disk_space < 1:
            raise Exception("Insufficient disk space, please clean up and try again")
        
        save_video(frames, output_path, fps=int(fps), quality=5)

        mem_info = ""
        if torch.cuda.is_available():
            mem_used = torch.cuda.max_memory_allocated() // (1024 ** 3)
            mem_reserved = torch.cuda.max_memory_reserved() // (1024 ** 3)
            mem_info = f"Video memory usage: {mem_used}GB / Peak reserved: {mem_reserved}GB\n"

        time_cost = time.time() - start_time
        info = f"""{pipe.hardware_info}
Generate information:
- Resolution: {width}x{height}
- Resolution: {width}x{height}
- Total number of frames: {num_frames}
- Number of inferrence steps: {num_inferrence_steps}
- Random seed: {actual_seed} {'(randomly generated)' if seed == -1 else ''}
- Total time: {time_cost:.2f} seconds
- Frame rate: {fps} FPS
- Video duration: {num_frames / int(fps):.1f} seconds
{mem_info}
- Model version: DIT: {', '.join(dit_models)}, T5: {t5_model}, VAE: {vae_model}{', Image Encoder: ' + image_encoder_model if image_encoder_model else ''}
- Using Tiled: {'yes' if tiled else 'no'}
- Tile Size: ({tile_size_x}, {tile_size_y})
- Tile Stride: ({tile_stride_x}, {tile_stride_y})
- Torch data types: {pipe.torch_dtype_info}
- Use USP: {'yes' if use_usp else 'no'}
- Video memory management parameters (num_persistent_param_in_dit): {num_persistent_param_in_dit if num_persistent_param_in_dit is not None else 'unlimited'}
- Loaded LoRA: {pipe.lora_info}
- Whether to use control video: {'yes' if control_video else 'no'}
"""
        return output_path, info
    except Exception as e:
        return None, f"Generation failed: {str(e)}"
    finally:
        del pipe # Ensure that the pipe is cleaned up after each generation