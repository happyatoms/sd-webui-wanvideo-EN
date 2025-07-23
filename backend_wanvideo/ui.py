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
from modelscope import snapshot_download, dataset_snapshot_download
from backend_wanvideo.inferrence import *

# Setting up logging
logging.basicConfig(level=logging.INFO)
try:
    from scripts.gradio_patch import money_patch_gradio
    if money_patch_gradio():
        logging.info("gradio patch applied successfully")
    else:
        logging.warning("gradio patch import failed")
except Exception as e:
    logging.error(e)

# Check if it is running in the WebUI environment
try:
    from modules import script_callbacks, shared
    IN_WEBUI = True
except ImportError:
    IN_WEBUI = False
    shared = type('Shared', (), {'opts': type('Opts', (), {'outdir_samples': '', 'outdir_txt2img_samples': ''})})()

# Create the interface
def create_wan_video_tab():
    # Define model directory
    base_dir = "models/wan2.1"
    dit_dir = os.path.join(base_dir, "dit")
    t5_dir = os.path.join(base_dir, "t5")
    vae_dir = os.path.join(base_dir, "vae")
    image_encoder_dir = os.path.join(base_dir, "image_encoder")
    lora_dir = os.path.join(base_dir, "lora")
    
    # Get the model file list
    dit_models = get_model_files(dit_dir)
    t5_models = get_model_files(t5_dir)
    vae_models = get_model_files(vae_dir)
    image_encoder_models = get_model_files(image_encoder_dir)
    lora_models = get_model_files(lora_dir)

    with gr.Blocks(analytics_enabled=False) as wan_interface:
        gr.Markdown("## Wan2.1 text/image/video generation video")
        gr.Markdown("Tip: Add `<lora:model file name:weight>` to the prompt to load LoRA, for example `<lora:example_lora.ckpt:1.0>`")
        
        # Top model selection
        with gr.Row():
            dit_model = gr.Dropdown(
                label="Select DIT model (multiple selections are allowed, multiple files will be merged into one model)",
                choices=dit_models,
                value=[dit_models[0]], # The first model is selected by default
                multiselect=True
            )
            t5_model = gr.Dropdown(label="Select T5 model", choices=t5_models, value=t5_models[0])
            vae_model = gr.Dropdown(label="Select VAE model", choices=vae_models, value=vae_models[0])
            image_encoder_model = gr.Dropdown(label="Select Image Encoder model (required for image-generated video)", choices=["None"] + image_encoder_models, value="None")
        
        with gr.Tabs():
            with gr.Tab("Text-to-video"):
                with gr.Row():
                    with gr.Column():
                        prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Enter the prompt word describing the video content, which can contain <lora:model file name:weight>")
                        negative_prompt = gr.Textbox(
                            label="Negative prompts",
                            lines=3,
                            value="Gorgeous colors, overexposed, static, blurred details, subtitles, style, work, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, deformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards"
                        )
                        
                        with gr.Accordion("Basic parameters", open=True):
                            width = gr.Slider(label="width", minimum=256, maximum=1920, value=832, step=8)
                            height = gr.Slider(label="height", minimum=256, maximum=1080, value=480, step=8)
                            num_frames = gr.Number(label="Number of frames", value=81, minimum=1, precision=0)
                            fps = gr.Slider(label="Output frame rate (FPS)", minimum=1, maximum=60, value=15, step=1)

                        with gr.Accordion("Advanced parameters", open=False):
                            num_inferrence_steps = gr.Slider(label="Number of inferrence steps", minimum=1, maximum=100, value=15, step=1)
                            cfg_scale = gr.Number(label="CFG Scale", value=5.0)
                            sigma_shift = gr.Number(label="Sigma Shift", value=5.0)
                            seed = gr.Number(label="Random seed (-1 is random)", value=-1, precision=0)
                            denoising_strength = gr.Slider(label="Noise reduction strength", minimum=0.0, maximum=1.0, value=1.0, step=0.01)
                            rand_device = gr.Dropdown(label="Random device", choices=["cpu", "cuda"], value="cpu")
                            tiled = gr.Checkbox(label="Use Tiled", value=True)
                            tile_size_x = gr.Number(label="Tile Size X", value=30, precision=0)
                            tile_size_y = gr.Number(label="Tile Size Y", value=52, precision=0)
                            tile_stride_x = gr.Number(label="Tile Stride X", value=15, precision=0)
                            tile_stride_y = gr.Number(label="Tile Stride Y", value=26, precision=0)
                            torch_dtype = gr.Dropdown(label="DIT/T5/VAE data type", choices=["float16", "bfloat16", "float8_e4m3fn"], value="bfloat16")
                            image_encoder_torch_dtype = gr.Dropdown(label="Image Encoder data type", choices=["float16", "float32", "bfloat16"], value="float32")
                            use_usp = gr.Checkbox(label="Use USP (Unified Sequence Parallel)", value=False)
                            nproc_per_node = gr.Number(label="USP number of processes per node (need to be run by torchrun)", value=1, minimum=1, precision=0, visible=False)
                            enable_num_persistent = gr.Checkbox(label="Enable video memory optimization parameters (num_persistent_param_in_dit)", value=False)
                            num_persistent_param_in_dit = gr.Slider(
                                label="Video memory management parameter value (the smaller the value, the less video memory is required, but it takes longer)",
                                minimum=0,
                                maximum=10**10,
                                value=7*10**9,
                                step=10**8,
                                visible=False,
                                info="Adjust this value after enabling, 0 means the minimum video memory requirement"
                            )

                            # Dynamically display nproc_per_node and num_persistent_param_in_dit
                            def toggle_nproc_visibility(use_usp):
                                return gr.update(visible=use_usp)
                            use_usp.change(fn=toggle_nproc_visibility, inputs=use_usp, outputs=nproc_per_node)

                            def toggle_num_persistent_visibility(enable):
                                return gr.update(visible=enable)
                            enable_num_persistent.change(fn=toggle_num_persistent_visibility, inputs=enable_num_persistent, outputs=num_persistent_param_in_dit)

                        with gr.Accordion("TeaCache parameters", open=False):
                            tea_cache_l1_thresh = gr.Number(label="TeaCache L1 threshold (the larger the faster but the quality decreases)", value=0.07)
                            tea_cache_model_id = gr.Dropdown(label="TeaCache Model ID", choices=["Wan2.1-T2V-1.3B", "Wan2.1-T2V-14B", "Wan2.1-I2V-14B-480P", "Wan2.1-I2V-14B-720P"], value="Wan2.1-T2V-1.3B")

                        generate_btn = gr.Button("Generate Video")

                    with gr.Column():
                        output_video = gr.Video(label="Generated results")
                        info_output = gr.Textbox(label="System Information", interactive=False, lines=16)

                generate_btn.click(
                    fn=generate_t2v,
                    inputs=[
                        prompt,
                        negative_prompt,
                        num_inferrence_steps,
                        seed,
                        height,
                        width,
                        num_frames,
                        cfg_scale,
                        sigma_shift,
                        tea_cache_l1_thresh,
                        tea_cache_model_id,
                        dit_model,
                        t5_model,
                        vae_model,
                        image_encoder_model,
                        fps,
                        denoising_strength,
                        rand_device,
                        tiled,
                        tile_size_x,
                        tile_size_y,
                        tile_stride_x,
                        tile_stride_y,
                        torch_dtype,
                        image_encoder_torch_dtype,
                        use_usp,
                        enable_num_persistent,
                        num_persistent_param_in_dit
                    ],
                    outputs=[output_video, info_output]
                )

            with gr.Tab("Image-to-video"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(label="Upload first frame", type="filepath")
                        end_image_input = gr.Image(label="Upload last frame (optional)", type="filepath") # Add last frame input
                        adapt_resolution_btn = gr.Button("Adaptive picture resolution")
                        prompt_i2v = gr.Textbox(label="Prompt", lines=3, placeholder="Enter the prompt word describing the video content, which can contain <lora:model file name:weight>")
                        negative_prompt_i2v = gr.Textbox(
                            label="Negative prompts",
                            lines=3,
                            value="Gorgeous colors, overexposed, static, blurred details, subtitles, style, work, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, deformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards"
                        )
                        
                        with gr.Accordion("Basic parameters", open=True):
                            width_i2v = gr.Slider(label="width", minimum=256, maximum=1920, value=832, step=8)
                            height_i2v = gr.Slider(label="height", minimum=256, maximum=1080, value=480, step=8)
                            num_frames_i2v = gr.Number(label="Number of frames", value=81, minimum=1, precision=0)
                            fps_i2v = gr.Slider(label="Output frame rate (FPS)", minimum=1, maximum=60, value=15, step=1)

                        with gr.Accordion("Advanced parameters", open=False):
                            num_inferrence_steps_i2v = gr.Slider(label="Number of inferrence steps", minimum=1, maximum=100, value=15, step=1)
                            cfg_scale_i2v = gr.Number(label="CFG Scale", value=5.0)
                            sigma_shift_i2v = gr.Number(label="Sigma Shift", value=5.0)
                            seed_i2v = gr.Number(label="Random seed (-1 is random)", value=-1, precision=0)
                            denoising_strength_i2v = gr.Slider(label="Noise Reduction Strength", minimum=0.0, maximum=1.0, value=1.0, step=0.01)
                            rand_device_i2v = gr.Dropdown(label="Random device", choices=["cpu", "cuda"], value="cpu")
                            tiled_i2v = gr.Checkbox(label="Use Tiled", value=True)
                            tile_size_x_i2v = gr.Number(label="Tile Size X", value=30, precision=0)
                            tile_size_y_i2v = gr.Number(label="Tile Size Y", value=52, precision=0)
                            tile_stride_x_i2v = gr.Number(label="Tile Stride X", value=15, precision=0)
                            tile_stride_y_i2v = gr.Number(label="Tile Stride Y", value=26, precision=0)
                            torch_dtype_i2v = gr.Dropdown(label="DIT/T5/VAE data type", choices=["float16", "bfloat16", "float8_e4m3fn"], value="bfloat16")
                            image_encoder_torch_dtype_i2v = gr.Dropdown(label="Image Encoder data type", choices=["float16", "float32", "bfloat16"], value="float32")
                            use_usp_i2v = gr.Checkbox(label="Use USP (Unified Sequence Parallel)", value=False)
                            nproc_per_node_i2v = gr.Number(label="USP number of processes per node (need to run torchrun)", value=1, minimum=1, precision=0, visible=False)
                            enable_num_persistent_i2v = gr.Checkbox(label="Enable video memory optimization parameters (num_persistent_param_in_dit)", value=False)
                            num_persistent_param_in_dit_i2v = gr.Slider(
                                label="Video memory management parameter value (the smaller the value, the less video memory is required, but it takes longer)",
                                minimum=0,
                                maximum=10**10,
                                value=7*10**9,
                                step=10**8,
                                visible=False,
                                info="Adjust this value after enabling, 0 means the minimum video memory requirement"
                            )

                            # Dynamically display nproc_per_node_i2v and num_persistent_param_in_dit_i2v
                            def toggle_nproc_visibility(use_usp):
                                return gr.update(visible=use_usp)
                            use_usp_i2v.change(fn=toggle_nproc_visibility, inputs=use_usp_i2v, outputs=nproc_per_node_i2v)

                            def toggle_num_persistent_visibility(enable):
                                return gr.update(visible=enable)
                            enable_num_persistent_i2v.change(fn=toggle_num_persistent_visibility, inputs=enable_num_persistent_i2v, outputs=num_persistent_param_in_dit_i2v)

                        with gr.Accordion("TeaCache parameters", open=False):
                            tea_cache_l1_thresh_i2v = gr.Number(label="TeaCache L1 threshold (larger is faster but quality is reduced)", value=0.19)
                            tea_cache_model_id_i2v = gr.Dropdown(label="TeaCache Model ID", choices=["Wan2.1-T2V-1.3B", "Wan2.1-T2V-14B", "Wan2.1-I2V-14B-480P", "Wan2.1-I2V-14B-720P"], value="Wan2.1-I2V-14B-480P")

                        generate_i2v_btn = gr.Button("Generate Video")

                    with gr.Column():
                        output_video_i2v = gr.Video(label="Generated results")
                        info_output_i2v = gr.Textbox(label="System Information", interactive=False, lines=16)

                adapt_resolution_btn.click(
                    fn=adaptive_resolution,
                    inputs=[image_input],
                    outputs=[width_i2v, height_i2v]
                )

                generate_i2v_btn.click(
                    fn=generate_i2v,
                    inputs=[
                        image_input,
                        end_image_input, # Add the last frame input
                        prompt_i2v,
                        negative_prompt_i2v,
                        num_inferrence_steps_i2v,
                        seed_i2v,
                        height_i2v,
                        width_i2v,
                        num_frames_i2v,
                        cfg_scale_i2v,
                        sigma_shift_i2v,
                        tea_cache_l1_thresh_i2v,
                        tea_cache_model_id_i2v,
                        dit_model,
                        t5_model,
                        vae_model,
                        image_encoder_model,
                        fps_i2v,
                        denoising_strength_i2v,
                        rand_device_i2v,
                        tiled_i2v,
                        tile_size_x_i2v,
                        tile_size_y_i2v,
                        tile_stride_x_i2v,
                        tile_stride_y_i2v,
                        torch_dtype_i2v,
                        image_encoder_torch_dtype_i2v,
                        use_usp_i2v,
                        enable_num_persistent_i2v,
                        num_persistent_param_in_dit_i2v
                    ],
                    outputs=[output_video_i2v, info_output_i2v]
                )

            with gr.Tab("Video-to-video"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(label="Upload initial video", format="mp4")
                        control_video_input = gr.Video(label="Upload control video (optional)", format="mp4") # Add control video input
                        prompt_v2v = gr.Textbox(label="Prompts", lines=3, placeholder="Enter prompt words describing the video content, which can include <lora:model file name:weight>")
                        negative_prompt_v2v = gr.Textbox(
                            label="Negative prompts",
                            lines=3,
                            value="Gorgeous colors, overexposed, static, blurred details, subtitles, style, work, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, deformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards"
                        )
                        
                        with gr.Accordion("Basic parameters", open=True):
                            width_v2v = gr.Slider(label="width", minimum=256, maximum=1920, value=832, step=8)
                            height_v2v = gr.Slider(label="height", minimum=256, maximum=1080, value=480, step=8)
                            num_frames_v2v = gr.Number(label="Number of frames", value=81, minimum=1, precision=0)
                            fps_v2v = gr.Slider(label="Output frame rate (FPS)", minimum=1, maximum=60, value=15, step=1)

                        with gr.Accordion("Advanced parameters", open=False):
                            num_inferrence_steps_v2v = gr.Slider(label="Number of inferrence steps", minimum=1, maximum=100, value=15, step=1)
                            cfg_scale_v2v = gr.Number(label="CFG Scale", value=5.0)
                            sigma_shift_v2v = gr.Number(label="Sigma Shift", value=5.0)
                            seed_v2v = gr.Number(label="Random seed (-1 is random)", value=-1, precision=0)
                            denoising_strength_v2v = gr.Slider(label="Noise reduction strength", minimum=0.0, maximum=1.0, value=0.7, step=0.01)
                            rand_device_v2v = gr.Dropdown(label="Random device", choices=["cpu", "cuda"], value="cpu")
                            tiled_v2v = gr.Checkbox(label="Use Tiled", value=True)
                            tile_size_x_v2v = gr.Number(label="Tile Size X", value=30, precision=0)
                            tile_size_y_v2v = gr.Number(label="Tile Size Y", value=52, precision=0)
                            tile_stride_x_v2v = gr.Number(label="Tile Stride X", value=15, precision=0)
                            tile_stride_y_v2v = gr.Number(label="Tile Stride Y", value=26, precision=0)
                            torch_dtype_v2v = gr.Dropdown(label="DIT/T5/VAE data type", choices=["float16", "bfloat16", "float8_e4m3fn"], value="bfloat16")
                            image_encoder_torch_dtype_v2v = gr.Dropdown(label="Image Encoder data type", choices=["float16", "float32", "bfloat16"], value="float32")
                            use_usp_v2v = gr.Checkbox(label="Use USP (Unified Sequence Parallel)", value=False)
                            nproc_per_node_v2v = gr.Number(label="USP number of processes per node (need to run torchrun)", value=1, minimum=1, precision=0, visible=False)
                            enable_num_persistent_v2v = gr.Checkbox(label="Enable video memory optimization parameters (num_persistent_param_in_dit)", value=False)
                            num_persistent_param_in_dit_v2v = gr.Slider(
                                label="Video memory management parameter value (the smaller the value, the less video memory is required, but it takes longer)",
                                minimum=0,
                                maximum=10**10,
                                value=7*10**9,
                                step=10**8,
                                visible=False,
                                info="Adjust this value after enabling, 0 means the minimum video memory requirement"
                            )

                            # Dynamically display nproc_per_node_v2v and num_persistent_param_in_dit_v2v
                            def toggle_nproc_visibility(use_usp):
                                return gr.update(visible=use_usp)
                            use_usp_v2v.change(fn=toggle_nproc_visibility, inputs=use_usp_v2v, outputs=nproc_per_node_v2v)

                            def toggle_num_persistent_visibility(enable):
                                return gr.update(visible=enable)
                            enable_num_persistent_v2v.change(fn=toggle_num_persistent_visibility, inputs=enable_num_persistent_v2v, outputs=num_persistent_param_in_dit_v2v)

                        generate_v2v_btn = gr.Button("Generate Video")

                    with gr.Column():
                        output_video_v2v = gr.Video(label="Generated result")
                        info_output_v2v = gr.Textbox(label="System Information", interactive=False, lines=16)

                generate_v2v_btn.click(
                    fn=generate_v2v,
                    inputs=[
                        video_input,
                        control_video_input, # Add control video input
                        prompt_v2v,
                        negative_prompt_v2v,
                        num_inferrence_steps_v2v,
                        seed_v2v,
                        height_v2v,
                        width_v2v,
                        num_frames_v2v,
                        cfg_scale_v2v,
                        sigma_shift_v2v,
                        dit_model,
                        t5_model,
                        vae_model,
                        image_encoder_model,
                        fps_v2v,
                        denoising_strength_v2v,
                        rand_device_v2v,
                        tiled_v2v,
                        tile_size_x_v2v,
                        tile_size_y_v2v,
                        tile_stride_x_v2v,
                        tile_stride_y_v2v,
                        torch_dtype_v2v,
                        image_encoder_torch_dtype_v2v,
                        use_usp_v2v,
                        enable_num_persistent_v2v,
                        num_persistent_param_in_dit_v2v
                    ],
                    outputs=[output_video_v2v, info_output_v2v]
                )
    
    return wan_interface
