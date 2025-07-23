# Below is a direct translation of the Wan2.1 README.md by Spawner1145. 
#           All credits go to him and his outstanding work.

-----------------------------------------------------------

# sd-webui-wanvideo

I'm too new to understand kj nodes, so I just use diffusers (

This project already supports api

---

## Notes

1. **models folder location**

   * If you are using the `sd-webui` plugin, please cut the `models` folder in the root directory to the `webui` root directory and merge it with the original `models` folder.
   * If you run the project separately, the path structure of the model folder is as follows:

     ```
     sd-webui-wanvideo/
     ├── install.py
     ├── requirements.txt
     ├── scripts/
     │ └── app.py # Main script file
     ├── backend/
     │ ├── api.py
     │ ├── inferrence.py
     │ └── ui.py
     ├── models/wan2.1/
     │ ├── dit/ # for the bottom mold
     │ │ ├── xxx001.safetensors
     │ │ ├── xxx002.safetensors
     │ │ └── ......
     │ ├── t5/ # T5 model
     │ ├── vae/ # VAE model
     │ ├── lora/ # LoRA model
     │ └── image_encoder/ # CLIP model
     ├── api_examples/ #api call example file
     │ ├── t2v.py
     │ ├── i2v.py
     │ └── v2v.py
     ├── license
     └── README.md
     ```
2. **Startup method**

   * **As an `sd-webui` plugin**: Put the project into the `extensions` folder.
   * **Running standalone**:
     Run the following command in the project root directory:

     ```
     python -m scripts.app
     ```

   **Notice** :

   * When running alone, you need to install dependencies first. Please make sure that `torch` and `torchvision` are installed, and then run the following command to install other dependencies:

     ```
     pip install -r requirements.txt
     ```
   * **DO NOT run `install.py`**, it is only used in plugin mode.

---

## Model download and configuration

### Download address

* [ModelScope](https://www.modelscope.cn/) or [HuggingFace](https://huggingface.co/)
  (Take Tongyi Wanxiang 2.1 Wensheng Video 1.3B model as an example)

#### Vincent video model (Wan2.1-T2V-1.3B)

* **dit model**
  Download [diffusion_pytorch_model.safetensors](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B/file/view/master?fileName=diffusion_pytorch_model.safetensors&status=2) and put it in the `models/wan2.1/dit/` folder
* **t5 model**
  Download [models_t5_umt5-xxl-enc-bf16.pth](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B/file/view/master?fileName=models_t5_umt5-xxl-enc-bf16.pth&status=2) and put it into the `models/wan2.1/t5/` folder
* **vae model**
  Download [Wan2.1_VAE.pth](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B/file/view/master?fileName=Wan2.1_VAE.pth&status=2) and put it in the `models/wan2.1/vae/` folder

#### Image video model (Wan2.1-I2V-14B-480P)

* **image_encoder model**
  Download [models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P/file/view/master?fileName=models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth&status=2) and put it in the `models/wan2.1/image_encoder/` folder

#### Sharding Model

* If you encounter a sharded model (such as `diffusion_pytorch_model.safetensors` with a suffix of `00001 of 00001`), you need to download all sharded files and put them all into the `dit` folder
* At the same time, download the corresponding index file (such as `diffusion_pytorch_model.safetensors.index.json`) and put it in the `dit` folder
* When used, the interface will automatically load all related shards

---

## Control Model and Inpaint Model

* **Control Model**

  HuggingFace address: [alibaba-pai/Wan2.1-Fun-1.3B-Control](https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-Control)

  ModelScope address: [pai/Wan2.1-Fun-1.3B-Control](https://www.modelscope.cn/models/pai/Wan2.1-Fun-1.3B-Control)
* **Inpaint Model**

  HuggingFace address: [alibaba-pai/Wan2.1-Fun-1.3B-InP](https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-InP)

  ModelScope address: [pai/Wan2.1-Fun-1.3B-InP](https://www.modelscope.cn/models/pai/Wan2.1-Fun-1.3B-InP)
