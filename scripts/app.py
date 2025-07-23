import logging
import threading
from threading import Lock
from fastapi import FastAPI
from backend_wanvideo.inferrence import *
from backend_wanvideo.ui import *
from backend_wanvideo.api import Api
import uvicorn

# Setting up logging
logging.basicConfig(level=logging.INFO)
try:
    from scripts.gradio_patch import money_patch_gradio
    if money_patch_gradio():
        logging.info("gradio patch applied successfully")
    else:
        logging.warning("gradio patch import failed")
except Exception as e:
    #logging.error(f"Gradio patch loading failed: {e}")
    pass

# Check if it is running in the WebUI environment
try:
    from modules import script_callbacks, shared
    IN_WEBUI = True
except ImportError:
    IN_WEBUI = False
    shared = type('Shared', (), {'opts': type('Opts', (), {'outdir_samples': '', 'outdir_txt2img_samples': ''})})()

# Hard-coded configuration
HOST = "127.0.0.1"
PORT_API = 7870 # Dedicated FastAPI port
NPROC_PER_NODE = 1 # Default number of USP processes

if IN_WEBUI:
    # In WebUI environment, register UI and API callbacks
    from backend_wanvideo.api import on_app_started
    script_callbacks.on_ui_tabs(lambda: [(create_wan_video_tab(), "Wan Video", "wan_video_tab")])
    script_callbacks.on_app_started(on_app_started)
else:
    # In a non-WebUI environment, start Gradio UI and FastAPI separately
    if __name__ == "__main__":
        # Create the Gradio interface
        interface = create_wan_video_tab()
        logging.info("Gradio interface created")

        # Create a standalone FastAPI instance
        app = FastAPI(docs_url="/docs", openapi_url="/openapi.json")
        queue_lock = Lock() # Provide thread lock for non-WebUI environment
        api = Api(app, queue_lock, prefix="/wanvideo/v1")
        logging.info("API routing has been mounted to a separate FastAPI instance")

        # Print USP and API documentation tips
        if NPROC_PER_NODE > 1:
            print("Tip: USP has been enabled, you need to run it with the following command:")
            print(f"torchrun --standalone --nproc_per_node={NPROC_PER_NODE} generation.py")
        else:
            print("Tip: To enable USP, run the following command (modify NPROC_PER_NODE):")
            print("torchrun --standalone --nproc_per_node=<number of processes> generation.py")
        print(f"API documentation is available at: http://{HOST}:{PORT_API}/docs")

        # Start Gradio in a separate thread
        def run_gradio():
            try:
                interface.launch(
                    server_name=HOST,
                    allowed_paths=["outputs"],
                    prevent_thread_lock=True,
                    share=True
                )
            except Exception as e:
                logging.error(f"Gradio startup failed: {str(e)}")

        gradio_thread = threading.Thread(target=run_gradio)
        gradio_thread.start()

        # Start a standalone FastAPI server
        try:
            uvicorn.run(
                app,
                host=HOST,
                port=PORT_API,
                log_level="info"
            )
        except Exception as e:
            logging.error(f"FastAPI startup failed: {str(e)}")
        finally:
            interface.close()
            gradio_thread.join()