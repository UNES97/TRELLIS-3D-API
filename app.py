from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import imageio
import uuid
import os
from typing import Optional, Dict, Any
from PIL import Image
import io
from easydict import EasyDict as edict
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils

# Constants
MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = "/tmp/Trellis-demo"
os.makedirs(TMP_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="TRELLIS 3D Generation API",
    description="API for converting images to 3D assets using TRELLIS",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline at startup
pipeline: Optional[TrellisImageTo3DPipeline] = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()

# Pydantic models for request validation
class GenerationSettings(BaseModel):
    seed: int = 0
    randomize_seed: bool = True
    ss_guidance_strength: float = 7.5
    ss_sampling_steps: int = 12
    slat_guidance_strength: float = 3.0
    slat_sampling_steps: int = 12

class GLBSettings(BaseModel):
    mesh_simplify: float = 0.95
    texture_size: int = 1024

# Utility functions
def preprocess_image(image: Image.Image) -> tuple[str, Image.Image]:
    trial_id = str(uuid.uuid4())
    processed_image = pipeline.preprocess_image(image)
    processed_image.save(f"{TMP_DIR}/{trial_id}.png")
    return trial_id, processed_image

def pack_state(gs: Gaussian, mesh: MeshExtractResult, trial_id: str) -> dict:
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy().tolist(),
            '_features_dc': gs._features_dc.cpu().numpy().tolist(),
            '_scaling': gs._scaling.cpu().numpy().tolist(),
            '_rotation': gs._rotation.cpu().numpy().tolist(),
            '_opacity': gs._opacity.cpu().numpy().tolist(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy().tolist(),
            'faces': mesh.faces.cpu().numpy().tolist(),
        },
        'trial_id': trial_id,
    }

def unpack_state(state: dict) -> tuple[Gaussian, edict, str]:
    gs = Gaussian(
        aabb=state['gaussian']['aabb'],
        sh_degree=state['gaussian']['sh_degree'],
        mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
        scaling_bias=state['gaussian']['scaling_bias'],
        opacity_bias=state['gaussian']['opacity_bias'],
        scaling_activation=state['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
    gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
    gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')
    
    mesh = edict(
        vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
        faces=torch.tensor(state['mesh']['faces'], device='cuda'),
    )
    
    return gs, mesh, state['trial_id']

# API endpoints
@app.post("/generate")
async def generate_3d(
    image: UploadFile = File(...),
    settings: GenerationSettings = GenerationSettings()
) -> Dict[str, Any]:
    try:
        # Read and validate image
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        # Preprocess image
        trial_id, processed_image = preprocess_image(img)

        # Generate 3D model
        seed = np.random.randint(0, MAX_SEED) if settings.randomize_seed else settings.seed
        
        outputs = pipeline.run(
            processed_image,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": settings.ss_sampling_steps,
                "cfg_strength": settings.ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": settings.slat_sampling_steps,
                "cfg_strength": settings.slat_guidance_strength,
            },
        )

        # Generate preview video
        video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
        video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
        video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
        
        video_path = f"{TMP_DIR}/{trial_id}_preview.mp4"
        imageio.mimsave(video_path, video, fps=15)

        # Pack state and return response
        state = pack_state(outputs['gaussian'][0], outputs['mesh'][0], trial_id)
        
        return {
            "status": "success",
            "state": state,
            "video_url": f"/preview/{trial_id}",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/preview/{trial_id}")
async def get_preview(trial_id: str):
    video_path = f"{TMP_DIR}/{trial_id}_preview.mp4"
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Preview not found")
    return FileResponse(video_path, media_type="video/mp4")

@app.post("/extract-glb")
async def extract_glb(
    state: dict,
    settings: GLBSettings = GLBSettings()
):
    try:
        gs, mesh, trial_id = unpack_state(state)
        glb = postprocessing_utils.to_glb(
            gs, 
            mesh, 
            simplify=settings.mesh_simplify, 
            texture_size=settings.texture_size, 
            verbose=False
        )
        
        glb_path = f"{TMP_DIR}/{trial_id}.glb"
        glb.export(glb_path)
        
        return FileResponse(
            glb_path,
            media_type="model/gltf-binary",
            filename=f"model_{trial_id}.glb"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Cleanup temporary files periodically
@app.on_event("shutdown")
async def cleanup():
    import shutil
    shutil.rmtree(TMP_DIR, ignore_errors=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)