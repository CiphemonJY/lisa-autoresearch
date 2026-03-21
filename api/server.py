#!/usr/bin/env python3
"""
FastAPI Wrapper for LISA+Offload

Serves LISA+Offload as a local API for easy integration.

Endpoints:
- POST /train - Start training
- POST /inference - Run inference
- GET /status - Check training status
- GET /models - List available models
- POST /config - Set configuration

Usage:
    uvicorn api_server:app --host 0.0.0.0 --port 8000
    
    # Train a model
    curl -X POST http://localhost:8000/train \
        -H "Content-Type: application/json" \
        -d '{"model": "Qwen/Qwen2.5-7B-Instruct", "data_dir": "training_data"}'
    
    # Run inference
    curl -X POST http://localhost:8000/inference \
        -H "Content-Type: application/json" \
        -d '{"prompt": "Hello, world!", "model": "Qwen/Qwen2.5-7B-Instruct"}'
"""

import os
import sys
import json
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import uuid

os.environ['PYTHONWARNINGS'] = 'ignore::urllib3.warnings.NotOpenSSLWarning'

# Try to import FastAPI
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("Installing FastAPI...")
    subprocess.run([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn", "pydantic"], check=True)
    from fastapi import FastApp, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn

# Add LISA+Offload to path
sys.path.insert(0, str(Path(__file__).parent))

# Try to import LISA+Offload
try:
    from lisa_offload import LISAOffloadedTrainer, LISAConfig
    from hardware_detection import detect_hardware
    LISA_AVAILABLE = True
except ImportError:
    LISA_AVAILABLE = False
    print("Warning: LISA+Offload not available. Using simulation mode.")


# ============================================================================
# Data Models
# ============================================================================

class TrainRequest(BaseModel):
    """Training request."""
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    data_dir: str = "training_data"
    iterations: int = 100
    learning_rate: float = 1e-5
    lisa_config: Optional[Dict[str, Any]] = None


class InferenceRequest(BaseModel):
    """Inference request."""
    prompt: str
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    max_tokens: int = 100
    temperature: float = 0.7


class ConfigRequest(BaseModel):
    """Configuration request."""
    max_memory_gb: Optional[float] = 6.0
    layer_groups: Optional[int] = 6
    bottom_layers: Optional[int] = 5
    top_layers: Optional[int] = 5
    middle_sample: Optional[int] = 2


# ============================================================================
# Training State
# ============================================================================

@dataclass
class TrainingJob:
    """Training job state."""
    job_id: str
    model: str
    status: str = "pending"  # pending, running, completed, failed
    progress: float = 0.0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TrainingManager:
    """Manages training jobs."""
    
    def __init__(self):
        self.jobs: Dict[str, TrainingJob] = {}
        self.config = {
            "max_memory_gb": 6.0,
            "layer_groups": 6,
            "bottom_layers": 5,
            "top_layers": 5,
            "middle_sample": 2,
        }
    
    def create_job(self, model: str) -> TrainingJob:
        """Create a new training job."""
        job_id = str(uuid.uuid4())[:8]
        job = TrainingJob(
            job_id=job_id,
            model=model,
            status="pending",
            start_time=datetime.now().isoformat(),
        )
        self.jobs[job_id] = job
        return job
    
    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a training job by ID."""
        return self.jobs.get(job_id)
    
    def update_job(self, job_id: str, **kwargs):
        """Update a training job."""
        if job_id in self.jobs:
            for key, value in kwargs.items():
                setattr(self.jobs[job_id], key, value)
    
    def list_jobs(self) -> List[TrainingJob]:
        """List all training jobs."""
        return list(self.jobs.values())


# ============================================================================
# API Server
# ============================================================================

app = FastAPI(
    title="LISA+Offload API",
    description="Serve LISA+Offload as a local API",
    version="1.0.0",
)

manager = TrainingManager()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "LISA+Offload API",
        "version": "1.0.0",
        "status": "running",
        "lisa_available": LISA_AVAILABLE,
    }


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy"}


@app.get("/models")
async def list_models():
    """List available models."""
    models = [
        {"id": "Qwen/Qwen2.5-0.5B-Instruct", "size": "0.5B", "memory_gb": 2.0},
        {"id": "Qwen/Qwen2.5-1.5B-Instruct", "size": "1.5B", "memory_gb": 3.0},
        {"id": "Qwen/Qwen2.5-3B-Instruct", "size": "3B", "memory_gb": 4.0},
        {"id": "Qwen/Qwen2.5-7B-Instruct", "size": "7B", "memory_gb": 6.0},
        {"id": "Qwen/Qwen2.5-14B-Instruct", "size": "14B", "memory_gb": 10.0},
        {"id": "Qwen/Qwen2.5-32B-Instruct", "size": "32B", "memory_gb": 5.2},  # With LISA+Offload
    ]
    return {"models": models}


@app.get("/hardware")
async def get_hardware():
    """Get hardware information."""
    if LISA_AVAILABLE:
        try:
            hw = detect_hardware()
            return {
                "cpu": hw.cpu_brand,
                "cores": hw.cpu_cores,
                "ram_gb": hw.ram_total_gb,
                "gpu": hw.gpu_name,
                "max_model_normal": hw.max_model_size_normal,
                "max_model_offload": hw.max_model_size_offload,
                "recommended_groups": hw.recommended_layer_groups,
            }
        except Exception as e:
            return {"error": str(e)}
    else:
        # Simulation mode
        return {
            "cpu": "Simulation",
            "cores": 8,
            "ram_gb": 16.0,
            "gpu": "Simulation",
            "max_model_normal": "3B",
            "max_model_offload": "32B",
            "recommended_groups": 6,
        }


@app.post("/config")
async def set_config(config: ConfigRequest):
    """Set configuration."""
    # Handle both Pydantic v1 and v2
    try:
        config_dict = config.model_dump(exclude_none=True)
    except AttributeError:
        config_dict = config.dict(exclude_none=True)
    manager.config.update(config_dict)
    return {"status": "updated", "config": manager.config}


@app.post("/train")
async def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """Start training."""
    job = manager.create_job(request.model)
    
    # Start training in background
    background_tasks.add_task(
        run_training,
        job.job_id,
        request.model,
        request.data_dir,
        request.iterations,
        request.learning_rate,
        request.lisa_config,
    )
    
    return {"job_id": job.job_id, "status": "started"}


async def run_training(
    job_id: str,
    model: str,
    data_dir: str,
    iterations: int,
    learning_rate: float,
    lisa_config: Optional[Dict[str, Any]],
):
    """Run training in background."""
    manager.update_job(job_id, status="running")
    
    try:
        if LISA_AVAILABLE:
            # Real training
            config = LISAConfig(
                bottom_layers=lisa_config.get("bottom_layers", manager.config["bottom_layers"]) if lisa_config else manager.config["bottom_layers"],
                top_layers=lisa_config.get("top_layers", manager.config["top_layers"]) if lisa_config else manager.config["top_layers"],
                middle_sample=lisa_config.get("middle_sample", manager.config["middle_sample"]) if lisa_config else manager.config["middle_sample"],
                total_layers=60,
            )
            
            trainer = LISAOffloadedTrainer(
                model_id=model,
                lisa_config=config,
                max_memory_gb=manager.config["max_memory_gb"],
            )
            
            result = trainer.train(
                data_dir=data_dir,
                iterations=iterations,
                learning_rate=learning_rate,
            )
            
            manager.update_job(
                job_id,
                status="completed",
                progress=100.0,
                end_time=datetime.now().isoformat(),
                result=result,
            )
        else:
            # Simulation mode
            import time
            for i in range(iterations):
                progress = (i + 1) / iterations * 100
                manager.update_job(job_id, progress=progress)
                time.sleep(0.1)
            
            manager.update_job(
                job_id,
                status="completed",
                progress=100.0,
                end_time=datetime.now().isoformat(),
                result={"iterations": iterations, "simulated": True},
            )
    
    except Exception as e:
        manager.update_job(
            job_id,
            status="failed",
            error=str(e),
            end_time=datetime.now().isoformat(),
        )


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get training status."""
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return asdict(job)


@app.get("/jobs")
async def list_jobs():
    """List all training jobs."""
    return {"jobs": [asdict(job) for job in manager.list_jobs()]}


@app.post("/inference")
async def run_inference(request: InferenceRequest):
    """Run inference."""
    # Note: Full inference requires model loading
    # This is a simplified version
    
    if LISA_AVAILABLE:
        # Real inference would go here
        return {
            "prompt": request.prompt,
            "response": f"[LISA+Offload inference for {request.model} - Not implemented in demo]",
            "model": request.model,
        }
    else:
        # Simulation
        return {
            "prompt": request.prompt,
            "response": f"[Simulation mode - {request.model}]",
            "model": request.model,
        }


# ============================================================================
# Main
# ============================================================================

def main():
    """Run the API server."""
    print("="*60)
    print("LISA+Offload API Server")
    print("="*60)
    print(f"LISA Available: {LISA_AVAILABLE}")
    print(f"Config: {manager.config}")
    print("")
    print("Endpoints:")
    print("  GET  /              - Root")
    print("  GET  /health        - Health check")
    print("  GET  /models        - List models")
    print("  GET  /hardware      - Hardware info")
    print("  POST /config        - Set config")
    print("  POST /train         - Start training")
    print("  GET  /status/{id}   - Training status")
    print("  GET  /jobs          - List jobs")
    print("  POST /inference     - Run inference")
    print("")
    print("Starting server on http://0.0.0.0:8000")
    print("")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()