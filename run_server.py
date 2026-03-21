#!/usr/bin/env python3
"""
Run the Federated Learning Server

Usage:
    python run_server.py [--host HOST] [--port PORT] [--rounds N]

Starts the FastAPI-based federated learning server that:
- Receives gradient updates from clients
- Validates and aggregates them
- Tracks client reputation
- Saves model checkpoints
"""

import sys
import os
import argparse
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fed-server")


def main():
    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated rounds")
    parser.add_argument("--model", default="distilbert/distilgpt2", help="Model name")
    parser.add_argument("--min-clients", type=int, default=2, help="Min clients per round")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory")
    args = parser.parse_args()

    # Import after path is set
    from federated.server import FederatedServer, DEFAULT_CONFIG
    import uvicorn

    # Build config
    config = DEFAULT_CONFIG.copy()
    config["model_name"] = args.model
    config["num_rounds"] = args.rounds
    config["min_clients_per_round"] = args.min_clients
    config["checkpoint_dir"] = args.checkpoint_dir

    # Create server
    logger.info("Initializing federated server...")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Rounds: {args.rounds}")
    logger.info(f"  Min clients/round: {args.min_clients}")
    logger.info(f"  Checkpoint dir: {args.checkpoint_dir}")

    server = FederatedServer(config)

    # Get or create FastAPI app
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="LISA Federated Learning Server")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach server to app state
    app.state.server = server

    # Import models for request bodies
    from typing import Optional
    from pydantic import BaseModel

    class GradientSubmitRequest(BaseModel):
        client_id: str
        round_number: int
        timestamp: float
        num_samples: int
        gradient_norm: float
        loss_before: float
        loss_after: float
        compression_method: str
        compressed_size: int
        dp_epsilon: Optional[float] = None
        gradient_data: Optional[str] = None  # base64 encoded
        compression_info: Optional[dict] = None

    class RegisterRequest(BaseModel):
        client_id: str

    @app.get("/")
    async def root():
        return {"message": "LISA Federated Learning Server", "status": "running"}

    @app.get("/status")
    async def status():
        return server.get_status()

    @app.post("/register")
    async def register(req: RegisterRequest):
        return server.register_client(req.client_id)

    @app.post("/submit")
    async def submit(req: GradientSubmitRequest):
        update = req.model_dump()
        return server.receive_gradient(update)

    @app.get("/round/{round_num}")
    async def get_round(round_num: int):
        result = server.get_round_status(round_num)
        if result is None:
            return {"error": "Round not found"}
        return result

    @app.get("/model/{client_id}")
    async def get_model(client_id: str, since_round: int = 0):
        update = server.get_model_update(client_id, since_round)
        if update is None:
            return {"error": "No model update available"}
        return {
            "round": update["round"],
            "model_size": update["model_size"],
            "message": "Model bytes available - use POST /model/{client_id} to fetch",
        }

    # Start server
    logger.info(f"\nStarting server on http://{args.host}:{args.port}")
    logger.info("Endpoints:")
    logger.info("  GET  /              - Server info")
    logger.info("  GET  /status        - Server status")
    logger.info("  POST /register      - Register client")
    logger.info("  POST /submit        - Submit gradient")
    logger.info("  GET  /round/{n}     - Round status")
    logger.info("  GET  /model/{id}    - Get model update")
    logger.info("\nPress Ctrl+C to stop\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
