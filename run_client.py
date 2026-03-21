#!/usr/bin/env python3
"""
Run a Federated Learning Client

Usage:
    python run_client.py --client-id MY_CLIENT --server http://localhost:8000 --rounds 3

Connects to a federated learning server and participates in training rounds.
"""

import sys
import os
import argparse
import logging
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fed-client")


def main():
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--client-id", required=True, help="Unique client ID")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds to participate")
    parser.add_argument("--model", default="distilbert/distilgpt2", help="Model name")
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--max-steps", type=int, default=10, help="Max training steps per round")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--dp", action="store_true", help="Enable differential privacy")
    parser.add_argument("--dp-epsilon", type=float, default=1.0, help="DP epsilon")
    args = parser.parse_args()

    from federated.client import FederatedClient, DEFAULT_CONFIG
    import requests

    # Build config
    config = DEFAULT_CONFIG.copy()
    config["model_name"] = args.model
    config["model_name_fallback"] = args.model
    config["local_epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["max_train_steps"] = args.max_steps
    config["learning_rate"] = args.lr
    if args.dp:
        config["differential_privacy"]["enabled"] = True
        config["differential_privacy"]["epsilon"] = args.dp_epsilon

    # Create client
    logger.info("Initializing federated client...")
    logger.info(f"  Client ID: {args.client_id}")
    logger.info(f"  Server: {args.server}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Local epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Max steps: {args.max_steps}")
    logger.info(f"  Learning rate: {args.lr}")
    if args.dp:
        logger.info(f"  Differential privacy: epsilon={args.dp_epsilon}")

    client = FederatedClient(
        client_id=args.client_id,
        server_url=args.server,
        config=config,
    )

    # Register with server
    logger.info("Registering with server...")
    try:
        resp = requests.post(
            f"{args.server}/register",
            json={"client_id": args.client_id},
            timeout=30,
        )
        resp.raise_for_status()
        logger.info(f"  Registered: {resp.json()}")
    except Exception as e:
        logger.warning(f"  Server registration failed: {e} (continuing anyway)")

    # Run rounds
    logger.info(f"\nParticipating in {args.rounds} rounds...\n")

    for round_num in range(1, args.rounds + 1):
        logger.info(f"{'='*50}")
        logger.info(f"Round {round_num}/{args.rounds}")
        logger.info(f"{'='*50}")

        round_start = time.time()

        # Train locally
        train_start = time.time()
        update = client.train_and_submit(round_number=round_num)
        train_time = time.time() - train_start

        # Check result
        if update.get("status") == "error":
            logger.warning(f"  Submission failed: {update.get('message')}")
            # Server might not be running - just continue
        else:
            logger.info(f"  Training time: {train_time:.1f}s")
            logger.info(f"  Gradient submitted successfully")
            logger.info(f"  Server response: {update}")

        # Check round status
        try:
            status_resp = requests.get(
                f"{args.server}/round/{round_num}",
                timeout=10,
            )
            if status_resp.status_code == 200:
                round_status = status_resp.json()
                logger.info(f"  Round status: {round_status.get('status', 'unknown')}")
                if round_status.get("gradients_accepted"):
                    logger.info(f"  Gradients accepted: {round_status['gradients_accepted']}")
        except:
            pass

        # Wait before next round
        if round_num < args.rounds:
            time.sleep(2)

    # Final status
    logger.info(f"\n{'='*50}")
    logger.info("Client run complete")
    logger.info(f"{'='*50}")
    logger.info(f"  Total rounds: {args.rounds}")
    logger.info(f"  Client state: round={client.state.round_number}, "
                f"samples={client.state.total_samples_trained}")

    # Get final server status
    try:
        status_resp = requests.get(f"{args.server}/status", timeout=10)
        if status_resp.status_code == 200:
            server_status = status_resp.json()
            logger.info(f"\n  Server status:")
            logger.info(f"    Global round: {server_status.get('global_round', 'N/A')}")
            logger.info(f"    Clients: {server_status.get('clients_registered', 'N/A')} registered, "
                        f"{server_status.get('active_clients', 'N/A')} active")
            logger.info(f"    Gradients received: {server_status.get('total_gradients_received', 'N/A')}")
            logger.info(f"    Model size: {server_status.get('current_model_size_mb', 'N/A')} MB")
    except Exception as e:
        logger.warning(f"  Could not fetch server status: {e}")


if __name__ == "__main__":
    main()
