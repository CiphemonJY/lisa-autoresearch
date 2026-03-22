#!/usr/bin/env python3
"""
P2P Discovery and Networking Layer for Federated Learning

Provides:
- P2PRegistry: lightweight peer discovery via a known bootstrap server
- P2PClient: optional peer-to-peer gradient exchange

In production this would use a distributed hash table (DHT) like Kademlia
or a libp2p/go-libp2p stack. Here we use a simple HTTP-based registry
hosted by the bootstrap server for demonstration purposes.

Key design decisions:
- P2P is opt-in (--p2p-enable flag)
- The bootstrap server also registers as a peer so clients can find it
- Gradient exchange uses a simple averaging scheme (FedAvg)
- Real systems would use Krum, CoMed, or Bulyan for Byzantine-resilient aggregation
"""

import os
import sys
import json
import time
import threading
import hashlib
import logging
import struct
import socket
import socketserver
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from urllib.request import urlopen, Request
from urllib.error import URLError

import numpy as np

logger = logging.getLogger("p2p")


# ============================================================================
# Peer Info
# ============================================================================

@dataclass
class PeerInfo:
    """Information about a known peer."""
    peer_id: str
    address: str  # host:port
    registered_at: float
    last_heartbeat: float
    is_bootstrap: bool = False


# ============================================================================
# P2P Registry (Bootstrap Server Side)
# ============================================================================

class P2PRegistry:
    """
    Lightweight peer discovery service.

    In a real deployment this would use a distributed hash table (DHT).
    Here we use a simple HTTP-based approach:
    - Bootstrap server exposes REST endpoints to register/heartbeat/query peers
    - Clients poll periodically to discover new peers and stay in the list

    The registry stores:
    - List of known peers (host:port addresses)
    - Their last heartbeat timestamp (for liveness)
    """

    def __init__(self, bootstrap_server: str, port: int, my_address: Optional[str] = None):
        """
        Args:
            bootstrap_server: Address of the bootstrap server (host:port or URL).
                             If this IS the bootstrap server, use --bootstrap mode instead.
            port: Local port this client will listen on for P2P connections.
            my_address: Public address to advertise to other peers
                         (defaults to bootstrap_server if not set).
        """
        self.port = port
        self.my_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
        self.bootstrap_server = bootstrap_server.rstrip("/")
        # My advertised address - use provided address or derive from bootstrap server
        self.my_address = my_address or f"{self._host_from_url(self.bootstrap_server)}:{port}"
        self.peers: List[PeerInfo] = []
        self._lock = threading.RLock()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_heartbeat = threading.Event()

        # HTTP paths on the bootstrap server
        self._register_path = f"{self.bootstrap_server}/p2p/register"
        self._heartbeat_path = f"{self.bootstrap_server}/p2p/heartbeat"
        self._peers_path = f"{self.bootstrap_server}/p2p/peers"

    @staticmethod
    def _host_from_url(url: str) -> str:
        """Extract host from URL like http://127.0.0.1:8081."""
        return url.split("://")[1].split(":")[0] if "://" in url else url.split(":")[0]

    def register(self) -> List[str]:
        """
        Register this client with the bootstrap server.

        Returns:
            List of known peer addresses (host:port strings).
        """
        payload = json.dumps({
            "peer_id": self.my_id,
            "address": self.my_address,
            "port": self.port,
        }).encode("utf-8")

        try:
            req = Request(
                self._register_path,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                peer_addresses: List[str] = data.get("peers", [])

                with self._lock:
                    self.peers = [
                        PeerInfo(
                            peer_id=p.get("peer_id", ""),
                            address=p.get("address", ""),
                            registered_at=time.time(),
                            last_heartbeat=time.time(),
                            is_bootstrap=False,
                        )
                        for p in peer_addresses
                        if p.get("address") != self.my_address
                    ]

                logger.info(f"P2P registered: {len(peer_addresses)} known peers")
                return peer_addresses

        except (URLError, OSError, json.JSONDecodeError) as e:
            logger.warning(f"P2P registration failed (bootstrap unreachable): {e}")
            return []

    def heartbeat(self):
        """Send a heartbeat to the bootstrap server to keep registration alive."""
        try:
            payload = json.dumps({
                "peer_id": self.my_id,
                "address": self.my_address,
            }).encode("utf-8")

            req = Request(
                self._heartbeat_path,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=5):
                pass  # 200 OK means heartbeat recorded
        except (URLError, OSError):
            pass  # Best-effort heartbeat

    def get_peers(self) -> List[str]:
        """
        Fetch the latest list of known peers from the bootstrap server.

        Returns:
            List of peer addresses (host:port strings).
        """
        try:
            req = Request(
                self._peers_path,
                headers={"Accept": "application/json"},
                method="GET",
            )
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                peer_addresses: List[str] = data.get("peers", [])

                with self._lock:
                    self.peers = [
                        PeerInfo(
                            peer_id=p.get("peer_id", ""),
                            address=p.get("address", ""),
                            registered_at=time.time(),
                            last_heartbeat=time.time(),
                            is_bootstrap=False,
                        )
                        for p in peer_addresses
                        if p.get("address") != self.my_address
                    ]

                return peer_addresses

        except (URLError, OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to fetch peer list: {e}")
            # Return stale cached list on failure
            with self._lock:
                return [p.address for p in self.peers if p.address != self.my_address]

    def start_heartbeat(self, interval_secs: float = 30.0):
        """Start a background thread that sends heartbeats periodically."""
        def _heartbeat_loop():
            while not self._stop_heartbeat.is_set():
                self.heartbeat()
                # Also refresh peer list occasionally
                self.get_peers()
                self._stop_heartbeat.wait(timeout=interval_secs)

        self._heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        logger.info(f"P2P heartbeat started (interval={interval_secs}s)")

    def stop_heartbeat(self):
        """Stop the background heartbeat thread."""
        self._stop_heartbeat.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2)


# ============================================================================
# P2P Gradient Exchange Server
# ============================================================================

class P2PExchangeHandler(socketserver.BaseRequestHandler):
    """
    Handle incoming P2P gradient exchange requests.

    Protocol:
      1. Peer sends JSON: {"type": "gradient_request", "from_peer": "...", "round": N}
      2. Server sends JSON response: {"type": "gradient_response", "gradients": {...}}
      3. Or sends {"type": "no_gradient"} if nothing to share
    """

    def handle(self):
        try:
            # Receive JSON header (4-byte length prefix)
            len_header = self._recv_exact(4)
            if len(len_header) < 4:
                return
            msg_len = struct.unpack("!I", len_header)[0]

            msg_data = self._recv_exact(msg_len)
            if len(msg_data) < msg_len:
                return

            msg = json.loads(msg_data.decode("utf-8"))
            msg_type = msg.get("type", "")

            server = self.server.p2p_server
            from_peer = msg.get("from_peer", "unknown")
            round_num = msg.get("round", 0)

            logger.info(f"P2P [{from_peer}] request (round {round_num}, type={msg_type})")

            if msg_type == "gradient_request":
                self._handle_gradient_request(server, from_peer, round_num)
            else:
                logger.warning(f"P2P unknown message type: {msg_type}")

        except Exception as e:
            logger.error(f"P2P handler error: {e}")

    def _recv_exact(self, n: int) -> bytes:
        """Receive exactly n bytes."""
        data = b""
        while len(data) < n:
            try:
                chunk = self.request.recv(n - len(data))
                if not chunk:
                    return data
                data += chunk
            except socket.timeout:
                return data
        return data

    def _handle_gradient_request(self, server, from_peer: str, round_num: int):
        """Handle a peer's request for our gradients."""
        import pickle

        # Check if we have gradients to share for this round
        with server._lock:
            has_grads = (
                server.latest_gradients is not None
                and server.latest_round >= round_num
            )

        if has_grads:
            with server._lock:
                grad_data = pickle.dumps(server.latest_gradients)
                n_tensors = len(server.latest_gradients)

            # Send gradient response
            resp = {
                "type": "gradient_response",
                "round": server.latest_round,
                "n_tensors": n_tensors,
            }
            resp_bytes = json.dumps(resp).encode("utf-8")
            try:
                self.request.sendall(struct.pack("!I", len(resp_bytes)) + resp_bytes)
                self.request.sendall(struct.pack("!I", len(grad_data)) + grad_data)
                logger.info(f"P2P [{from_peer}] sent {n_tensors} gradients (round {server.latest_round})")
            except (ConnectionResetError, BrokenPipeError, OSError):
                logger.warning(f"P2P [{from_peer}] disconnected during gradient send")
        else:
            # Nothing to share
            resp = {"type": "no_gradient", "round": server.latest_round}
            resp_bytes = json.dumps(resp).encode("utf-8")
            try:
                self.request.sendall(struct.pack("!I", len(resp_bytes)) + resp_bytes)
            except (ConnectionResetError, BrokenPipeError, OSError):
                pass


class P2PExchangeServer(socketserver.ThreadingTCPServer):
    """TCP server for P2P gradient exchange."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, port: int, p2p_server_ref):
        self.p2p_server = p2p_server_ref  # Reference to P2PClient
        self.latest_gradients: Optional[Dict] = None
        self.latest_round: int = 0
        self._lock = threading.RLock()
        super().__init__(("0.0.0.0", port), P2PExchangeHandler)
        logger.info(f"P2P exchange server listening on port {port}")


# ============================================================================
# P2P Client
# ============================================================================

class P2PClient:
    """
    Peer-to-peer gradient exchange client.

    Allows clients to exchange gradients directly with peers, not just
    through the central federated server. This reduces single-point-of-failure
    and can speed up convergence when the central server is a bottleneck.

    Simplified design (production would differ significantly):
    - Uses a simple averaging scheme (FedAvg) for combining peer gradients
    - No Byzantine fault tolerance (Krum, CoMed, Bulyan would be used in prod)
    - No sophisticated gossip protocol - just request/response with each peer
    - No peer reputation or trust scoring
    """

    def __init__(self, my_id: str, registry: P2PRegistry, port: int = 0):
        """
        Args:
            my_id: Unique identifier for this client.
            registry: P2PRegistry instance for peer discovery.
            port: Local port to listen on for incoming P2P requests (0 = any).
        """
        self.my_id = my_id
        self.registry = registry
        self.port = port
        self.peers: List[str] = []  # List of peer addresses (host:port)
        self._exchange_server: Optional[P2PExchangeServer] = None
        self._lock = threading.RLock()

    def start_exchange_server(self, port: Optional[int] = None):
        """Start the TCP server that handles incoming peer requests."""
        listen_port = port or self.port
        if listen_port == 0:
            listen_port = 8090  # Default P2P exchange port

        self._exchange_server = P2PExchangeServer(listen_port, self)
        self.port = listen_port
        logger.info(f"P2P exchange server started on port {listen_port}")

    def update_local_gradient(self, gradients: Dict, round_num: int):
        """
        Store our latest gradient for sharing with peers.

        Called by the federated client after local training.
        Peers can request this gradient via sync_with_peers().
        """
        if self._exchange_server:
            with self._exchange_server._lock:
                self._exchange_server.latest_gradients = gradients
                self._exchange_server.latest_round = round_num

    def sync_with_peers(self, my_grads: Dict) -> Dict:
        """
        Exchange gradients with all known peers.

        Sends our gradients to all peers and collects theirs.
        Returns a combined gradient dict by averaging received gradients
        with our own (weighted equally, simplified FedAvg).

        In a real production system this would use:
        - Krum / multi-Krum for Byzantine-resilient aggregation
        - CoMed for distributed robust optimization
        - Bulyan for combined robustness
        - Gossip protocols for eventually-consistent state
        - Secure aggregation (CryptoCollective) to prevent honest-but-curious servers

        Args:
            my_grads: Our local gradient state dict {name: np.array}

        Returns:
            Averaged gradient dict {name: np.array}
        """
        import pickle

        peer_addresses = self.registry.get_peers()
        if not peer_addresses:
            logger.info("P2P sync: no peers available, returning own gradients")
            return my_grads

        received_grads: List[Dict] = []

        for addr in peer_addresses:
            try:
                peer_grads = self._request_gradients_from_peer(addr)
                if peer_grads:
                    received_grads.append(peer_grads)
                    logger.info(f"P2P received {len(peer_grads)} tensors from {addr}")
            except Exception as e:
                logger.warning(f"P2P failed to get gradients from {addr}: {e}")
                continue

        if not received_grads:
            logger.info("P2P sync: no gradients received from peers, returning own")
            return my_grads

        # Simple FedAvg: average all gradients (including our own)
        # All peers get equal weight (simplified - real systems weight by sample count)
        all_grads = [my_grads] + received_grads
        return self._average_gradients(all_grads)

    def _request_gradients_from_peer(
        self, peer_addr: str, timeout: float = 10.0
    ) -> Optional[Dict]:
        """Request the latest gradients from a specific peer."""
        import pickle

        host, port_str = peer_addr.rsplit(":", 1)
        port = int(port_str)

        msg = {
            "type": "gradient_request",
            "from_peer": self.my_id,
            "round": 0,  # Request latest available
        }
        msg_bytes = json.dumps(msg).encode("utf-8")

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((host, port))

            # Send request
            sock.sendall(struct.pack("!I", len(msg_bytes)) + msg_bytes)

            # Receive response header
            len_header = b""
            while len(len_header) < 4:
                chunk = sock.recv(4 - len(len_header))
                if not chunk:
                    sock.close()
                    return None
                len_header += chunk

            resp_len = struct.unpack("!I", len_header)[0]
            resp_data = b""
            while len(resp_data) < resp_len:
                chunk = sock.recv(resp_len - len(resp_data))
                if not chunk:
                    break
                resp_data += chunk

            resp = json.loads(resp_data.decode("utf-8"))
            resp_type = resp.get("type", "")

            if resp_type == "gradient_response":
                # Receive gradient payload
                payload_len_header = b""
                while len(payload_len_header) < 4:
                    chunk = sock.recv(4 - len(payload_len_header))
                    if not chunk:
                        break
                    payload_len_header += chunk

                if len(payload_len_header) < 4:
                    sock.close()
                    return None

                payload_len = struct.unpack("!I", payload_len_header)[0]
                payload = b""
                while len(payload) < payload_len:
                    chunk = sock.recv(payload_len - len(payload))
                    if not chunk:
                        break
                    payload += chunk

                gradients: Dict = pickle.loads(payload)
                sock.close()
                return gradients

            elif resp_type == "no_gradient":
                sock.close()
                return None
            else:
                sock.close()
                return None

        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            logger.warning(f"P2P request to {peer_addr} failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"P2P request to {peer_addr} error: {e}")
            return None

    @staticmethod
    def _average_gradients(grad_list: List[Dict]) -> Dict:
        """
        Average multiple gradient dicts element-wise (FedAvg with equal weights).

        Args:
            grad_list: List of gradient state dicts to average.

        Returns:
            Averaged gradient dict.
        """
        if not grad_list:
            return {}

        if len(grad_list) == 1:
            return grad_list[0]

        result = {}
        keys = grad_list[0].keys()

        for key in keys:
            tensors = []
            for grads in grad_list:
                if key in grads:
                    t = grads[key]
                    if isinstance(t, np.ndarray):
                        tensors.append(t.astype(np.float64))
                    else:
                        tensors.append(np.array(t, dtype=np.float64))

            if tensors:
                # Simple mean - real system would weight by sample count
                stacked = np.stack(tensors, axis=0)
                result[key] = np.mean(stacked, axis=0).astype(np.float32)

        return result

    def stop(self):
        """Stop the P2P exchange server."""
        if self._exchange_server:
            self._exchange_server.shutdown()
            self._exchange_server = None
            logger.info("P2P exchange server stopped")


# ============================================================================
# Bootstrap Server (Standalone P2P Registry Service)
# ============================================================================

class P2PBootstrapServer:
    """
    Standalone bootstrap/registry server that peers can register with.

    This is the bootstrap node that holds the peer list. All peers
    register on startup and query periodically to discover each other.

    HTTP endpoints:
      POST /p2p/register   - Register a new peer
      POST /p2p/heartbeat  - Keep registration alive
      GET  /p2p/peers      - Get current peer list
      GET  /p2p/health     - Health check

    Usage:
        python -m federated.p2p --bootstrap --port 8081
        python -m federated.p2p --bootstrap --port 8081  # On another machine
    """

    def __init__(self, port: int = 8081):
        self.port = port
        self.peers: Dict[str, Dict] = {}  # peer_id -> peer info
        self._lock = threading.RLock()
        self._heartbeat_timeout_secs = 120.0

        # Note: We could use FastAPI here but to keep dependencies low,
        # we use a simple built-in HTTP server instead
        import http.server
        import socketserver

        class P2PHandler(http.server.BaseHTTPRequestHandler):
            _registry = self  # type: ignore

            def _send_json(self, status: int, data: dict):
                body = json.dumps(data).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", len(body))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/p2p/health":
                    self._send_json(200, {"status": "ok", "peers": len(self._registry.peers)})
                elif self.path == "/p2p/peers":
                    with self._registry._lock:
                        # Remove stale peers
                        now = time.time()
                        stale = [
                            pid for pid, p in self._registry.peers.items()
                            if now - p.get("last_heartbeat", 0) > self._registry._heartbeat_timeout_secs
                        ]
                        for pid in stale:
                            del self._registry.peers[pid]

                        peer_list = [
                            {"peer_id": pid, "address": p["address"]}
                            for pid, p in self._registry.peers.items()
                        ]
                    self._send_json(200, {"peers": peer_list})
                else:
                    self._send_json(404, {"error": "not found"})

            def do_POST(self):
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len) if content_len > 0 else b""

                if self.path == "/p2p/register":
                    try:
                        data = json.loads(body.decode("utf-8"))
                        peer_id = data.get("peer_id", "")
                        address = data.get("address", "")

                        if not peer_id or not address:
                            self._send_json(400, {"error": "peer_id and address required"})
                            return

                        with self._registry._lock:
                            now = time.time()
                            # Include all peers except the new one in the response
                            existing = [
                                {"peer_id": pid, "address": p["address"]}
                                for pid, p in self._registry.peers.items()
                                if pid != peer_id
                            ]
                            self._registry.peers[peer_id] = {
                                "address": address,
                                "registered_at": now,
                                "last_heartbeat": now,
                            }

                        logger.info(f"P2P bootstrap: registered {peer_id} @ {address} ({len(existing)} existing peers)")
                        self._send_json(200, {"status": "registered", "peers": existing})

                    except (json.JSONDecodeError, KeyError) as e:
                        self._send_json(400, {"error": str(e)})

                elif self.path == "/p2p/heartbeat":
                    try:
                        data = json.loads(body.decode("utf-8"))
                        peer_id = data.get("peer_id", "")

                        with self._registry._lock:
                            if peer_id in self._registry.peers:
                                self._registry.peers[peer_id]["last_heartbeat"] = time.time()

                        self._send_json(200, {"status": "ok"})
                    except json.JSONDecodeError:
                        self._send_json(400, {"error": "invalid json"})

                else:
                    self._send_json(404, {"error": "not found"})

            def log_message(self, format, *args):
                # Suppress default HTTP logging, use our logger instead
                logger.info(f"[HTTP] {format % args}")

        self.handler = P2PHandler
        self.httpd: Optional[socketserver.TCPServer] = None

    def start(self):
        """Start the bootstrap server."""
        import http.server
        import socketserver

        socketserver.TCPServer.allow_reuse_address = True
        self.httpd = socketserver.TCPServer(("0.0.0.0", self.port), self.handler)
        logger.info(f"P2P bootstrap server listening on port {self.port}")
        logger.info(f"  Register: POST /p2p/register")
        logger.info(f"  Heartbeat: POST /p2p/heartbeat")
        logger.info(f"  Peers: GET /p2p/peers")
        self.httpd.serve_forever()

    def stop(self):
        """Stop the bootstrap server."""
        if self.httpd:
            self.httpd.shutdown()
            self.httpd = None


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="P2P Discovery Layer - run as bootstrap node or show help"
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Run as a P2P bootstrap/registry server (enables peer discovery)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8081,
        help="Port for bootstrap server (default 8081)",
    )

    args = parser.parse_args()

    if args.bootstrap:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [p2p-bootstrap] %(message)s",
        )
        server = P2PBootstrapServer(port=args.port)
        try:
            server.start()
        except KeyboardInterrupt:
            server.stop()
    else:
        print("P2P Discovery Layer")
        print("=" * 50)
        print()
        print("To run as a bootstrap node (enables peer discovery):")
        print(f"  python -m federated.p2p --bootstrap --port 8081")
        print()
        print("To connect a client with P2P enabled:")
        print(f"  python federated/client.py --p2p-enable --bootstrap-server 127.0.0.1:8081")
        print()
        print("The bootstrap server exposes:")
        print("  POST /p2p/register   - Register a new peer")
        print("  POST /p2p/heartbeat  - Keep registration alive")
        print("  GET  /p2p/peers      - Get current peer list")


if __name__ == "__main__":
    main()
