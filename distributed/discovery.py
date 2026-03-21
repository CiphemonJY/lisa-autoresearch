"""
Network Discovery Module for P2P Training

Supports multiple discovery methods:
1. DHT (Distributed Hash Table) - Decentralized peer discovery
2. Bootstrap Nodes - Centralized peer discovery
3. Direct Connections - Manual peer list
4. Local Network - LAN discovery

Users can select which method to use based on their needs.
"""

import os
import sys
import json
import time
import socket
import hashlib
import secrets
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging
import queue

os.environ['PYTHONWARNINGS'] = 'ignore::urllib3.warnings.NotOpenSSLWarning'

# Optional imports
try:
    import asyncio
    ASYNCIO_AVAILABLE = True
except ImportError:
    ASYNCIO_AVAILABLE = False

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


# ============================================================================
# Configuration
# ============================================================================

DISCOVERY_CONFIG = {
    # General
    "node_id_length": 16,
    "heartbeat_interval": 30,  # seconds
    "peer_timeout": 300,       # seconds before peer considered offline
    
    # DHT
    "dht_enabled": True,
    "dht_port": 6881,
    "dht_bootstrap_nodes": [
        ("router.bittorrent.com", 6881),
        ("dht.transmissionbt.com", 6881),
    ],
    "dht_bucket_size": 8,
    "dht_republish_interval": 3600,  # seconds
    
    # Bootstrap
    "bootstrap_nodes": [],
    "bootstrap_timeout": 30,  # seconds
    
    # Direct
    "direct_peers": [],
    
    # Local network
    "local_discovery_enabled": True,
    "local_port": 6882,
    "local_broadcast_interval": 60,  # seconds
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PeerInfo:
    """Information about a peer."""
    peer_id: str
    address: str
    port: int
    public_key: Optional[bytes] = None
    capabilities: List[str] = field(default_factory=list)
    reputation: float = 50.0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveryResult:
    """Result from discovery operation."""
    success: bool
    peers: List[PeerInfo] = field(default_factory=list)
    error: Optional[str] = None
    method: str = "unknown"


# ============================================================================
# Base Discovery Class
# ============================================================================

class BaseDiscovery:
    """Base class for discovery methods."""
    
    def __init__(self, config: Dict = None):
        self.config = config or DISCOVERY_CONFIG
        self.logger = logging.getLogger("discovery")
        self.peers: Dict[str, PeerInfo] = {}
        self.node_id = secrets.token_hex(self.config["node_id_length"])
    
    def discover(self) -> DiscoveryResult:
        """Discover peers. Override in subclasses."""
        raise NotImplementedError
    
    def announce(self):
        """Announce presence to network. Override in subclasses."""
        raise NotImplementedError
    
    def get_peers(self) -> List[PeerInfo]:
        """Get discovered peers."""
        return list(self.peers.values())
    
    def add_peer(self, peer: PeerInfo):
        """Add or update peer."""
        if peer.peer_id in self.peers:
            self.peers[peer.peer_id].last_seen = time.time()
        else:
            self.peers[peer.peer_id] = peer
    
    def remove_stale_peers(self):
        """Remove peers not seen recently."""
        now = time.time()
        timeout = self.config["peer_timeout"]
        stale = [
            peer_id for peer_id, peer in self.peers.items()
            if now - peer.last_seen > timeout
        ]
        for peer_id in stale:
            del self.peers[peer_id]
            self.logger.info(f"Removed stale peer: {peer_id}")


# ============================================================================
# DHT Discovery (Like BitTorrent)
# ============================================================================

class DHTDiscovery(BaseDiscovery):
    """
    Distributed Hash Table discovery.
    
    Like BitTorrent's DHT, but for finding training peers.
    
    How it works:
    1. Node generates unique ID (hash of public key)
    2. Contacts bootstrap nodes to enter network
    3. Maintains routing table of other nodes
    4. Searches for peers by topic (e.g., "lisa-training-32b")
    5. Announces own presence for topic
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.logger = logging.getLogger("dht_discovery")
        self.routing_table: Dict[str, List[PeerInfo]] = {}
        self.topics: Dict[str, List[str]] = {}  # topic -> peer_ids
    
    def discover(self, topic: str = None) -> DiscoveryResult:
        """
        Discover peers via DHT.
        
        Args:
            topic: Topic to search for (e.g., "lisa-training-32b")
        
        Returns:
            DiscoveryResult with discovered peers
        """
        self.logger.info(f"Discovering peers via DHT for topic: {topic}")
        
        # In production, this would:
        # 1. Contact bootstrap nodes
        # 2. Query DHT for peers
        # 3. Validate peer responses
        # 4. Return discovered peers
        
        # Simulation for demo
        discovered = []
        
        # Try to connect to bootstrap nodes
        for host, port in self.config["dht_bootstrap_nodes"]:
            try:
                peer = self._try_connect(host, port, topic)
                if peer:
                    discovered.append(peer)
            except Exception as e:
                self.logger.warning(f"Failed to connect to {host}:{port}: {e}")
        
        # Add discovered peers
        for peer in discovered:
            self.add_peer(peer)
        
        return DiscoveryResult(
            success=len(discovered) > 0,
            peers=discovered,
            method="dht",
        )
    
    def announce(self, topic: str, port: int):
        """
        Announce presence for topic.
        
        Args:
            topic: Topic to announce for
            port: Port to listen on
        """
        self.logger.info(f"Announcing presence for topic: {topic} on port {port}")
        
        # In production, this would:
        # 1. Generate announce message
        # 2. Send to DHT nodes
        # 3. Store in topic list
        
        # Track topic
        if topic not in self.topics:
            self.topics[topic] = []
        self.topics[topic].append(self.node_id)
        
        self.logger.info(f"Announced for topic: {topic}")
    
    def _try_connect(self, host: str, port: int, topic: str) -> Optional[PeerInfo]:
        """Try to connect to a DHT node."""
        # Simulation - in production would use actual DHT protocol
        try:
            # Try to resolve and connect
            socket.gethostbyname(host)
            
            # Return simulated peer
            return PeerInfo(
                peer_id=secrets.token_hex(8),
                address=host,
                port=port,
                capabilities=["training", "inference"],
            )
        except Exception:
            return None


# ============================================================================
# Bootstrap Discovery
# ============================================================================

class BootstrapDiscovery(BaseDiscovery):
    """
    Bootstrap node discovery.
    
    Uses predefined bootstrap nodes to find peers.
    
    How it works:
    1. Node contacts bootstrap server
    2. Bootstrap server returns list of active peers
    3. Node connects to returned peers
    4. Regular heartbeat to bootstrap server
    
    More centralized than DHT, but simpler.
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.logger = logging.getLogger("bootstrap_discovery")
    
    def discover(self) -> DiscoveryResult:
        """
        Discover peers via bootstrap server.
        
        Returns:
            DiscoveryResult with discovered peers
        """
        self.logger.info("Discovering peers via bootstrap nodes")
        
        discovered = []
        
        # Try each bootstrap node
        for bootstrap in self.config["bootstrap_nodes"]:
            try:
                peers = self._query_bootstrap(bootstrap)
                discovered.extend(peers)
            except Exception as e:
                self.logger.warning(f"Bootstrap failed: {bootstrap}: {e}")
        
        # Add discovered peers
        for peer in discovered:
            self.add_peer(peer)
        
        return DiscoveryResult(
            success=len(discovered) > 0,
            peers=discovered,
            method="bootstrap",
        )
    
    def announce(self, port: int):
        """Announce to bootstrap server."""
        self.logger.info(f"Announcing to bootstrap server on port {port}")
        
        for bootstrap in self.config["bootstrap_nodes"]:
            try:
                self._register_with_bootstrap(bootstrap, port)
            except Exception as e:
                self.logger.warning(f"Failed to register: {e}")
    
    def _query_bootstrap(self, bootstrap: str) -> List[PeerInfo]:
        """Query bootstrap server for peers."""
        # Simulation - in production would use HTTP/WebSocket
        return []
    
    def _register_with_bootstrap(self, bootstrap: str, port: int):
        """Register with bootstrap server."""
        # Simulation - in production would use HTTP/WebSocket
        pass


# ============================================================================
# Direct Discovery
# ============================================================================

class DirectDiscovery(BaseDiscovery):
    """
    Direct peer list discovery.
    
    Uses manually configured peer list.
    
    How it works:
    1. Node has list of known peers
    2. Connects directly to each peer
    3. Exchanges peer lists
    4. Updates local peer list
    
    Most controlled approach, good for private networks.
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.logger = logging.getLogger("direct_discovery")
    
    def discover(self) -> DiscoveryResult:
        """
        Discover peers from direct list.
        
        Returns:
            DiscoveryResult with discovered peers
        """
        self.logger.info("Discovering peers from direct list")
        
        discovered = []
        
        # Try each direct peer
        for peer_addr in self.config["direct_peers"]:
            try:
                peer = self._connect_peer(peer_addr)
                if peer:
                    discovered.append(peer)
            except Exception as e:
                self.logger.warning(f"Failed to connect to {peer_addr}: {e}")
        
        # Add discovered peers
        for peer in discovered:
            self.add_peer(peer)
        
        return DiscoveryResult(
            success=len(discovered) > 0,
            peers=discovered,
            method="direct",
        )
    
    def announce(self):
        """No announcement needed for direct discovery."""
        self.logger.info("Direct discovery does not require announcement")
    
    def add_direct_peer(self, address: str, port: int):
        """Add peer to direct list."""
        self.config["direct_peers"].append(f"{address}:{port}")
    
    def _connect_peer(self, peer_addr: str) -> Optional[PeerInfo]:
        """Connect to a direct peer."""
        try:
            parts = peer_addr.split(":")
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 6881
            
            # Try to connect
            # In production would use actual socket connection
            return PeerInfo(
                peer_id=secrets.token_hex(8),
                address=host,
                port=port,
                capabilities=["training"],
            )
        except Exception:
            return None


# ============================================================================
# Local Network Discovery
# ============================================================================

class LocalDiscovery(BaseDiscovery):
    """
    Local network discovery via broadcast/multicast.
    
    Finds peers on the same LAN automatically.
    
    How it works:
    1. Node broadcasts presence on LAN
    2. Other nodes respond with their info
    3. All nodes discover each other
    4. Regular heartbeat to maintain presence
    
    Good for office/home networks.
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.logger = logging.getLogger("local_discovery")
        self.broadcast_socket = None
        self.listen_thread = None
        self.running = False
    
    def discover(self) -> DiscoveryResult:
        """
        Discover peers on local network.
        
        Returns:
            DiscoveryResult with discovered peers
        """
        self.logger.info("Discovering peers on local network")
        
        discovered = []
        
        # Broadcast presence
        self._broadcast_presence()
        
        # Listen for responses
        # In production would listen for responses
        
        return DiscoveryResult(
            success=True,
            peers=discovered,
            method="local",
        )
    
    def announce(self, port: int):
        """Announce presence on local network."""
        self.logger.info(f"Announcing on local network port {port}")
        self._broadcast_presence()
    
    def start_listening(self, port: int):
        """Start listening for broadcasts."""
        if self.running:
            return
        
        self.running = True
        self.port = port
        
        # In production would create UDP socket and listen
        self.logger.info(f"Listening for broadcasts on port {port}")
    
    def stop_listening(self):
        """Stop listening."""
        self.running = False
        if self.broadcast_socket:
            self.broadcast_socket.close()
    
    def _broadcast_presence(self):
        """Broadcast presence on local network."""
        # In production would send UDP broadcast
        self.logger.info("Broadcasting presence on local network")


# ============================================================================
# Discovery Manager (Select Method)
# ============================================================================

class DiscoveryManager:
    """
    Manages multiple discovery methods.
    
    Users can select which method(s) to use:
    - "dht": DHT discovery (BitTorrent-style)
    - "bootstrap": Bootstrap server discovery
    - "direct": Direct peer list
    - "local": Local network discovery
    - "all": Use all methods
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or DISCOVERY_CONFIG
        self.logger = logging.getLogger("discovery_manager")
        
        # Initialize discovery methods
        self.discoverers = {
            "dht": DHTDiscovery(config),
            "bootstrap": BootstrapDiscovery(config),
            "direct": DirectDiscovery(config),
            "local": LocalDiscovery(config),
        }
        
        self.peers: Dict[str, PeerInfo] = {}
    
    def discover(self, methods: List[str] = None) -> DiscoveryResult:
        """
        Discover peers using specified methods.
        
        Args:
            methods: List of methods to use, or ["all"]
        
        Returns:
            Combined DiscoveryResult
        """
        if methods is None or "all" in methods:
            methods = list(self.discoverers.keys())
        
        self.logger.info(f"Discovering peers using methods: {methods}")
        
        all_peers = []
        errors = []
        
        for method in methods:
            if method not in self.discoverers:
                self.logger.warning(f"Unknown discovery method: {method}")
                continue
            
            self.logger.info(f"Trying discovery method: {method}")
            
            try:
                result = self.discoverers[method].discover()
                
                if result.success:
                    all_peers.extend(result.peers)
                    self.logger.info(f"Discovered {len(result.peers)} peers via {method}")
                else:
                    errors.append(f"{method}: {result.error}")
                    self.logger.warning(f"Discovery failed for {method}: {result.error}")
            
            except Exception as e:
                errors.append(f"{method}: {str(e)}")
                self.logger.error(f"Discovery error for {method}: {e}")
        
        # Deduplicate by peer_id
        seen = set()
        unique_peers = []
        for peer in all_peers:
            if peer.peer_id not in seen:
                seen.add(peer.peer_id)
                unique_peers.append(peer)
                self.peers[peer.peer_id] = peer
        
        return DiscoveryResult(
            success=len(unique_peers) > 0,
            peers=unique_peers,
            error="; ".join(errors) if errors else None,
            method="+".join(methods),
        )
    
    def announce(self, methods: List[str] = None, port: int = 6881):
        """
        Announce presence using specified methods.
        
        Args:
            methods: List of methods to use, or ["all"]
            port: Port to announce
        """
        if methods is None or "all" in methods:
            methods = list(self.discoverers.keys())
        
        for method in methods:
            if method in self.discoverers:
                try:
                    self.discoverers[method].announce(port if method != "dht" else None)
                    self.logger.info(f"Announced via {method}")
                except Exception as e:
                    self.logger.error(f"Announce error for {method}: {e}")
    
    def get_peers(self) -> List[PeerInfo]:
        """Get all discovered peers."""
        return list(self.peers.values())
    
    def add_direct_peer(self, address: str, port: int):
        """Add a direct peer."""
        self.discoverers["direct"].add_direct_peer(address, port)
    
    def set_bootstrap_nodes(self, nodes: List[str]):
        """Set bootstrap nodes."""
        self.discoverers["bootstrap"].config["bootstrap_nodes"] = nodes
    
    def start_local_discovery(self, port: int = 6882):
        """Start local network discovery."""
        self.discoverers["local"].start_listening(port)
    
    def stop_local_discovery(self):
        """Stop local network discovery."""
        self.discoverers["local"].stop_listening()


# ============================================================================
# Main
# ============================================================================

def main():
    """Demo discovery methods."""
    print("="*70)
    print("P2P TRAINING - DISCOVERY METHODS")
    print("="*70)
    print()
    
    # Create discovery manager
    manager = DiscoveryManager()
    
    print("Available Discovery Methods:")
    print("-"*70)
    print()
    
    print("1. DHT (Distributed Hash Table)")
    print("   - Decentralized peer discovery")
    print("   - Like BitTorrent's DHT")
    print("   - No central server needed")
    print("   - Good for public networks")
    print()
    
    print("2. Bootstrap")
    print("   - Centralized peer discovery")
    print("   - Bootstrap server provides peer list")
    print("   - Simpler than DHT")
    print("   - Good for controlled networks")
    print()
    
    print("3. Direct")
    print("   - Manual peer list")
    print("   - Most controlled")
    print("   - No discovery needed")
    print("   - Good for private networks")
    print()
    
    print("4. Local Network")
    print("   - Automatic LAN discovery")
    print("   - Broadcast/multicast")
    print("   - No configuration needed")
    print("   - Good for office/home networks")
    print()
    
    print("="*70)
    print("USAGE EXAMPLES")
    print("="*70)
    print()
    
    print("# Use all discovery methods")
    print('result = manager.discover(["all"])')
    print()
    
    print("# Use specific methods")
    print('result = manager.discover(["dht", "local"])')
    print()
    
    print("# Add direct peers")
    print('manager.add_direct_peer("192.168.1.100", 6881)')
    print()
    
    print("# Set bootstrap servers")
    print('manager.set_bootstrap_nodes(["bootstrap1.example.com:6881"])')
    print()
    
    print("# Announce presence")
    print('manager.announce(port=6881)')
    print()
    
    print("="*70)
    print("DISCOVERY SELECTION")
    print("="*70)
    print()
    
    print("User can select discovery method based on needs:")
    print()
    
    print("┌─────────────────────┬────────────┬────────────┬────────────┐")
    print("│ Method              │ Decentralized│ Private  │ Easy Setup │")
    print("├─────────────────────┼────────────┼────────────┼────────────┤")
    print("│ DHT                 │    ✅       │    ❌     │    Medium  │")
    print("│ Bootstrap           │    ❌       │    ✅     │    Easy    │")
    print("│ Direct              │    ❌       │    ✅     │    Manual  │")
    print("│ Local               │    ✅       │    ✅     │    Easy    │")
    print("└─────────────────────┴────────────┴────────────┴────────────┘")
    print()
    
    print("Recommended combinations:")
    print("  • Public network: DHT + Local")
    print("  • Private network: Direct + Local")
    print("  • Controlled network: Bootstrap + Direct")
    print("  • Office network: Local only")
    print()


if __name__ == "__main__":
    main()