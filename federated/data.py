#!/usr/bin/env python3
"""
Training Data Distribution - Where Does Data Come From?

This answers: "Where does training data come from and how is it distributed?"

BITCOIN'S APPROACH:
──────────────────────────────────────────────────────────────────
Transactions:
- Users submit transactions to network
- Transactions go to mempool (waiting area)
- Miners select transactions from mempool
- Include in block template
- Other nodes validate

Our Approach:
- Training data can come from multiple sources
- Data distribution happens at template creation
- Each approach has tradeoffs

OPTIONS:
──────────────────────────────────────────────────────────────────
1. LOCAL DATA (Federated Learning)
   - Each node has its own data
   - Never share raw data
   - Maximum privacy
   - Data stays on node

2. HOST-DISTRIBUTED DATA
   - Host provides data references
   - Nodes download what they need
   - Good for public datasets
   - Centralized data source

3. PEER-TO-PEER DATA
   - Nodes share data with each other
   - Like BitTorrent for data
   - Decentralized
   - More complex

4. DATA MARKETPLACE
   - Data providers list datasets
   - Nodes subscribe to datasets
   - Economic incentives
   - Most complex

IMPLEMENTATION:
"""

import os
import sys
import time
import hashlib
import secrets
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

os.environ['PYTHONWARNINGS'] = 'ignore::urllib3.warnings.NotOpenSSLWarning'


# ============================================================================
# Training Data Structures
# ============================================================================

@dataclass
class DataBatch:
    """A batch of training data (like Bitcoin transaction)."""
    batch_id: str
    data_hash: str  # Hash of data
    size: int  # Number of samples
    source: str  # Where data came from
    location: str  # How to access (local path, URL, IPFS, etc.)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "batch_id": self.batch_id,
            "data_hash": self.data_hash,
            "size": self.size,
            "source": self.source,
            "location": self.location,
            "metadata": self.metadata,
        }


@dataclass
class DataReference:
    """Reference to training data (like SPV proof)."""
    batch_id: str
    data_hash: str
    verifier: str  # Who verified this data
    signature: str  # Cryptographic proof
    location: str  # Where to get data


# ============================================================================
# Option 1: Local Data (Federated Learning)
# ============================================================================

class LocalDataNode:
    """
    Each node has its own local data.
    
    Like Bitcoin full node:
    - Has all transactions locally
    - No need to request from network
    - Maximum privacy
    
    Federated Learning approach:
    - Data never leaves node
    - Only gradients shared
    - Best for private data
    """
    
    def __init__(self, node_id: str, local_data_path: str = None):
        self.node_id = node_id
        self.local_data_path = local_data_path
        self.data_batches: Dict[str, DataBatch] = {}
        self.logger = logging.getLogger(f"local-node-{node_id}")
        
        # Load local data
        if local_data_path:
            self._load_local_data()
    
    def _load_local_data(self):
        """Load local training data."""
        # In production, would load actual training data
        # For demo, create simulated batches
        
        for i in range(10):
            batch_id = f"batch-{i}"
            self.data_batches[batch_id] = DataBatch(
                batch_id=batch_id,
                data_hash=hashlib.sha256(f"data-{i}".encode()).hexdigest(),
                size=1000,  # 1000 samples per batch
                source="local",
                location=f"{self.local_data_path}/batch-{i}.json",
            )
        
        self.logger.info(f"Loaded {len(self.data_batches)} local data batches")
    
    def get_batch(self, batch_id: str) -> Optional[DataBatch]:
        """Get data batch by ID."""
        return self.data_batches.get(batch_id)
    
    def get_available_batches(self) -> List[str]:
        """Get list of available batch IDs."""
        return list(self.data_batches.keys())
    
    def get_batch_count(self) -> int:
        """Get number of available batches."""
        return len(self.data_batches)


# ============================================================================
# Option 2: Host-Distributed Data
# ============================================================================

class DataHost:
    """
    Host distributes training data references.
    
    Like Bitcoin mining pool:
    - Pool provides block template with transactions
    - Miners get template with transaction references
    - Download/verify transactions
    
    Data Host:
    - Provides data references in template
    - Nodes download data they need
    - Good for public datasets
    """
    
    def __init__(self, host_id: str):
        self.host_id = host_id
        self.data_pool: Dict[str, DataBatch] = {}  # Like mempool
        self.data_providers: Dict[str, Dict] = {}  # Registered data sources
        self.logger = logging.getLogger(f"data-host-{host_id}")
        
        # Initialize with some data
        self._initialize_data_pool()
    
    def _initialize_data_pool(self):
        """Initialize data pool with available data."""
        # In production, would load real datasets
        # For demo, create simulated data batches
        
        for i in range(20):
            batch_id = f"data-{i}"
            self.data_pool[batch_id] = DataBatch(
                batch_id=batch_id,
                data_hash=hashlib.sha256(f"training-data-{i}".encode()).hexdigest(),
                size=1000,
                source="host",
                location=f"https://data.example.com/batch-{i}.json",
                metadata={
                    "format": "json",
                    "compressed": True,
                    "checksum": hashlib.md5(f"data-{i}".encode()).hexdigest(),
                }
            )
        
        self.logger.info(f"Initialized data pool with {len(self.data_pool)} batches")
    
    def register_data_provider(self, provider_id: str, data_url: str, metadata: Dict):
        """Register a data provider (like adding transaction to mempool)."""
        self.data_providers[provider_id] = {
            "url": data_url,
            "metadata": metadata,
            "registered_at": time.time(),
        }
        
        self.logger.info(f"Registered data provider: {provider_id}")
    
    def get_data_for_template(self, difficulty: int = 100) -> List[DataBatch]:
        """
        Get data batches for template.
        
        Like Bitcoin's getblocktemplate:
        - Selects transactions from mempool
        - Returns batch of transactions
        - Miner includes in block
        """
        # Select data batches based on difficulty
        # More difficulty = more data
        num_batches = max(1, difficulty // 10)
        
        # Get available batches
        available = list(self.data_pool.values())
        
        # Select random subset
        import random
        selected = random.sample(available, min(num_batches, len(available)))
        
        return selected
    
    def get_data_batch(self, batch_id: str) -> Optional[DataBatch]:
        """Get specific data batch."""
        return self.data_pool.get(batch_id)
    
    def list_available_data(self) -> List[Dict]:
        """List all available data batches."""
        return [batch.to_dict() for batch in self.data_pool.values()]


# ============================================================================
# Option 3: Peer-to-Peer Data Distribution
# ============================================================================

class P2PDataNetwork:
    """
    Peer-to-peer data distribution.
    
    Like BitTorrent:
    - Data split into pieces
    - Nodes share pieces with each other
    - Rare pieces prioritized
    - No central server
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.local_data: Dict[str, DataBatch] = {}
        self.peers: Dict[str, Dict] = {}  # peer_id -> info
        self.data_requests: Dict[str, List] = {}  # batch_id -> requesters
        self.logger = logging.getLogger(f"p2p-{node_id}")
    
    def add_peer(self, peer_id: str, peer_info: Dict):
        """Add peer to network."""
        self.peers[peer_id] = peer_info
        self.logger.info(f"Added peer: {peer_id}")
    
    def request_data(self, batch_id: str, peer_id: str) -> Optional[DataBatch]:
        """Request data from peer."""
        if peer_id not in self.peers:
            self.logger.warning(f"Unknown peer: {peer_id}")
            return None
        
        # In production, would make network request
        # For demo, return simulated data
        
        self.data_requests[batch_id] = self.data_requests.get(batch_id, [])
        self.data_requests[batch_id].append(peer_id)
        
        return DataBatch(
            batch_id=batch_id,
            data_hash=hashlib.sha256(batch_id.encode()).hexdigest(),
            size=1000,
            source="p2p",
            location=f"p2p://{peer_id}/{batch_id}",
        )
    
    def share_data(self, batch_id: str, data: DataBatch):
        """Share data with network."""
        self.local_data[batch_id] = data
        self.logger.info(f"Sharing data: {batch_id}")
    
    def get_data_from_network(self, batch_id: str) -> Optional[DataBatch]:
        """Get data from any available peer."""
        for peer_id, peer_info in self.peers.items():
            if batch_id in peer_info.get("has_data", []):
                return self.request_data(batch_id, peer_id)
        
        return None


# ============================================================================
# Data Distribution in Template
# ============================================================================

class DataDistributor:
    """
    Distributes training data through templates.
    
    Combines all approaches:
    - Local data (privacy)
    - Host data (convenience)
    - P2P data (resilience)
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            "mode": "local",  # local, host, p2p, hybrid
            "cache_size": 100,
            "max_downloads": 5,
        }
        
        self.local_node: Optional[LocalDataNode] = None
        self.data_host: Optional[DataHost] = None
        self.p2p_network: Optional[P2PDataNetwork] = None
        
        self.logger = logging.getLogger("data-distributor")
    
    def setup_local(self, node_id: str, data_path: str):
        """Setup local data mode."""
        self.local_node = LocalDataNode(node_id, data_path)
        self.logger.info(f"Setup local data: {len(self.local_node.data_batches)} batches")
    
    def setup_host(self, host_id: str):
        """Setup host data mode."""
        self.data_host = DataHost(host_id)
        self.logger.info(f"Setup host data: {len(self.data_host.data_pool)} batches")
    
    def setup_p2p(self, node_id: str, peers: List[str]):
        """Setup P2P data mode."""
        self.p2p_network = P2PDataNetwork(node_id)
        for peer_id in peers:
            self.p2p_network.add_peer(peer_id, {})
        self.logger.info(f"Setup P2P data: {len(peers)} peers")
    
    def get_training_data(self, difficulty: int = 100) -> List[DataBatch]:
        """
        Get training data for template.
        
        Like Bitcoin's getblocktemplate:
        - Returns list of transactions (data batches)
        - Each batch has reference to data
        - Node downloads what it needs
        """
        batches = []
        
        # Try local first
        if self.local_node:
            available = self.local_node.get_available_batches()
            num_local = min(difficulty // 10, len(available))
            
            import random
            selected = random.sample(available, num_local)
            
            for batch_id in selected:
                batch = self.local_node.get_batch(batch_id)
                if batch:
                    batches.append(batch)
        
        # If need more, try host
        if self.data_host and len(batches) < difficulty // 10:
            needed = (difficulty // 10) - len(batches)
            host_batches = self.data_host.get_data_for_template(needed * 10)
            batches.extend(host_batches[:needed])
        
        # If need more, try P2P
        if self.p2p_network and len(batches) < difficulty // 10:
            # In production, would request from network
            pass
        
        return batches


# ============================================================================
# Demo
# ============================================================================

def main():
    """Demo data distribution."""
    print("="*70)
    print("TRAINING DATA DISTRIBUTION")
    print("="*70)
    print()
    
    print("QUESTION: Where does training data come from?")
    print("-"*70)
    print()
    
    print("ANSWER: Multiple sources, like Bitcoin transactions:")
    print()
    
    print("="*70)
    print("OPTION 1: LOCAL DATA (Federated Learning)")
    print("="*70)
    print()
    
    print("Each node has its own data:")
    print()
    print("  Node A: [batch-0, batch-1, batch-2]  (private data)")
    print("  Node B: [batch-0, batch-1, batch-3]  (private data)")
    print("  Node C: [batch-0, batch-2, batch-3]  (private data)")
    print()
    print("Benefits:")
    print("  ✅ Maximum privacy (data never leaves node)")
    print("  ✅ Works for sensitive data (healthcare, finance)")
    print("  ✅ No network overhead for data transfer")
    print("  ✅ Each node trains on unique data")
    print()
    print("Trade-offs:")
    print("  ⚠️ Requires each node to have enough data")
    print("  ⚠️ Data heterogeneity (different distributions)")
    print("  ⚠️ No data sharing")
    print()
    
    # Demo local data
    print("Demo:")
    local_node = LocalDataNode("node-1", "/data/local")
    print(f"  Local batches: {local_node.get_batch_count()}")
    print(f"  Available: {local_node.get_available_batches()[:3]}...")
    print()
    
    print("="*70)
    print("OPTION 2: HOST-DISTRIBUTED DATA")
    print("="*70)
    print()
    
    print("Host provides data references:")
    print()
    print("  Host:                                            ")
    print("  ├── data-0.json (https://data.example.com/batch-0.json)")
    print("  ├── data-1.json (https://data.example.com/batch-1.json)")
    print("  └── ...")
    print()
    print("  Template includes:")
    print("  ├── batch_ids: [data-0, data-1, data-2]")
    print("  └── locations: [https://..., https://..., https://...]")
    print()
    print("Benefits:")
    print("  ✅ Centralized data source")
    print("  ✅ All nodes train on same data")
    print("  ✅ Easy for public datasets")
    print()
    print("Trade-offs:")
    print("  ⚠️ Single point of failure")
    print("  ⚠️ Network overhead for data download")
    print("  ⚠️ Privacy concerns (data leaves host)")
    print()
    
    # Demo host data
    print("Demo:")
    host = DataHost("host-1")
    print(f"  Data pool: {len(host.data_pool)} batches")
    batches = host.get_data_for_template(100)
    print(f"  Template batches: {len(batches)}")
    print(f"  Example: {batches[0].batch_id}")
    print()
    
    print("="*70)
    print("OPTION 3: PEER-TO-PEER DATA")
    print("="*70)
    print()
    
    print("Nodes share data with each other:")
    print()
    print("  ┌─────┐     ┌─────┐     ┌─────┐")
    print("  │Node │◄───►│Node │◄───►│Node │")
    print("  │  A  │     │  B  │     │  C  │")
    print("  └──┬──┘     └──┬──┘     └──┬──┘")
    print("     │    ╲      │      ╱    │")
    print("     │     ╲     │     ╱     │")
    print("     ▼      ╲    ▼    ╱      ▼")
    print("  ┌─────┐   ╲ ┌─────┐ ╱   ┌─────┐")
    print("  │Node │◄───►│Data │◄───►│Node │")
    print("  │  D  │     │Peer │     │  E  │")
    print("  └─────┘     └─────┘     └─────┘")
    print()
    print("Benefits:")
    print("  ✅ No central server")
    print("  ✅ Resilient (data replicated)")
    print("  ✅ Efficient (download from nearest)")
    print()
    print("Trade-offs:")
    print("  ⚠️ Complex implementation")
    print("  ⚠️ Data consistency issues")
    print("  ⚠️ Sync overhead")
    print()
    
    # Demo P2P
    print("Demo:")
    p2p = P2PDataNetwork("node-1")
    p2p.add_peer("node-2", {"has_data": ["data-0", "data-1"]})
    p2p.add_peer("node-3", {"has_data": ["data-2", "data-3"]})
    print(f"  Peers: {len(p2p.peers)}")
    print(f"  Can request: data-0, data-1, data-2, data-3")
    print()
    
    print("="*70)
    print("RECOMMENDED APPROACH")
    print("="*70)
    print()
    
    print("HYBRID: Combine all three!")
    print()
    print("  1. Each node has LOCAL private data")
    print("  2. Host provides PUBLIC data references")
    print("  3. Nodes share via P2P for redundancy")
    print()
    
    print("  Template includes:")
    print("  ├── local_batches: [batch-0, batch-1]  # From local data")
    print("  ├── host_batches: [data-2, data-3]      # Download from host")
    print("  └── p2p_batches: [data-4, data-5]      # Get from peers")
    print()
    
    print("  This gives:")
    print("  ✅ Privacy for sensitive data")
    print("  ✅ Convenience for public data")
    print("  ✅ Resilience from P2P")
    print()
    
    print("="*70)
    print("BITCOIN ANALOGY")
    print("="*70)
    print()
    
    print("BITCOIN:")
    print("  Transactions come from mempool")
    print("  Pool selects transactions for block")
    print("  Nodes can validate locally")
    print("  SPV nodes just verify headers")
    print()
    
    print("TRAINING:")
    print("  Data comes from local/host/P2P")
    print("  Host selects data for template")
    print("  Nodes can train locally")
    print("  Light nodes just verify gradients")
    print()
    
    print("✅ Same concept, different application!")


if __name__ == "__main__":
    main()