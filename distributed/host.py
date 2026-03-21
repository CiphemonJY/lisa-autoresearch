#!/usr/bin/env python3
"""
Model State Chain - Who Creates Templates?

This answers: "Where do templates come from?"

BITCOIN'S APPROACH:
──────────────────────────────────────────────────────────────────
Mining Pools:
- Pool creates block template
- Includes transactions from mempool
- Sets difficulty target
- Distributes to miners
- Miners don't need to know about transactions

Solo Mining:
- Miner creates their own template
- Gathers transactions from network
- Sets own parameters
- Mines independently

OUR APPROACH:
──────────────────────────────────────────────────────────────────
Model Host (Coordinator):
- Creates initial model state
- Distributes training tasks
- Aggregates completed blocks
- Updates model state

Distributed Coordinator:
- Multiple hosts (like mining pools)
- Nodes can choose which host to follow
- No single point of failure

State Chain:
- Each block references previous model state
- Anyone can create template from current state
- Like Bitcoin's longest chain rule

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

os.environ['PYTHONWARNMENTS'] = 'ignore::urllib3.warnings.NotOpenSSLWarning'


# ============================================================================
# Model State (Like Bitcoin Block)
# ============================================================================

@dataclass
class ModelState:
    """
    Model state is like a Bitcoin block.
    
    Bitcoin Block:
    - Previous block hash (chain linkage)
    - Transactions (data to process)
    - Merkle root (data integrity)
    - Timestamp
    - Difficulty target
    - Nonce (proof of work)
    
    Model State:
    - Previous state hash (chain linkage)
    - Training data (data to process)
    - Gradient hash (data integrity)
    - Timestamp
    - Difficulty (iterations to train)
    - Proof of training
    """
    state_id: str
    previous_state_hash: str
    model_version: str
    training_data: Any  # Reference to training data
    gradient_hash: str  # Hash of completed gradients
    timestamp: float
    difficulty: int
    proof_of_training: str  # Hash proving training was done
    creator_id: str  # Who created this state
    nonce: int = 0  # For state hash
    
    def compute_hash(self) -> str:
        """Compute state hash (like Bitcoin block hash)."""
        content = (
            f"{self.state_id}:{self.previous_state_hash}:"
            f"{self.model_version}:{self.gradient_hash}:"
            f"{self.timestamp}:{self.difficulty}:{self.nonce}"
        )
        return hashlib.sha256(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        return {
            "state_id": self.state_id,
            "previous_state_hash": self.previous_state_hash,
            "model_version": self.model_version,
            "gradient_hash": self.gradient_hash,
            "timestamp": self.timestamp,
            "difficulty": self.difficulty,
            "creator_id": self.creator_id,
            "nonce": self.nonce,
        }


# ============================================================================
# Model Host (Like Mining Pool)
# ============================================================================

class ModelHost:
    """
    Model host creates templates from model state.
    
    Like Bitcoin Mining Pool:
    - Maintains current model state
    - Creates templates for miners
    - Distributes work fairly
    - Aggregates completed blocks
    - Updates model state
    
    This is ONE approach. Alternatives:
    - Multiple hosts (like multiple pools)
    - Peer-to-peer state sharing
    - Consensus on model state
    """
    
    def __init__(self, host_id: str, config: Dict = None):
        self.host_id = host_id
        self.config = config or {}
        self.current_state: Optional[ModelState] = None
        self.state_chain: List[ModelState] = []
        self.template_counter = 0
        self.participants: Dict[str, Dict] = {}  # node_id -> stats
        self.logger = logging.getLogger(f"host-{host_id}")
        
        # Initialize genesis state
        self._create_genesis_state()
    
    def _create_genesis_state(self):
        """Create genesis model state (like genesis block)."""
        genesis = ModelState(
            state_id="genesis",
            previous_state_hash="0" * 64,
            model_version="0.0.1",
            training_data=None,  # Initial state has no training
            gradient_hash="genesis",
            timestamp=time.time(),
            difficulty=0,
            proof_of_training="genesis",
            creator_id=self.host_id,
        )
        
        self.current_state = genesis
        self.state_chain.append(genesis)
        
        self.logger.info(f"Created genesis state: {genesis.state_id}")
    
    def get_current_state(self) -> ModelState:
        """Get current model state."""
        return self.current_state
    
    def create_template(self, node_id: str) -> Dict:
        """
        Create template for training node.
        
        Like Bitcoin's getblocktemplate:
        - Takes current state
        - Creates template for node to work on
        - Assigns to node
        - Returns immediately
        """
        if not self.current_state:
            raise ValueError("No current state")
        
        self.template_counter += 1
        
        # Create template from current state
        template = {
            "template_id": f"template-{self.template_counter}",
            "previous_state_hash": self.current_state.compute_hash(),
            "model_version": self.current_state.model_version,
            "difficulty": 100,  # Training iterations
            "created_at": time.time(),
            "assigned_to": node_id,
            "host_id": self.host_id,
        }
        
        # Track participant
        if node_id not in self.participants:
            self.participants[node_id] = {
                "templates_requested": 0,
                "templates_completed": 0,
                "reputation": 50.0,
            }
        
        self.participants[node_id]["templates_requested"] += 1
        
        self.logger.info(f"Created template {template['template_id']} for {node_id}")
        
        return template
    
    def submit_block(self, template_id: str, gradient_block: Dict, node_id: str) -> bool:
        """
        Submit completed gradient block.
        
        Like Bitcoin's submitblock:
        - Validates block
        - Updates model state
        - Distributes new state
        """
        if not self._validate_block(gradient_block, node_id):
            self.logger.warning(f"Invalid block from {node_id}")
            return False
        
        # Create new model state
        new_state = ModelState(
            state_id=f"state-{len(self.state_chain)}",
            previous_state_hash=self.current_state.compute_hash(),
            model_version=self.current_state.model_version,
            training_data=gradient_block.get("training_data"),
            gradient_hash=hashlib.sha256(
                str(gradient_block.get("gradient", "")).encode()
            ).hexdigest()[:16],
            timestamp=time.time(),
            difficulty=gradient_block.get("difficulty", 100),
            proof_of_training=gradient_block.get("proof", ""),
            creator_id=node_id,
        )
        
        # Update chain
        self.state_chain.append(new_state)
        self.current_state = new_state
        
        # Update participant
        if node_id in self.participants:
            self.participants[node_id]["templates_completed"] += 1
            self.participants[node_id]["reputation"] += 1.0
        
        self.logger.info(f"Accepted block from {node_id}, new state: {new_state.state_id}")
        
        return True
    
    def _validate_block(self, block: Dict, node_id: str) -> bool:
        """Validate gradient block."""
        # In production, would validate:
        # - Proof of training
        # - Gradient correctness
        # - Difficulty met
        # - Reputation sufficient
        
        return True
    
    def get_state_chain(self, limit: int = 10) -> List[Dict]:
        """Get recent model states."""
        return [s.to_dict() for s in self.state_chain[-limit:]]
    
    def get_participants(self) -> Dict:
        """Get participant statistics."""
        return self.participants


# ============================================================================
# Distributed Coordinator (Alternative to Single Host)
# ============================================================================

class DistributedCoordinator:
    """
    Alternative: Multiple hosts, no single point of failure.
    
    Like Multiple Mining Pools:
    - Nodes can choose which host to follow
    - Hosts compete to provide best templates
    - If one host fails, nodes switch to another
    
    State Consensus:
    - Longest valid chain wins (like Bitcoin)
    - Nodes can submit to multiple hosts
    - Hosts can sync with each other
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.hosts: Dict[str, ModelHost] = {}  # host_id -> host
        self.preferred_host: Optional[str] = None
        self.logger = logging.getLogger(f"coordinator-{node_id}")
    
    def register_host(self, host: ModelHost):
        """Register a host."""
        self.hosts[host.host_id] = host
        self.logger.info(f"Registered host: {host.host_id}")
        
        # Use first host as preferred
        if not self.preferred_host:
            self.preferred_host = host.host_id
    
    def get_template(self, host_id: str = None) -> Dict:
        """Get template from preferred host."""
        host_id = host_id or self.preferred_host
        
        if host_id not in self.hosts:
            raise ValueError(f"Unknown host: {host_id}")
        
        return self.hosts[host_id].create_template(self.node_id)
    
    def submit_block(self, template_id: str, gradient_block: Dict, host_id: str = None) -> bool:
        """Submit block to preferred host."""
        host_id = host_id or self.preferred_host
        
        if host_id not in self.hosts:
            raise ValueError(f"Unknown host: {host_id}")
        
        return self.hosts[host_id].submit_block(template_id, gradient_block, self.node_id)
    
    def get_longest_chain(self) -> List[ModelState]:
        """Get longest valid chain (like Bitcoin)."""
        longest_host = None
        longest_length = 0
        
        for host_id, host in self.hosts.items():
            if len(host.state_chain) > longest_length:
                longest_length = len(host.state_chain)
                longest_host = host_id
        
        if longest_host:
            return self.hosts[longest_host].state_chain
        
        return []


# ============================================================================
# Training Node (Like Bitcoin Miner)
# ============================================================================

class TrainingNode:
    """
    Training node that works on templates.
    
    Like Bitcoin Miner:
    - Requests template from host
    - Works on template (trains)
    - Submits result
    - Gets next template
    - Never waits
    """
    
    def __init__(self, node_id: str, host: ModelHost):
        self.node_id = node_id
        self.host = host
        self.current_template: Optional[Dict] = None
        self.stats = {
            "templates_received": 0,
            "blocks_submitted": 0,
            "reputation": 50.0,
        }
        self.logger = logging.getLogger(f"node-{node_id}")
    
    def request_work(self) -> Dict:
        """
        Request work from host.
        
        Like Bitcoin's getblocktemplate:
        - Returns immediately
        - Always has work available
        - No waiting
        """
        template = self.host.create_template(self.node_id)
        self.current_template = template
        self.stats["templates_received"] += 1
        
        self.logger.info(f"Received template {template['template_id']}")
        
        return template
    
    def complete_work(self, gradient_block: Dict) -> bool:
        """
        Complete work and submit to host.
        
        Like Bitcoin's submitblock:
        - Validates work
        - Submits to host
        - Gets new template immediately
        """
        if not self.current_template:
            self.logger.error("No current template")
            return False
        
        # Submit to host
        success = self.host.submit_block(
            self.current_template["template_id"],
            gradient_block,
            self.node_id,
        )
        
        if success:
            self.stats["blocks_submitted"] += 1
            self.logger.info(f"Submitted block, total: {self.stats['blocks_submitted']}")
        
        # Clear template (will get new one immediately)
        self.current_template = None
        
        return success
    
    def mine_continuously(self):
        """
        Continuous mining loop.
        
        Like Bitcoin miner:
        - Request work
        - Do work
        - Submit
        - Repeat
        - Never wait
        """
        while True:
            # Get work (immediate)
            template = self.request_work()
            
            # Do work (train)
            gradient_block = self._do_work(template)
            
            # Submit (immediate)
            self.complete_work(gradient_block)
            
            # Get next work (immediate, loop continues)
    
    def _do_work(self, template: Dict) -> Dict:
        """
        Do training work on template.
        
        In production, would:
        - Load training data
        - Train for difficulty iterations
        - Compute gradient
        - Return gradient block
        """
        # Simulate training
        iterations = template.get("difficulty", 100)
        
        self.logger.debug(f"Training for {iterations} iterations...")
        
        # Return simulated gradient block
        return {
            "template_id": template["template_id"],
            "gradient": "simulated_gradient",
            "difficulty": iterations,
            "proof": f"proof-{secrets.token_urlsafe(8)}",
        }


# ============================================================================
# Demo
# ============================================================================

def main():
    """Demo model host and template creation."""
    print("="*70)
    print("MODEL HOST - WHO CREATES TEMPLATES?")
    print("="*70)
    print()
    
    print("QUESTION: Where do templates come from?")
    print("-"*70)
    print()
    
    print("ANSWER: Multiple options, like Bitcoin:")
    print()
    
    print("OPTION 1: SINGLE HOST (Like Mining Pool)")
    print("-"*70)
    print("- Host creates templates from model state")
    print("- Distributes to nodes")
    print("- Aggregates completed blocks")
    print("- Updates model state")
    print("- Simple, but centralized")
    print()
    
    print("OPTION 2: MULTIPLE HOSTS (Like Multiple Pools)")
    print("-"*70)
    print("- Multiple hosts compete")
    print("- Nodes choose which host to use")
    print("- No single point of failure")
    print("- More decentralized")
    print()
    
    print("OPTION 3: PEER-TO-PEER (Like Bitcoin Network)")
    print("-"*70)
    print("- Nodes share model state directly")
    print("- Longest valid chain wins")
    print("- Fully decentralized")
    print("- More complex")
    print()
    
    print("="*70)
    print("DEMO: SINGLE HOST")
    print("="*70)
    print()
    
    # Create host
    host = ModelHost("host-1")
    
    print("1. Host creates genesis state:")
    genesis = host.get_current_state()
    print(f"   State ID: {genesis.state_id}")
    print(f"   Model version: {genesis.model_version}")
    print()
    
    print("2. Node requests template:")
    node = TrainingNode("node-1", host)
    template = node.request_work()
    print(f"   Template ID: {template['template_id']}")
    print(f"   Previous state: {template['previous_state_hash'][:16]}...")
    print(f"   Difficulty: {template['difficulty']} iterations")
    print()
    
    print("3. Node trains and submits block:")
    gradient_block = node._do_work(template)
    success = node.complete_work(gradient_block)
    print(f"   Submit result: {'✅ Accepted' if success else '❌ Rejected'}")
    print()
    
    print("4. Host updates model state:")
    new_state = host.get_current_state()
    print(f"   New state ID: {new_state.state_id}")
    print(f"   Previous: {new_state.previous_state_hash[:16]}...")
    print(f"   Created by: {new_state.creator_id}")
    print()
    
    print("5. Node immediately gets next template:")
    template2 = node.request_work()
    print(f"   Template ID: {template2['template_id']}")
    print(f"   Previous state: {template2['previous_state_hash'][:16]}...")
    print()
    
    print("✅ Continuous work, never waiting!")
    print()
    
    print("="*70)
    print("HOW IT WORKS")
    print("="*70)
    print()
    
    print("1. HOST MAINTAINS MODEL STATE:")
    print("   ├── Current model state")
    print("   ├── State chain (like blockchain)")
    print("   └── Template pool")
    print()
    
    print("2. NODE REQUESTS WORK:")
    print("   ├── Calls host.create_template()")
    print("   ├── Gets template immediately")
    print("   └── No waiting")
    print()
    
    print("3. NODE TRAINS:")
    print("   ├── Uses template's training data")
    print("   ├── Trains for difficulty iterations")
    print("   └── Computes gradient")
    print()
    
    print("4. NODE SUBMITS:")
    print("   ├── Calls host.submit_block()")
    print("   ├── Host validates and accepts")
    print("   └── Host updates model state")
    print()
    
    print("5. NODE GETS NEXT WORK:")
    print("   ├── Immediately requests new template")
    print("   └── Never waits!")
    print()
    
    print("="*70)
    print("BITCOIN ANALOGY")
    print("="*70)
    print()
    
    print("BITCOIN:")
    print("  Pool creates block template")
    print("  Miner works on template")
    print("  Miner submits completed block")
    print("  Pool updates blockchain")
    print("  Miner gets next template")
    print("  → Continuous mining")
    print()
    
    print("TRAINING:")
    print("  Host creates model template")
    print("  Node trains on template")
    print("  Node submits gradient block")
    print("  Host updates model state")
    print("  Node gets next template")
    print("  → Continuous training")
    print()
    
    print("✅ Same concept, different application!")


if __name__ == "__main__":
    main()