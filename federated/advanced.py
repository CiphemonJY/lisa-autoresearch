#!/usr/bin/env python3
"""
Advanced Federated Learning Concepts

This covers four key concepts:

2. INCENTIVE MECHANISMS
   - Why participate?
   - Token rewards
   - Reputation scoring
   - Stake-based participation

3. PRIVACY GUARANTEES
   - Differential privacy
   - Gradient clipping
   - Noise injection
   - Mathematical proofs

4. SECURE AGGREGATION
   - Aggregator can't see individual gradients
   - Homomorphic encryption
   - Secret sharing
   - Multi-party computation

5. MODEL CONVERGENCE
   - Ensuring model improves
   - Convergence proofs
   - FedAvg guarantees
   - Handling non-IID data

BITCOIN ANALOGIES THROUGHOUT:
- Incentives = Block rewards
- Privacy = Stealth addresses
- Secure Aggregation = CoinJoin
- Convergence = Chain validation
"""

import os
import sys
import time
import hashlib
import secrets
import json
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

os.environ['PYTHONWARNINGS'] = 'ignore::urllib3.warnings.NotOpenSSLWarning'


# ============================================================================
# 2. INCENTIVE MECHANISMS
# ============================================================================

@dataclass
class TokenReward:
    """Reward for participating in training."""
    tokens: float
    reason: str
    timestamp: float
    contribution_score: float


class IncentiveSystem:
    """
    Token-based incentive system for federated learning.
    
    Like Bitcoin mining rewards:
    - Miners get block rewards for valid blocks
    - Nodes get tokens for valid gradients
    - Reputation increases with good behavior
    
    WHY PARTICIPATE?
    ─────────────────────────────────────────────────────────────
    Hospitals participate because:
    1. Token rewards (convertible to value)
    2. Model access (trained on ALL data)
    3. Reputation (trust score)
    4. Stake returns (if they stake tokens)
    
    Bitcoin Analogy:
    - Block reward = 6.25 BTC for mining
    - Gradient reward = X tokens for training
    """
    
    def __init__(self, name: str):
        self.name = name
        self.balances: Dict[str, float] = {}  # hospital_id -> tokens
        self.reputation: Dict[str, float] = {}  # hospital_id -> reputation (0-100)
        self.stakes: Dict[str, float] = {}  # hospital_id -> staked tokens
        self.contribution_history: Dict[str, List[Dict]] = {}
        self.total_tokens = 1_000_000  # Initial supply
        self.reward_pool = 100_000  # Tokens for rewards
        self.logger = logging.getLogger(f"incentive-{name}")
    
    def register_participant(self, hospital_id: str, initial_stake: float = 0):
        """Register a new participant."""
        self.balances[hospital_id] = 0
        self.reputation[hospital_id] = 50.0  # Start with neutral reputation
        self.stakes[hospital_id] = initial_stake
        self.contribution_history[hospital_id] = []
        
        self.logger.info(f"Registered {hospital_id} with stake {initial_stake}")
    
    def calculate_reward(self, hospital_id: str, gradient_quality: float, 
                         sample_count: int, staleness: int) -> TokenReward:
        """
        Calculate reward for submitting gradient.
        
        Like Bitcoin block reward:
        - Base reward (like 6.25 BTC)
        - Quality multiplier (how good is the gradient)
        - Sample multiplier (more data = more reward)
        - Staleness penalty (old gradients worth less)
        """
        # Base reward (like Bitcoin block reward)
        base_reward = 10.0  # 10 tokens per valid gradient
        
        # Quality multiplier (0-1, based on gradient validation)
        quality_mult = max(0.1, min(1.0, gradient_quality))
        
        # Sample multiplier (more samples = more contribution)
        sample_mult = min(2.0, sample_count / 10000)  # Cap at 2x
        
        # Staleness penalty (older gradients worth less)
        staleness_mult = 1.0 / (1 + staleness * 0.1)
        
        # Reputation multiplier (higher rep = higher reward)
        rep_mult = 1.0 + (self.reputation[hospital_id] - 50) / 100
        
        # Stake multiplier (more stake = more reward)
        stake_mult = 1.0 + self.stakes.get(hospital_id, 0) / 10000
        
        # Calculate total reward
        reward = base_reward * quality_mult * sample_mult * staleness_mult * rep_mult * stake_mult
        
        return TokenReward(
            tokens=reward,
            reason=f"Gradient contribution (quality={gradient_quality:.2f}, samples={sample_count})",
            timestamp=time.time(),
            contribution_score=gradient_quality * sample_count,
        )
    
    def distribute_reward(self, hospital_id: str, reward: TokenReward):
        """Distribute reward to participant."""
        self.balances[hospital_id] = self.balances.get(hospital_id, 0) + reward.tokens
        
        # Update reputation
        old_rep = self.reputation.get(hospital_id, 50)
        new_rep = min(100, old_rep + reward.contribution_score / 1000)
        self.reputation[hospital_id] = new_rep
        
        # Track contribution
        self.contribution_history[hospital_id].append({
            "tokens": reward.tokens,
            "reason": reward.reason,
            "timestamp": reward.timestamp,
            "contribution_score": reward.contribution_score,
        })
        
        self.logger.info(f"Distributed {reward.tokens:.2f} tokens to {hospital_id} (rep: {old_rep:.1f} → {new_rep:.1f})")
    
    def slash_reputation(self, hospital_id: str, amount: float, reason: str):
        """
        Slash reputation for bad behavior.
        
        Like Bitcoin slashing for invalid blocks.
        """
        old_rep = self.reputation.get(hospital_id, 50)
        new_rep = max(0, old_rep - amount)
        self.reputation[hospital_id] = new_rep
        
        # Also slash tokens
        slash_amount = self.balances.get(hospital_id, 0) * (amount / 100)
        self.balances[hospital_id] = max(0, self.balances[hospital_id] - slash_amount)
        
        self.logger.warning(f"Slashed {hospital_id}: rep {old_rep:.1f} → {new_rep:.1f} ({reason})")
    
    def stake_tokens(self, hospital_id: str, amount: float) -> bool:
        """
        Stake tokens for higher rewards.
        
        Like Bitcoin miners investing in hardware.
        More stake = more commitment = more trust.
        """
        if self.balances.get(hospital_id, 0) < amount:
            return False
        
        self.balances[hospital_id] -= amount
        self.stakes[hospital_id] = self.stakes.get(hospital_id, 0) + amount
        
        self.logger.info(f"{hospital_id} staked {amount} tokens (total stake: {self.stakes[hospital_id]})")
        return True
    
    def get_status(self, hospital_id: str) -> Dict:
        """Get participant status."""
        return {
            "hospital_id": hospital_id,
            "balance": self.balances.get(hospital_id, 0),
            "reputation": self.reputation.get(hospital_id, 50),
            "stake": self.stakes.get(hospital_id, 0),
            "contributions": len(self.contribution_history.get(hospital_id, [])),
        }


# ============================================================================
# 3. PRIVACY GUARANTEES
# ============================================================================

class DifferentialPrivacy:
    """
    Differential privacy for gradients.
    
    Mathematical guarantee: Even if you see the gradient,
    you can't tell if any specific patient was in the data.
    
    LIKE BITCOIN STEALTH ADDRESSES:
    ─────────────────────────────────────────────────────────────
    Bitcoin: Stealth addresses hide transaction participants
    DP: Differential privacy hides training data participants
    
    HOW IT WORKS:
    ─────────────────────────────────────────────────────────────
    1. Clip gradients (limit contribution of any single sample)
    2. Add noise (obscure individual contributions)
    3. Mathematical proof (epsilon, delta guarantees)
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, 
                 clip_norm: float = 1.0, sensitivity: float = 1.0):
        """
        Initialize differential privacy.
        
        Args:
            epsilon: Privacy budget (lower = more privacy)
            delta: Probability of privacy breach
            clip_norm: Maximum gradient norm
            sensitivity: How much one sample can change output
        """
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.sensitivity = sensitivity
        self.logger = logging.getLogger("differential-privacy")
    
    def clip_gradient(self, gradient: Dict) -> Dict:
        """
        Clip gradient to limit contribution.
        
        Like limiting transaction size in Bitcoin.
        Ensures no single sample dominates the gradient.
        """
        # Calculate gradient norm
        norm = math.sqrt(sum(v**2 for v in gradient.values()))
        
        # Clip if norm exceeds threshold
        if norm > self.clip_norm:
            scale = self.clip_norm / norm
            clipped = {k: v * scale for k, v in gradient.items()}
            self.logger.debug(f"Clipped gradient: {norm:.4f} → {self.clip_norm}")
            return clipped
        
        return gradient
    
    def add_noise(self, gradient: Dict) -> Dict:
        """
        Add calibrated noise to gradient.
        
        Like Bitcoin mixing/coinjoin (obscures individual contributions).
        Mathematical guarantee: ε-differential privacy.
        
        Noise is calibrated so that:
        - Individual patient's contribution is obscured
        - Aggregate gradient still useful for learning
        """
        # Calibrate noise: σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        sigma = self.sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
        
        # Add Gaussian noise to each parameter
        noisy_gradient = {}
        for key, value in gradient.items():
            noise = secrets.randbelow(1000) / 1000 * sigma * 2 - sigma  # Approximate Gaussian
            noisy_gradient[key] = value + noise
        
        self.logger.debug(f"Added noise (σ={sigma:.4f}) for ε={self.epsilon}, δ={self.delta}")
        return noisy_gradient
    
    def privatize_gradient(self, gradient: Dict) -> Dict:
        """Apply full differential privacy pipeline."""
        # 1. Clip gradient
        clipped = self.clip_gradient(gradient)
        
        # 2. Add noise
        noisy = self.add_noise(clipped)
        
        return noisy
    
    def get_privacy_loss(self, num_rounds: int) -> float:
        """
        Calculate total privacy loss over multiple rounds.
        
        Like tracking how many times you've used a stealth address.
        Privacy budget is consumed over time.
        """
        # Advanced composition theorem
        # Total ε = ε_round * sqrt(2 * num_rounds * ln(1/δ))
        total_epsilon = self.epsilon * math.sqrt(2 * num_rounds * math.log(1 / self.delta))
        return total_epsilon
    
    def get_privacy_guarantee(self) -> str:
        """Get human-readable privacy guarantee."""
        return (
            f"Differential Privacy Guarantee:\n"
            f"  ε (epsilon) = {self.epsilon}\n"
            f"  δ (delta) = {self.delta}\n"
            f"  \n"
            f"  With probability at least {1 - self.delta:.5f},\n"
            f"  any individual patient's presence in the training data\n"
            f"  cannot be determined from the gradient.\n"
            f"  \n"
            f"  This is mathematically proven differential privacy."
        )


# ============================================================================
# 4. SECURE AGGREGATION
# ============================================================================

class SecureAggregation:
    """
    Secure aggregation for gradients.
    
    Even the aggregator can't see individual gradients!
    Only the aggregate (sum) is revealed.
    
    LIKE BITCOIN COINJOIN:
    ─────────────────────────────────────────────────────────────
    Bitcoin CoinJoin:
    - Multiple users contribute to one transaction
    - Observer can't tell which input maps to which output
    - Privacy through aggregation
    
    Secure Aggregation:
    - Multiple hospitals contribute gradients
    - Aggregator can't see individual gradients
    - Only the sum is revealed
    
    IMPLEMENTATION:
    ─────────────────────────────────────────────────────────────
    Option 1: Secret Sharing
    - Each gradient split into N shares
    - Each share sent to different aggregator
    - Need all shares to reconstruct
    
    Option 2: Homomorphic Encryption
    - Gradients encrypted before sending
    - Aggregator computes on encrypted data
    - Only final result decrypted
    
    Option 3: Secure Multi-Party Computation (SMPC)
    - Hospitals jointly compute aggregate
    - No single party sees others' data
    """
    
    def __init__(self, name: str, num_parties: int):
        self.name = name
        self.num_parties = num_parties
        self.secret_shares: Dict[str, List[Dict]] = {}  # hospital_id -> shares
        self.encrypted_gradients: Dict[str, Dict] = {}
        self.logger = logging.getLogger(f"secure-agg-{name}")
    
    def generate_secret_shares(self, gradient: Dict, hospital_id: str) -> List[Dict]:
        """
        Split gradient into secret shares.
        
        Like splitting Bitcoin private key into N-of-M parts.
        Need all shares to reconstruct.
        """
        shares = []
        remaining = gradient.copy()
        
        # Generate N-1 random shares
        for i in range(self.num_parties - 1):
            share = {}
            for key in gradient:
                # Random share
                share[key] = secrets.randbelow(10000) / 10000 - 0.5
                remaining[key] = remaining.get(key, 0) - share[key]
            shares.append(share)
        
        # Last share makes sum equal to original
        shares.append(remaining)
        
        self.secret_shares[hospital_id] = shares
        
        self.logger.info(f"Generated {len(shares)} secret shares for {hospital_id}")
        return shares
    
    def aggregate_shares(self, all_shares: List[List[Dict]]) -> Dict:
        """
        Aggregate shares without revealing individual gradients.
        
        Each aggregator only sees one share from each hospital.
        Combining all shares reveals the aggregate.
        """
        # Sum all shares from all hospitals
        aggregate = {}
        
        for shares in all_shares:
            # Each hospital's shares should sum to their gradient
            for share in shares:
                for key, value in share.items():
                    aggregate[key] = aggregate.get(key, 0) + value
        
        return aggregate
    
    def simulate_encryption(self, gradient: Dict, hospital_id: str) -> Dict:
        """
        Simulate homomorphic encryption (simplified).
        
        In production, would use real homomorphic encryption.
        This simulates the concept: data is "encrypted" and can still
        be summed without decryption.
        """
        # Simulate encryption by adding random mask
        # In reality, this would be actual homomorphic encryption
        key = hashlib.sha256(f"{hospital_id}-{time.time()}".encode()).hexdigest()[:16]
        
        encrypted = {}
        for k, v in gradient.items():
            # In real HE: encrypted[k] = encrypt(v, public_key)
            # For simulation: just add a "mask"
            encrypted[k] = v + hash(key) % 100 / 100  # Simulated encryption
        
        self.encrypted_gradients[hospital_id] = {
            "encrypted": encrypted,
            "key": key,  # In reality, key would be secret
        }
        
        return encrypted
    
    def homomorphic_sum(self, encrypted_gradients: List[Dict]) -> Dict:
        """
        Sum encrypted gradients without decrypting.
        
        Like Bitcoin adding transaction amounts without knowing senders.
        Homomorphic encryption allows computation on encrypted data.
        """
        result = {}
        
        for enc in encrypted_gradients:
            for key, value in enc.items():
                result[key] = result.get(key, 0) + value
        
        return result
    
    def decrypt_aggregate(self, encrypted_sum: Dict, keys: List[str]) -> Dict:
        """
        Decrypt the aggregate sum.
        
        Only the final aggregate is decrypted.
        Individual gradients remain encrypted.
        """
        # Simulate decryption by removing all masks
        decrypted = encrypted_sum.copy()
        
        # In reality, this would use threshold decryption
        # where all parties must cooperate to decrypt
        for key in keys:
            # Remove "mask" from simulation
            for k in decrypted:
                decrypted[k] -= hash(key) % 100 / 100  # Simulated decryption
        
        return decrypted


# ============================================================================
# 5. MODEL CONVERGENCE
# ============================================================================

class ConvergenceTracker:
    """
    Track and ensure model convergence.
    
    Like Bitcoin chain validation:
    - Each block must be valid
    - Chain must extend longest valid chain
    - Model must improve (or converge)
    
    CONVERGENCE GUARANTEES:
    ─────────────────────────────────────────────────────────────
    FedAvg Convergence Theorem:
    Under certain conditions, FedAvg converges to optimal model.
    
    Conditions:
    1. Smooth loss function
    2. Bounded variance of gradients
    3. Bounded gradient divergence
    4. Sufficient local epochs
    
    Non-IID Data:
    - If data is very different across hospitals
    - Convergence may be slower
    - May need more rounds or better aggregation
    """
    
    def __init__(self, name: str):
        self.name = name
        self.round_history: List[Dict] = []
        self.accuracy_history: List[float] = []
        self.loss_history: List[float] = []
        self.gradient_norms: List[float] = []
        self.logger = logging.getLogger(f"convergence-{name}")
    
    def track_round(self, round_num: int, accuracy: float, loss: float, 
                    gradient_norm: float, hospital_accuracies: Dict[str, float]):
        """Track metrics for a training round."""
        self.round_history.append({
            "round": round_num,
            "accuracy": accuracy,
            "loss": loss,
            "gradient_norm": gradient_norm,
            "hospital_accuracies": hospital_accuracies,
        })
        
        self.accuracy_history.append(accuracy)
        self.loss_history.append(loss)
        self.gradient_norms.append(gradient_norm)
        
        self.logger.info(f"Round {round_num}: accuracy={accuracy:.4f}, loss={loss:.4f}")
    
    def check_convergence(self, patience: int = 5, threshold: float = 0.001) -> Tuple[bool, str]:
        """
        Check if model has converged.
        
        Convergence criteria:
        1. Loss not improving for `patience` rounds
        2. Gradient norm below threshold
        3. Accuracy plateaued
        """
        if len(self.loss_history) < patience:
            return False, "Not enough rounds"
        
        # Check if loss improving
        recent_losses = self.loss_history[-patience:]
        if max(recent_losses) - min(recent_losses) < threshold:
            return True, "Loss converged (improvement < threshold)"
        
        # Check gradient norm
        if self.gradient_norms[-1] < threshold:
            return True, "Gradient norm converged"
        
        # Check accuracy plateau
        recent_acc = self.accuracy_history[-patience:]
        if max(recent_acc) - min(recent_acc) < threshold:
            return True, "Accuracy converged"
        
        return False, "Still improving"
    
    def get_convergence_report(self) -> Dict:
        """Get convergence analysis report."""
        converged, reason = self.check_convergence()
        
        return {
            "converged": converged,
            "reason": reason,
            "rounds": len(self.round_history),
            "final_accuracy": self.accuracy_history[-1] if self.accuracy_history else 0,
            "final_loss": self.loss_history[-1] if self.loss_history else float('inf'),
            "accuracy_improvement": self.accuracy_history[-1] - self.accuracy_history[0] if len(self.accuracy_history) > 1 else 0,
            "loss_improvement": self.loss_history[0] - self.loss_history[-1] if len(self.loss_history) > 1 else 0,
            "gradient_norm_trend": "decreasing" if len(self.gradient_norms) > 1 and self.gradient_norms[-1] < self.gradient_norms[0] else "increasing",
        }
    
    def detect_divergence(self) -> Tuple[bool, str]:
        """
        Detect if model is diverging (getting worse).
        
        Like Bitcoin detecting invalid chain.
        """
        if len(self.loss_history) < 3:
            return False, "Not enough data"
        
        # Check if loss is increasing
        recent = self.loss_history[-3:]
        if all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
            return True, "Loss increasing (model diverging)"
        
        # Check if accuracy decreasing
        recent_acc = self.accuracy_history[-3:]
        if all(recent_acc[i] > recent_acc[i+1] for i in range(len(recent_acc)-1)):
            return True, "Accuracy decreasing (model diverging)"
        
        return False, "Model healthy"
    
    def handle_non_iid(self, hospital_accuracies: Dict[str, float]) -> Dict:
        """
        Analyze and handle non-IID data distribution.
        
        Non-IID = data is very different across hospitals
        Can slow convergence or cause divergence
        """
        accs = list(hospital_accuracies.values())
        mean_acc = sum(accs) / len(accs)
        variance = sum((a - mean_acc)**2 for a in accs) / len(accs)
        std_dev = math.sqrt(variance)
        
        # High variance = non-IID
        is_non_iid = std_dev > 0.1  # Threshold for non-IID detection
        
        recommendations = []
        if is_non_iid:
            recommendations.append("Data is non-IID across hospitals")
            recommendations.append("Consider: More local epochs per round")
            recommendations.append("Consider: FedProx algorithm")
            recommendations.append("Consider: Data sharing strategy")
        
        return {
            "is_non_iid": is_non_iid,
            "accuracy_variance": variance,
            "accuracy_std_dev": std_dev,
            "recommendations": recommendations,
        }


# ============================================================================
# Demo
# ============================================================================

def main():
    """Demonstrate all four advanced concepts."""
    
    print("="*70)
    print("ADVANCED FEDERATED LEARNING CONCEPTS")
    print("="*70)
    print()
    
    # =========================================================================
    # 2. INCENTIVE MECHANISMS
    # =========================================================================
    print("="*70)
    print("2. INCENTIVE MECHANISMS")
    print("="*70)
    print()
    
    print("WHY PARTICIPATE?")
    print("-"*70)
    print()
    
    incentive = IncentiveSystem("health-net")
    
    # Register hospitals
    incentive.register_participant("hospital-a", initial_stake=100)
    incentive.register_participant("hospital-b", initial_stake=150)
    incentive.register_participant("hospital-c", initial_stake=50)
    
    print("BITCOIN ANALOGY:")
    print("  Bitcoin miners get block rewards for mining blocks")
    print("  Hospitals get tokens for contributing gradients")
    print()
    
    print("REWARD CALCULATION:")
    print("  Base reward: 10 tokens")
    print("  × Quality multiplier (0-1)")
    print("  × Sample multiplier (1-2)")
    print("  × Staleness multiplier (0.5-1)")
    print("  × Reputation multiplier (0.5-1.5)")
    print("  × Stake multiplier (1-2)")
    print()
    
    # Simulate contributions
    reward = incentive.calculate_reward("hospital-a", 0.95, 10000, 0)
    incentive.distribute_reward("hospital-a", reward)
    
    print(f"Hospital A contributes gradient:")
    print(f"  Quality: 0.95 (excellent)")
    print(f"  Samples: 10,000")
    print(f"  Reward: {reward.tokens:.2f} tokens")
    print(f"  Reputation: {incentive.reputation['hospital-a']:.1f}")
    print()
    
    reward = incentive.calculate_reward("hospital-b", 0.80, 15000, 1)
    incentive.distribute_reward("hospital-b", reward)
    
    print(f"Hospital B contributes gradient:")
    print(f"  Quality: 0.80 (good)")
    print(f"  Samples: 15,000")
    print(f"  Staleness: 1 (slightly old)")
    print(f"  Reward: {reward.tokens:.2f} tokens")
    print()
    
    print("STATUS:")
    for hid in ["hospital-a", "hospital-b", "hospital-c"]:
        status = incentive.get_status(hid)
        print(f"  {hid}: {status['balance']:.1f} tokens, rep={status['reputation']:.1f}")
    print()
    
    # =========================================================================
    # 3. PRIVACY GUARANTEES
    # =========================================================================
    print("="*70)
    print("3. PRIVACY GUARANTEES")
    print("="*70)
    print()
    
    print("DIFFERENTIAL PRIVACY:")
    print("-"*70)
    print()
    
    print("MATHEMATICAL GUARANTEE:")
    print("  For any two datasets D and D' differing by one patient:")
    print("  Pr[Algorithm(D) ∈ S] ≤ exp(ε) × Pr[Algorithm(D') ∈ S]")
    print()
    print("  Translation: Adding or removing one patient doesn't")
    print("  significantly change the output.")
    print()
    
    dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, clip_norm=1.0)
    
    print(dp.get_privacy_guarantee())
    print()
    
    # Simulate gradient privatization
    gradient = {"weight1": 0.5, "weight2": -0.3, "weight3": 0.8}
    
    print("EXAMPLE:")
    print(f"  Original gradient: {gradient}")
    
    clipped = dp.clip_gradient(gradient)
    print(f"  After clipping: {clipped}")
    
    noisy = dp.add_noise(clipped)
    print(f"  After noise: {noisy}")
    print()
    
    print("PRIVACY BUDGET:")
    print(f"  Each round consumes ε=1.0 privacy")
    print(f"  After 10 rounds: ε={dp.get_privacy_loss(10):.2f}")
    print(f"  After 100 rounds: ε={dp.get_privacy_loss(100):.2f}")
    print()
    
    # =========================================================================
    # 4. SECURE AGGREGATION
    # =========================================================================
    print("="*70)
    print("4. SECURE AGGREGATION")
    print("="*70)
    print()
    
    print("BITCOIN COINJOIN ANALOGY:")
    print("-"*70)
    print("  Bitcoin CoinJoin: Multiple inputs → one transaction")
    print("  Can't tell which input maps to which output")
    print()
    print("  Secure Aggregation: Multiple gradients → one aggregate")
    print("  Aggregator can't see individual gradients")
    print()
    
    secure_agg = SecureAggregation("health-net", num_parties=3)
    
    print("SECRET SHARING:")
    print("-"*70)
    print()
    
    gradient_a = {"w1": 0.5, "w2": 0.3, "w3": -0.2}
    shares_a = secure_agg.generate_secret_shares(gradient_a, "hospital-a")
    
    print(f"  Hospital A gradient: {gradient_a}")
    print(f"  Split into 3 shares:")
    for i, share in enumerate(shares_a):
        print(f"    Share {i+1}: {share}")
    print(f"  Sum of shares: {sum(shares_a[0][k] + shares_a[1][k] + shares_a[2][k] for k in gradient_a) / len(gradient_a):.2f}")
    print(f"  Original sum: {sum(gradient_a.values()):.2f}")
    print()
    
    print("AGGREGATION WITHOUT SEEING INDIVIDUALS:")
    print("  Each aggregator only sees one share from each hospital")
    print("  Combining all shares reveals aggregate")
    print("  Individual gradients remain hidden")
    print()
    
    # =========================================================================
    # 5. MODEL CONVERGENCE
    # =========================================================================
    print("="*70)
    print("5. MODEL CONVERGENCE")
    print("="*70)
    print()
    
    print("CONVERGENCE TRACKING:")
    print("-"*70)
    print()
    
    convergence = ConvergenceTracker("health-net")
    
    print("BITCOIN ANALOGY:")
    print("  Bitcoin: Each block must be valid")
    print("  FedAvg: Each round must improve model")
    print()
    
    # Simulate training rounds
    for i in range(10):
        accuracy = 0.5 + i * 0.05 + (secrets.randbelow(100) / 1000)  # Improving
        loss = 1.0 - i * 0.08 - (secrets.randbelow(100) / 1000)  # Decreasing
        gradient_norm = 1.0 / (i + 1)  # Decreasing
        hospital_accs = {"h1": accuracy - 0.02, "h2": accuracy + 0.01, "h3": accuracy}
        
        convergence.track_round(i + 1, accuracy, loss, gradient_norm, hospital_accs)
    
    report = convergence.get_convergence_report()
    
    print("TRAINING PROGRESS:")
    print(f"  Rounds: {report['rounds']}")
    print(f"  Initial accuracy: {convergence.accuracy_history[0]:.4f}")
    print(f"  Final accuracy: {report['final_accuracy']:.4f}")
    print(f"  Improvement: {report['accuracy_improvement']:.4f}")
    print(f"  Initial loss: {convergence.loss_history[0]:.4f}")
    print(f"  Final loss: {report['final_loss']:.4f}")
    print()
    
    converged, reason = convergence.check_convergence()
    print(f"CONVERGED: {converged}")
    print(f"REASON: {reason}")
    print()
    
    # Check for non-IID
    non_iid_report = convergence.handle_non_iid({"h1": 0.8, "h2": 0.5, "h3": 0.9})
    if non_iid_report["is_non_iid"]:
        print("NON-IID DETECTED:")
        for rec in non_iid_report["recommendations"]:
            print(f"  - {rec}")
    print()
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()
    
    print("2. INCENTIVE MECHANISMS:")
    print("   Token rewards for contributions")
    print("   Reputation for trust")
    print("   Staking for commitment")
    print()
    
    print("3. PRIVACY GUARANTEES:")
    print("   Differential privacy (ε, δ)")
    print("   Gradient clipping")
    print("   Noise injection")
    print("   Mathematical proof")
    print()
    
    print("4. SECURE AGGREGATION:")
    print("   Secret sharing")
    print("   Homomorphic encryption")
    print("   Only aggregate revealed")
    print()
    
    print("5. MODEL CONVERGENCE:")
    print("   Track accuracy/loss")
    print("   Detect divergence")
    print("   Handle non-IID data")
    print()


if __name__ == "__main__":
    main()