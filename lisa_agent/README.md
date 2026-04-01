# LISA Agent - Self-Improving Agent

## Concept

A Claude-like agent that:
1. **Does tasks** using tool orchestration (exec, read, write, etc.)
2. **Uses LISA** to train custom LoRA adapters on interaction history
3. **Self-improves** by detecting failure patterns and triggering retraining

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LISA AGENT CORE                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐   │
│  │   Agent    │───▶│   Memory     │───▶│  LISA Trainer   │   │
│  │  Runtime   │    │   (History)  │    │  (Layer-by-Layer)│  │
│  └─────────────┘    └──────────────┘    └─────────────────┘   │
│        │                                       │               │
│        ▼                                       ▼               │
│  ┌─────────────┐                      ┌─────────────────┐    │
│  │   Tools    │                       │  LoRA Adapter   │    │
│  │ exec, read │                       │  (checkpoint)   │    │
│  │ write, etc │                       └─────────────────┘    │
│  └─────────────┘                              │               │
└─────────────────────────────────────────────────────────────────┘
```

## Self-Improvement Loop

1. **TASK**: Agent performs task
2. **LOG**: Store interaction in memory
3. **ANALYZE**: Detect repeated failures (same pattern 3x)
4. **TRAIN**: LISA layer-by-layer training on recent interactions
5. **UPDATE**: Load new LoRA adapter
6. **REPEAT**

## Demo Results (Jetson Orin 7.4GB)

```
Success rate improved: 50% → 72.73%
Self-improvements: 6
Checkpoint: /tmp/lisa_agent_Qwen2.5-3B.pt (0.27MB)
```

## Files

- `agent.py` - Core agent with LISA integration
- `self_improve_demo.py` - Demo of self-improvement cycle

## Usage

```python
from lisa_agent import LISAAgent, LISAAgentConfig

config = LISAAgentConfig(
    model_name="Qwen/Qwen2.5-3B",
    lisa_rank=4,
    improvement_threshold=3,
    training_steps=30
)

agent = LISAAgent(config)

# Agent runs tasks and self-improves
result = agent.run("Check system status")
```

## Next Steps

1. Add real tool execution (not just simulation)
2. Integrate with actual LLM for reasoning
3. Add semantic memory search
4. Deploy with OpenClaw bridge
