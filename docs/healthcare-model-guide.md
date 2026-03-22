# Healthcare Model Recommendations for LISA_FTM

## Base Model Guide

### Model Comparison Table

| Model | Size | Speed (CPU) | Speed (GPU) | Domain | Best for |
|-------|------|-------------|-------------|--------|---------|
| **Pythia-70m** | 70M | ~2s/step | ~0.1s/step | General | Pilot only, structured data |
| **BioBERT-base** | 110M | ~5s/step | ~0.2s/step | Biomedical | Good all-around healthcare |
| **PubMedBERT** | 110M | ~5s/step | ~0.2s/step | Biomedical | Best general biomedical |
| **ClinicalBERT** | 110M | ~5s/step | ~0.2s/step | Clinical notes | MIMIC, clinical text |
| **TinyLlama** | 1.1B | ~50s/step | ~1s/step | General | Production (Mac Mini), no GPU |
| **GatorTron-890M** | 890M | ~15s/step | ~0.5s/step | Clinical | Good clinical, moderate hardware |
| **GatorTron-3B** | 3B | minutes/step | ~3s/step | Clinical | Best quality, needs GPU |
| **Mistral-7B** | 7B | impractical | ~8s/step | General | Cloud GPU only |
| **Llama-3-8B** | 8B | impractical | ~10s/step | General | Cloud GPU only |

### Hardware Requirements

| Model | RAM (fp32) | RAM (fp16) | RAM (int8) | VRAM (fp16) |
|-------|-----------|-----------|-----------|------------|
| Pythia-70m | 280MB | 140MB | 70MB | 512MB |
| PubMedBERT | 440MB | 220MB | 110MB | 1GB |
| TinyLlama | 4.4GB | 2.2GB | 1.1GB | 4GB |
| GatorTron-3B | 12GB | 6GB | 3GB | 8GB |
| Mistral-7B |  28GB | 14GB | 7GB | 16GB |

---

## Use Case Recommendations

### Infectious Disease Surveillance (Pilot)
**Recommended: Pythia-70m**
- Simple structured data (culture results, susceptibility panels)
- Task is pattern detection, not complex reasoning
- Proof-of-concept — prove the pipeline works first
- Can always upgrade to PubMedBERT later

### Oncology Immunotherapy (Pilot)
**Recommended: PubMedBERT-base → TinyLlama-1.1B**
- Domain knowledge (genomic markers, treatment protocols) matters
- PubMedBERT already understands oncology concepts
- TinyLlama if running on Mac Mini CPU
- GatorTron-3B if GPU available

### Clinical Notes / Decision Support
**Recommended: ClinicalBERT → GatorTron-3B**
- Free-text clinical notes require clinical domain knowledge
- ClinicalBERT trained on MIMIC-III directly
- GatorTron-3B for production quality

### Rare Disease (small N)
**Recommended: PubMedBERT or GatorTron-3B**
- Small patient cohorts = limited training signal
- Domain pretraining matters more with less data
- Strong biomedical base = less federated training needed

---

## Two-Phase Approach: Domain Pretraining + Federated Fine-Tuning

### The Problem with Pure Federated Fine-Tuning

A general model (Pythia, Llama) needs significant patient data to learn oncology concepts from scratch. With 200 immunotherapy patients, you may not have enough signal to teach it everything.

### The Solution: Continued Pretraining → Then Federated

```
Phase 1: Continued pretraining (offline, one-time)
──────────────────────────────────────────────────
  Source: PubMed abstracts, clinical guidelines, oncology textbooks
  (~50M-100M tokens of domain text)
  Model: TinyLlama or PubMedBERT
  Compute: Your Mac Mini, ~4-8 hours, done once
  Output: Domain-adapted model checkpoint
  
Phase 2: Federated fine-tuning (ongoing)
────────────────────────────────────────
  Source: Hospital patient data (never leaves hospital)
  Model: Domain-adapted checkpoint from Phase 1
  Method: LISA_FTM federated rounds
  Output: Hospital-specific fine-tuned model
```

### Why This Works

A model that already knows:
- "PD-L1 is an immune checkpoint marker"
- "TMB = tumor mutational burden"
- "Pembro = pembrolizumab"
- "ORR = objective response rate"

Only needs to learn from federated patient data:
- Which patients with PD-L1 > 50% respond to pembrolizumab
- At what TMB threshold the response rate changes
- Which prior treatments predict resistance

The domain pretraining does the heavy lifting on general oncology knowledge. The federated fine-tuning personalizes to YOUR patient population.

### What PubMed Data to Use

| Source | Tokens | Where to get |
|--------|--------|-------------|
| PubMed abstracts | ~15B tokens | `datasets:pubmed` |
| Clinical guidelines (NCCN, ASCO) | ~50M tokens | Public PDFs |
| Oncology textbook chapters | ~100M tokens | Public textbooks |

**Simplest approach:** Use PubMedBERT which was already pretrained on PubMed data. You just need continued pretraining on the specific patient data domain.

### Timeline Impact

| Phase | Time | What it adds |
|-------|------|-------------|
| Phase 1 (continued pretraining) | 4–8 hours (one-time) | Significant quality improvement |
| Phase 2 (federated fine-tuning) | 2–4 hours per round | Model personalized to hospital data |

**Worth doing?** Yes, if:
- The oncology pilot succeeds and becomes a production system
- You have time before the pilot starts
- You want the best possible model quality

**Not worth doing if:**
- You're in a time crunch to show a pilot result
- The pilot is just to prove the concept works
- You can iterate quickly with a simpler model

---

## Integration with LISA_FTM

To use a biomedical base model in LISA_FTM:

```python
# In real_training.py or fed_client.py
MODEL_ID = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
# or for a larger model:
# MODEL_ID = "ufhealth/llmgpt-gatortron-3b"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
```

The LoRA + LISA layer selection works the same regardless of base model. The only difference is the initial quality of the base model.

---

## Quick Start Recommendations

| Your situation | Start with |
|---------------|-----------|
| ID surveillance pilot (fast) | Pythia-70m |
| Oncology pilot on Mac Mini (no GPU) | TinyLlama-1.1B |
| Oncology pilot with Jetson Nano | PubMedBERT or GatorTron-890M |
| Oncology production (GPU available) | GatorTron-3B |
| Clinical notes pilot | ClinicalBERT |
| Rare disease (small N) | PubMedBERT + continued pretraining |
