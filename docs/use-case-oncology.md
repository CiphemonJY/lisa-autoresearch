# Use Case: Federated Oncology AI

## The Problem

Oncology AI models require large, diverse patient populations to generalize. But:

- Single hospitals have small cohorts for rare cancers
- Academic medical centers hoard data competitively
- Pharma companies have money but can't access hospital data
- Sharing raw genomic data is legally and competitively impossible
- IRB reviews for retrospective oncology studies take 6–18 months

The result: Oncology AI is fragmented, biased toward large academic centers, and pharma pays enormous premiums for access to data they didn't help generate.

## Our Solution

Federated learning across hospital oncology departments:

```
Hospital A (Oncology Dept)
  └── Immunotherapy patient outcomes, genomic markers, radiology reports
Hospital B (Cancer Center)
  └── Same data types
Hospital C (University Oncology)
  └── Clinical trial matching data
           ↓  [LISA_FTM + DP + PSI]
           ↓  No raw data ever transmitted
           ↓
      Joint Oncology AI Model
      ─────────────────────────
      • Immunotherapy response prediction
      • Genomic marker → drug matching
      • Survival modeling across populations
      • Clinical trial eligibility matching
```

## Private Set Intersection

Before training begins, PSI answers: *"Do our patient populations overlap enough to make joint training valid?"*

This is critical for oncology:
- Prove sufficient patient overlap for statistical power
- Discover cross-institutional patient cohorts for rare diseases
- Qualify for FDA accelerated approval pathways that require diverse data

## Differential Privacy

Every gradient update has calibrated Gaussian noise added before transmission. With ε ≤ 2.0, the privacy guarantee is formally defensible to an IRB board.

**ε = 1.0** = strong privacy, negligible accuracy loss for models at 70M–1B parameters
**ε = 2.0** = moderate privacy, essentially no accuracy loss

## Target Customers

### Primary: Pharma / Biotech
| Use case | Willing to pay |
|---------|---------------|
| Immunotherapy response model | $500K–$2M |
| Clinical trial matching algorithm | $250K–$1M |
| Drug combination effectiveness | $500K–$1.5M |

### Secondary: Hospital Networks
| Use case | Willing to pay |
|---------|---------------|
| Oncology decision support tool | $100K–$300K |
| Research publication partnership | $25K–$100K |
| Clinical trial recruitment | $50K–$200K |

### Tertiary: NIH / Research Grants
| Grant | Amount |
|-------|--------|
| NIH R01 (cancer AI) | $250K–$500K/yr for 5 years |
| NCI SPORE grant | $5M–$10M over 5 years |
| DOD CDMRP (cancer research) | $500K–$2M |

## Pilot Use Case: Immunotherapy Response Prediction

**Setup:**
- 2–3 hospital oncology departments
- Retrospective cohort: patients who received checkpoint inhibitors (2018–2024)
- Data fields: genomic markers (PD-L1 expression, TMB), prior treatments, demographics, outcomes
- Federated training via LISA_FTM

**Timeline:**
- Month 1–2: IRB approval (retrospective, de-identified)
- Month 3–4: Data connector deployed to each hospital
- Month 5–6: Federated training + validation
- Month 7–8: Model evaluation + joint paper drafted
- Month 9+: Pharma licensing conversations begin

**Expected outcomes:**
- Model AUC ≥ 0.75 for response prediction (vs. 0.65 single-hospital baseline)
- Joint paper in npj Precision Oncology or JCO Clinical Cancer Informatics
- 2–3 pharma leads generated from the publication

## Data Fields (MIMIC-Oncology example)

| Category | Fields |
|----------|--------|
| Demographics | Age, sex, race, ethnicity |
| Tumor characteristics | Primary site, stage at diagnosis, grade, histologic type |
| Genomic markers | PD-L1 TPS, CMS subtype, MSI status, TMB score |
| Prior treatments | Lines of therapy, specific agents, duration |
| Treatment response | ORR, PFS, OS, best overall response |
| Lab values | Albumin, LDH, hemoglobin, white blood cell count |
| Comorbidities | Charlson Comorbidity Index |

## Compliance Approach

- **HIPAA**: All data de-identified per HIPAA Safe Harbor before ingestion
- **IRB**: Retrospective de-identified study = expedited review pathway
- **FDA**: Model registered as Clinical Decision Support Software (CDSS) under 21st Century Cures Act
- **GDPR**: Not applicable for US-only deployment (add if EU expansion)
- **Sullivan Audit**: Full HIPAA BAA with each participating hospital

## Technical Architecture

```
Hospital EHR (Epic / Cerner)
         ↓  [FHIR R4 or direct SQL]
   FHIR Connector / SQL Extractor
         ↓  [De-identification + hashing of patient IDs]
   LISA_FTM Client
         ↓  [Gradient + DP noise + PSI proof]
   Federated Server (your Mac Mini, or hospital's own server)
         ↓  [Aggregated model update]
   Joint Model Checkpoint
         ↓
   [Encrypted model distributed back to each hospital]
   [Model used locally for inference only — no data leaves]
```

## Competitive Advantage

| Factor | Traditional approach | LISA_FTM approach |
|--------|--------------------|--------------------|
| Time to joint model | 18–36 months (data sharing negotiation) | 6–9 months |
| Legal cost | $100K–$500K (contracts + BAA per party) | $2K–$5K (standard BAA template) |
| Data exposure | Raw data shared → re-identification risk | Zero raw data transmitted |
| IRB complexity | Full board review for each site | Single IRB with de-identified data |
| Pharma licensing | Each hospital negotiates separately | Consortium licenses once |

## Next Steps

1. **Identify 2–3 oncology departments** willing to participate (family MD/PhD network)
2. **Submit IRB application** with de-identified retrospective cohort
3. **Deploy FHIR connectors** to each participating hospital
4. **Run federated training** on immunotherapy response dataset
5. **Draft joint paper** — first author = you, last authors = clinical champions
6. **Approach pharma** with results

---

## Administrative Intelligence

The same oncology model also powers hospital operations — no additional data collection required:

| Prediction | Typical savings |
|-----------|----------------|
| Chemotherapy volume forecast | Better infusion center staffing |
| Treatment cycle → ED visit risk | Proactive intervention planning |
| Patient acuity scoring | Nurse ratio optimization |
| No-show risk | Targeted reminder campaigns |

---

*For questions: see docs/healthcare-product-brief.md*
