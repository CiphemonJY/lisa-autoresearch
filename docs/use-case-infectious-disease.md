# Use Case: Federated Infectious Disease Surveillance

## The Problem

Infectious disease surveillance in the US is broken:

- **Hospitals compete** — sharing infection data with competitors is unthinkable
- **CDC is blind** — real-time pathogen data takes weeks to reach CDC via voluntary reporting
- **Antibiotic resistance is invisible** — no hospital sees the full resistance pattern across their region
- **Outbreaks are detected too late** — regional detection requires data that no single institution has

COVID made this existential. Hospitals that shared COVID admission data were blamed for "fear-mongering." Hospitals that didn't share were blamed for the second wave.

The entire US infectious disease surveillance infrastructure relies on voluntary, delayed, incomplete reporting.

## Our Solution

Federated real-time pathogen surveillance:

```
Hospital A ─┐
Hospital B ─┼──→ Federated Server ──→ Local Alert System
Hospital C ─┘         │                    ↓
Hospital D ───────────┘            CDC / State Health Dept
                                         │
                                  (receives alerts only,
                                   not raw patient data)
```

**Every hospital sees the aggregate.** No hospital reveals its own numbers. But the federated aggregate reveals what matters: *"Sterile site MRSA is up 40% in Region X this week."*

## The Three Layers

### Layer 1: Antibiotic Resistance Surveillance

**Use case:** *"What is the current antibiotic resistance profile across our regional hospital network?"*

- Data: Culture results, susceptibility panels, antibiotic prescriptions
- PSI: Prove that the patient cohorts are distinct (no double-counting of regional patients)
- Output: Resistance prevalence heatmap by organism + region

**Who pays:**
- Hospital networks ($25K–$100K/year)
- State health departments ($50K–$200K/year)
- CDC ($100K–$500K)

**FDA Angle:** Post-market antibiotic surveillance required under 21st Century Cures Act. This fulfills it.

---

### Layer 2: Outbreak Early Detection

**Use case:** *"Has there been a statistically significant increase in Group A Strep admissions this week vs. the 5-year baseline?"*

- Data: Admission diagnoses, chief complaints, lab orders
- Method: Federated anomaly detection — each hospital computes local statistics, only the aggregate is shared
- Output: Alert threshold crossed → notification to public health authorities

**Who pays:**
- CDC (active surveillance contracts)
- State IDPH units ($100K–$500K/year)
- Hospital preparedness grants (HHS ASPR)

---

### Layer 3: Vaccine Effectiveness Tracking

**Use case:** *"How effective is the current flu vaccine across a diverse population?"*

- Data: Vaccination status, subsequent infection severity, hospitalization required
- Method: Federated cohort analysis
- Output: VE estimates by age group, strain, geography

**Who pays:**
- CDC ($250K–$1M)
- WHO ($500K–$2M globally)
- Vaccine manufacturers ($500K–$2M)

## Private Set Intersection — Why It Matters Here

PSI is critical for this use case:

1. **No patient double-counting** — If the same patient appears at two regional hospitals, PSI proves the overlap without revealing the patient's identity. This prevents inflated prevalence estimates.

2. **Cross-state analysis** — PSI can span state lines without sharing identities, enabling regional and national surveillance.

3. **Sterile site verification** — PSI can verify that positive cultures came from sterile sites (blood, CSF) without sharing which patients — critical for serious infection tracking.

## Differential Privacy for Public Health

Differential privacy is uniquely appropriate for public health:

- **Formal guarantee:** No individual patient's data influences the aggregate statistic
- **Tunable ε:** State health departments can set their own privacy thresholds
- **Auditability:** DP mechanisms can be independently verified

**Recommended ε values:**
- ε = 1.0: Strong privacy, suitable for public reporting
- ε = 0.5: Very strong privacy, for sensitive pathogen data
- ε = 2.0: Weak privacy, maximum utility — acceptable for aggregated weekly reports

## The BARDA Pitch

BARDA (Biomedical Advanced Research and Development Authority) funds epidemic preparedness technology. Their current priorities include:

- Real-time pathogen surveillance
- Antibiotic resistance tracking
- Federated data sharing systems

**The pitch to BARDA:**
> *"We're building the surveillance infrastructure that COVID showed we didn't have. LISA_FTM enables real-time pathogen monitoring across hospital networks without any hospital having to share identifiable patient data. The system pays for itself in outbreak weeks of early warning."*

**Relevant BARDA Solicitations:**
- BARDA BAA (Broad Agency Announcement) — rolling submission, $500K–$10M
- NIH AI surveillance grants — $250K–$1M

## Pilot Use Case: Regional MRSA Surveillance

**Setup:**
- 3–5 hospitals in one metropolitan area
- Federated tracking of MRSA sterile site infections (blood, bone, CSF)
- Weekly aggregated prevalence map
- Automatic alert when prevalence exceeds threshold

**Data fields:**

| Category | Fields |
|----------|--------|
| Culture data | Organism, specimen source, collection date |
| Susceptibility | Antibiotics tested, MIC values, susceptible/intermediate/resistant |
| Patient demographics | Age, sex, ZIP code (hashed) |
| Hospital metadata | Facility ID (not patient ID), bed count, ICU capacity |
| Time | Admission date, culture date (day precision, no exact timestamps) |

**Timeline:**
- Month 1: Hospital agreements + IRB (quality improvement, not research)
- Month 2: FHIR connector deployment
- Month 3: PSI verification + first aggregate report
- Month 4+: Ongoing surveillance + state health department briefing

## Compliance

| Requirement | Approach |
|-------------|---------|
| HIPAA | Aggregate statistics = not PHI. Individual alerts require de-identification. |
| State reporting laws | Each state has mandatory disease reporting — this supplements, not replaces |
| FDA | Antibiotic surveillance fulfills 21st Century Cures Act post-market requirements |
| IRB | Quality improvement surveillance typically does not require IRB |
| GDPR | Not applicable for US-only deployment |

## Why This Works for Infectious Disease

Unlike oncology (where the data has lasting scientific value), infectious disease surveillance data is **ephemeral and public by nature.** The value is in *timing*, not in the data itself.

A resistance pattern detected 2 weeks earlier saves lives and hospitals know this. The surveillance infrastructure has direct ROI: one prevented outbreak = millions in avoided infection control costs.

**The ask from hospitals is small:** Share 5 anonymized numbers per week. Get back a regional resistance map that makes you look smart to your board.

## Administrative Intelligence

The surveillance data immediately enables operational forecasting:

| Prediction | Value |
|-----------|-------|
| Seasonal flu volume forecast | ED staffing optimization, $500K–$2M/yr in avoided overtime |
| Outbreak early warning | Proactive bed capacity planning |
| Antibiotic demand prediction | Supply chain optimization, $100K–$300K/yr |
| Regional illness prevalence | Marketing + community health planning |

---

## Next Steps

1. **Identify 3 hospitals** in one region willing to participate (family MD/PhD network)
2. **Engage state IDPH** — they have statutory authority to request infection data
3. **Draft hospital agreement** — quality improvement, no IRB needed
4. **Deploy FHIR connectors** to each hospital's infection control system
5. **Run PSI verification** — prove distinct patient populations
6. **Begin weekly surveillance** — generate first regional report
7. **Approach CDC/BARDA** with proof of concept

---

*For questions: see docs/healthcare-product-brief.md*
