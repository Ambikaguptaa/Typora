# Dataset Citation & Academic Attribution

**Project**: Mental-State Detection System using Typing Behavior  
**Document**: Research Dataset Provenance and Citation Guidelines  
**Current Status**: *Dataset citation pending final dataset acquisition.*

---

## 1. Citation Status
**Current Dataset Status**: `REAL DATASET REQUIRED -- INTEGRATION PIPELINE READY`

No empirical training dataset has yet been imported into `data/raw/`. In strict accordance with scientific integrity and research reproducibility standards:
- Software implementation and adapters are 100% complete and tested.
- Real model training remains gated and safely blocked.
- No synthetic datasets will be credited as empirical benchmarks.
- As soon as a validated real research dataset is acquired under appropriate data transfer agreements, its formal citation, DOI, and author attribution will be recorded below.
- See [docs/dataset_acquisition_checklist.md](dataset_acquisition_checklist.md) for placement and verification instructions.

---

## 2. Research Dataset Candidates Under Consideration

### Candidate A: MobileStress Dataset
- **Protocol**: Smartphone-based keystroke dynamics captured across neutral and induced cognitive stress tasks.
- **Cohort**: 20 participants, 4 sessions per participant.
- **Conditions**: Neutral vs. Stressed experimental conditions.
- **Recorded Signals**: Event timestamps, key-down / key-up events, touch coordinates, and pressure.
- **Access Protocol**: Access requires formal academic research staff agreement.

### Candidate B: CMU Keystroke Stress Dynamics Dataset
- **Associated Research**: *"Stress Detection for Keystroke Dynamics"*
- **Cohort**: 116 participants.
- **Conditions**: Baseline/neutral, induced-stress (arithmetic/temporal pressure), and recovery conditions.
- **Recorded Signals**: Millisecond keystroke timings (dwell, flight), error rates, self-report stress surveys.
- **Access Protocol**: Academic repository access subject to institutional license terms.

---

## 3. Data Attribution Guidelines

When an approved dataset is placed in `data/raw/`, update `data/raw/dataset_manifest.json` and provide the full BibTeX citation entry here:

```bibtex
@article{pending_dataset_acquisition,
  title={Dataset citation pending final dataset acquisition},
  author={To Be Specified},
  journal={Academic Keystroke Research Repository},
  year={2026}
}
```
