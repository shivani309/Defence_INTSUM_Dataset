# INTSUMGen: Geo-Aware Intelligence Summarization

INTSUMGen is a reproducible synthetic dataset designed to bridge the gap between structured conflict event data and natural language intelligence summaries. The dataset enables research in geo-aware summarization, structured-to-text generation, and intelligence-focused NLP tasks.

---

##  Overview

Modern defence intelligence systems require both structured event data and unstructured narrative summaries. While datasets such as ACLED provide detailed event records, they lack corresponding intelligence-style textual reports.

INTSUMGen addresses this gap by:
- Transforming structured conflict events into intelligence summaries (INTSUMs)
- Preserving geospatial information (latitude & longitude)
- Providing multi-format datasets for machine learning and LLM fine-tuning
- Ensuring reproducibility and evaluation-driven filtering

---

## Dataset Components

The dataset is released in three formats:

### 1. Event-Specific Dataset (`synthetic_intsum_event_specific.csv`)
- Structured event data paired with generated INTSUM reports
- Includes:
  - Event type
  - Actors
  - Location (with latitude & longitude)
  - Generated intelligence summary

---

### 2. Evaluated Dataset (`synthetic_intsum_evaluated.csv`)
- Extends the base dataset with evaluation scores:
  - Accuracy
  - Completeness
  - Clarity
  - Tone

---

### 3. Instruction Dataset (`intsum_extraction_finetune_with_geo.jsonl`)
- JSONL format for LLM fine-tuning
- Each entry contains:
  - Instruction prompt
  - Structured output (actors, event type, location, geo-coordinates)

---

## Pipeline Overview

INTSUMGen is generated through a multi-stage pipeline:

1. **Data Source**  
   Structured event data from ACLED

2. **Preprocessing**
   - Filtering incomplete records
   - Standardizing geographic information
   - Normalizing actors and event types

3. **Balanced Sampling**
   - Ensures representation across event types

4. **Context Construction**
   - Combine actors, location, date, and event details

5. **LLM-based Generation**
   - Generate intelligence summaries (INTSUM format)

6. **Evaluation**
   - Score summaries using LLM-based evaluation

7. **Filtering**
   - Retain high-quality samples

8. **Instruction Dataset Creation**
   - Convert to JSONL for fine-tuning

---

##  Example INTSUM
INTSUM Report
Date: 28 April 2016
Location: Satmasa, Hooghly, West Bengal, India (22.7829, 87.8788)

Actors Involved:

Rioters
Civilians

Event Summary:
Political violence occurred involving clashes between groups in the region.

Outcome:
No fatalities reported.

Strategic Implications:

Potential escalation of local tensions

---

## 📊 Evaluation Metrics

The dataset is evaluated using:

- **Accuracy**
- **Completeness**
- **Clarity**
- **Tone**
- **Token-level F1 score**


### Key Results:
- Token F1: ~0.73
- Location consistency: ~0.80

---

## Reproducibility

- Fixed random seed (42)
- Deterministic preprocessing pipeline
- Controlled LLM generation parameters
- Evaluation and filtering thresholds defined

Evaluation performed using:
- `gpt-4o-mini` (LLM-based scoring)

---

##  Use Cases

INTSUMGen supports:

- LLM fine-tuning (instruction tuning)
- Structured-to-text generation
- Information extraction
- Geo-intelligence modeling
- Event classification
- Benchmarking NLP models

---

##  Limitations

- Synthetic (not real intelligence data)
- Template-guided generation may limit diversity
- Evaluation relies on automated scoring (LLM-based)
- Geospatial extraction may have coarse-level errors

---



##  Acknowledgements

- ACLED Dataset
- OpenAI / LLM-based evaluation
- Research in synthetic data generation and NLP
