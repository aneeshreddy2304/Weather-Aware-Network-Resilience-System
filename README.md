<p align="center"><img src="docs/assets/banner.svg" alt="WANRS — See how changing weather can shape network risk." width="100%"></p>

<p align="center"><img src="docs/assets/badges/python.svg" alt="python" height="28"> <img src="docs/assets/badges/streamlit.svg" alt="streamlit" height="28"> <img src="docs/assets/badges/open-meteo.svg" alt="open-meteo" height="28"> <img src="docs/assets/badges/scikit-learn.svg" alt="scikit-learn" height="28"> <img src="docs/assets/badges/tensorflow.svg" alt="tensorflow" height="28"></p>

<p align="center"><strong>A Streamlit decision-support prototype combining Open-Meteo conditions with simulated network metrics and predictive models.</strong></p>

## Overview

A Streamlit decision-support prototype combining Open-Meteo conditions with simulated network metrics and predictive models.

### Why it matters

WANRS makes environmental resilience easier to explore by connecting weather observations to interpretable risk levels and recommended network actions.

## Key features

- Live and historical Open-Meteo ingestion
- Rule-based simulation for multiple network types
- Model-backed metric predictions
- Risk classification and response recommendations

## Architecture

```mermaid
flowchart LR
  A[Open-Meteo weather] --> B[Feature pipeline]
  B --> C[Simulated network targets]
  C --> D[Predictive models]
  D --> E[Risk engine]
  E --> F[Streamlit recommendations]
```

## Technology stack

**python** · **streamlit** · **open meteo** · **scikit learn** · **tensorflow**

## Getting started

Install `requirements_txt.txt`, train or provide the expected model artifacts, then run `streamlit run app.py`. Network targets are simulated; the app does not ingest live network telemetry or apply changes automatically.
