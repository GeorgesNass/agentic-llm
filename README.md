# Agentic LLM

A curated **Agentic LLM suite** regrouping modular, production-oriented building blocks for real-world Large Language Model systems.

This repository acts as a **parent container** for multiple independent sub-projects, each focused on a specific capability of agentic LLM systems.

## Repository Structure

- **llm-proxy-gateway**  
  LLM routing and proxy layer for model selection, policies, observability, and cost control.

- **rag-drive-gcp**  
  Retrieval-Augmented Generation pipeline built on Google Drive, OCR, GCS, Vertex AI, and Streamlit.

- **local-finetuning**  
  Local fine-tuning workflows for domain adaptation, dataset preparation, and evaluation.

- **local-quantization**  
  Quantization pipelines for efficient local inference (CPU / GPU).

Each sub-project is:
- self-contained
- independently testable
- versioned in its own repository
- integrated here using **git subtree**

## Philosophy

This suite is designed with:
- clean architecture
- explicit configuration
- production-grade error handling
- minimal coupling between components

The goal is to demonstrate **end-to-end agentic LLM system design**, not isolated scripts.

## Contact

GitHub: https://github.com/GeorgesNass  
LinkedIn: Georges Nassopoulos