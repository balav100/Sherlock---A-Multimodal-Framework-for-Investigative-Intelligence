# Sherlock---A-Multimodal-Framework-for-Investigative-Intelligence

# MR. SHERLOCK
### A Multimodal Framework for Investigative Intelligence

**Connecting fragmented evidence. Revealing relationships. Supporting evidence-grounded investigations.**

---

## Overview

Investigations often involve evidence distributed across multiple formats, including images, videos, audio recordings, PDF documents, and text files. Examining these sources independently can make it difficult to identify relationships, reconstruct events, and retrieve relevant information efficiently.

**Mr. Sherlock** is a multimodal investigative intelligence framework designed to process heterogeneous evidence, extract meaningful information, and explore connections across evidence sources.

The system combines multimodal processing, Retrieval-Augmented Generation (RAG), vector similarity search, and graph-based relationship exploration to support structured investigative analysis.

Rather than treating each evidence item as an isolated document, Mr. Sherlock aims to establish a connected representation of information within an investigation.

---

## System Capabilities

| Capability | Description |
|---|---|
| Multimodal Evidence Processing | Supports image, video, audio, PDF, and text evidence ingestion. |
| Visual Understanding | Extracts visual information from image and video evidence using vision models. |
| Text Extraction | Applies OCR and document extraction techniques to retrieve textual information. |
| Audio Processing | Converts audio evidence into text for further analysis. |
| Evidence Integrity | Uses SHA-256 hashing to generate evidence fingerprints. |
| Semantic Retrieval | Uses sentence embeddings and FAISS for similarity-based evidence retrieval. |
| Graph-Based Relationships | Uses Neo4j to represent and explore relationships between investigative entities. |
| Investigative Analysis | Combines retrieved evidence and deterministic reasoning workflows to support contextual analysis. |

---

## Architecture

Mr. Sherlock follows a modular architecture that separates the user interface, backend services, evidence-processing pipelines, retrieval mechanisms, and graph storage.

### High-Level Workflow

```text
             INVESTIGATOR
                  |
                  v
          STREAMLIT INTERFACE
                  |
                  v
             FASTAPI BACKEND
                  |
                  v
          CASE & EVIDENCE MANAGEMENT
                  |
                  v
          MULTIMODAL PROCESSING
                  |
        +---------+---------+
        |         |         |
        v         v         v
      IMAGE     VIDEO     AUDIO
        |         |         |
        v         v         v
     VISION    FRAME     SPEECH
     MODEL    EXTRACTION TRANSCRIPTION
        |         |         |
        +---------+---------+
                  |
                  v
       DOCUMENT & TEXT PROCESSING
                  |
          OCR / TEXT EXTRACTION
                  |
                  v
          EVIDENCE REPRESENTATION
                  |
        +---------+---------+
        |                   |
        v                   v
   VECTOR INDEXING     GRAPH RELATIONSHIPS
        |                   |
        v                   v
       FAISS               NEO4J
        |                   |
        +---------+---------+
                  |
                  v
         EVIDENCE RETRIEVAL
                  |
                  v
       DETERMINISTIC ANALYSIS
                  |
                  v
       INVESTIGATIVE INSIGHTS
```

---

## Technology Stack

| Layer | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Backend | FastAPI |
| Vision Processing | SmolVLM2 |
| Optical Character Recognition | EasyOCR |
| Audio Transcription | Faster-Whisper |
| Text Embeddings | Sentence Transformers |
| Vector Search | FAISS |
| Graph Database | Neo4j |
| Evidence Integrity | SHA-256 |
| Retrieval Architecture | Multimodal RAG, Graph RAG |
| Testing | Pytest |

---

## Core Components

### 1. Multimodal Evidence Processing

The framework is designed to handle heterogeneous investigative evidence through dedicated processing pipelines.

- **Images:** Visual information extraction using a vision-language model.
- **Videos:** Frame extraction and subsequent visual analysis.
- **Audio:** Speech transcription for converting audio evidence into searchable text.
- **PDFs:** Document text extraction and OCR-based processing where required.
- **Text:** Direct ingestion and processing of textual evidence.

This approach provides a common representation of information extracted from different evidence formats.

### 2. Evidence Integrity Verification

Evidence integrity is an important consideration in investigative workflows.

Mr. Sherlock uses the SHA-256 hashing algorithm to generate a unique fingerprint for uploaded evidence files.

This allows the system to identify changes to stored evidence by comparing its hash with the recorded fingerprint.

### 3. Hybrid Retrieval Architecture

The retrieval layer combines semantic vector search and graph-based relationship exploration.

**Vector Retrieval — FAISS**

Sentence Transformers generate semantic embeddings that are indexed using FAISS. This enables similarity-based retrieval of relevant evidence information.

**Graph Relationships — Neo4j**

Neo4j provides a graph-based representation of entities and their relationships, supporting the exploration of connections across investigative records.

Together, these components provide complementary approaches to retrieving and examining investigative information.

### 4. Evidence-Grounded Analysis

The investigation workflow uses retrieved evidence and deterministic, rule-based processing to produce contextual investigative insights.

The objective is to keep analysis connected to available evidence rather than relying solely on unconstrained text generation.

---

## Project Structure

The following illustrates the intended modular organization. Actual filenames and directories may differ across implementation versions.

```text
Mr-Sherlock/
│
├── backend/
│   ├── main.py
│   ├── routes/
│   ├── services/
│   └── graph/
│
├── frontend/
│
├── tests/
│
├── investigation_indexes/
│
├── case_Storage/
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

Local evidence storage and generated indexes should remain outside the public Git repository.

---

## Installation and Setup

### Prerequisites

- Python 3.11
- Neo4j Desktop or a compatible Neo4j instance
- Git
- Required Python dependencies
- Hardware capable of running the selected local AI models

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Mr-Sherlock.git
cd Mr-Sherlock
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create your local `.env` file using `.env.example` as a reference.

Configure the required application settings and Neo4j connection details according to the project configuration.

Never commit credentials or private configuration files.

### 5. Start the Backend

From the project root, use:

```bash
uvicorn backend.main:app --reload --port 8001
```

### 6. Start the Frontend

Launch the Streamlit application using its actual entry-point file:

```bash
streamlit run PATH_TO_STREAMLIT_ENTRY_FILE
```

---

## Testing

The project uses Pytest for testing individual components and application workflows.

Run the test suite using:

```bash
pytest -v
```

Tests should be executed against the configured environment and required services.

---

## Design Principles

**Modularity**

Separate evidence ingestion, processing, indexing, graph operations, and investigation logic into maintainable components.

**Evidence Grounding**

Keep investigative analysis connected to retrieved evidence and identifiable source records.

**Multimodal Processing**

Support information extraction from different evidence formats within a unified investigation workflow.

**Integrity Awareness**

Use cryptographic hashing to identify changes to evidence files.

**Resource Awareness**

Consider CPU and memory constraints when selecting and executing local AI models.

---

## Research Direction

Mr. Sherlock explores the intersection of multimodal artificial intelligence, semantic information retrieval, graph-based knowledge representation, and investigative evidence analysis.

The broader research objective is to investigate how heterogeneous evidence can be represented, retrieved, and connected to support more structured investigative workflows.

---

## Limitations and Future Enhancements

Potential areas for further development include:

- Improving cross-modal evidence correlation.
- Optimizing evidence indexing and retrieval performance.
- Improving visual information extraction for complex evidence.
- Strengthening entity resolution and graph relationship construction.
- Expanding evidence-grounded answer evaluation.
- Improving asynchronous processing and frontend status updates.
- Establishing systematic evaluation datasets and reproducible benchmarks.


---

## Author

**Balasubramaniam V**

---

*Mr. Sherlock — Bringing fragmented evidence into context.*
