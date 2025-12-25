# Role
You are an expert Senior Software Architect and Python Developer specializing in Bioinformatics pipelines, File I/O, and Agentic AI.

# Project Overview
We are building the **"Bio-Data Harmonizer,"** a system to clean and standardize messy clinical data into **Ingenuity Pathway Analysis [IPA]** format.

The system features two key innovations:
1.  **Universal Ingestion:** It must accept *any* file type (tar, gz, nested archives, unknown binaries) and figure out how to extract the clinical data, using an LLM if necessary.
2.  **Noise-Matching RAG:** It repairs data by matching the "error fingerprint" to a Vector Map of past errors and retrieving the associated Python fix scripts.

# Architecture Requirements
Please generate the file structure and core Python code for the following modules.

## 1. The Universal Ingestion Engine
Create a class `SmartIngestor`.
* **Layer 1 (Hard-coded):** Use `python-magic` or file signatures to detect the actual file type.
* **Layer 2 (Standard):** Implement robust extraction for `.zip`, `.tar`, `.gz`, `.rar`, and `.7z`. It must handle nested archives recursively (e.g., `data.tar.gz`).
* **Layer 3 (Agentic Fallback):** If standard extraction fails, the system should trigger a `FormatScout` LLM.
    * *Logic:* Pass the file header/metadata to the LLM. Ask it to identify the format and generate a Python snippet to parse it into a pandas DataFrame.

## 2. The Pre-Made Standard (PSE) Generator
Create a class `NoiseGenerator`.
* **Requirements:**
    * Dynamic noise **Quantity**: Configurable % (Noisy/Total).
    * Dynamic noise **Distribution**: Target specific columns vs. uniform spread.
    * Dynamic noise **Quality**: Inject typos, semantic drift, and structural shifts.

## 3. The Vectorizer (The Fingerprint)
Create a class `NoiseVectorizer`.
* **Logic:** Do NOT embed raw text content. Embed the *structure* of the noise.
* **Strategy:** Convert columns into metadata tokens (e.g., `COL_1_INT_WITH_NANS`) and use TF-IDF to create the vector.

## 4. The Vector Database (The Map)
Design a schema (using ChromaDB or FAISS) to store:
* Vectors, Metadata, and **Attachments** (`script_content`, `status` ["SUCCESS", "FAILURE"]).

## 5. The Agentic Loop (Diagnostic & Fixing)
Sketch the logic for the agents:
* **Diagnostic Agent:** Checks data against IPA rules. Sees *nearby failed scripts* in the Map to avoid repeating mistakes.
* **Fixing Agent:** Writes fix scripts. Sees *nearby successful scripts* in the Map to use as templates.

# Deliverables
1.  File structure (e.g., `src/ingestion.py`, `src/vectorizer.py`, `src/agents.py`).
2.  **Detailed Python code for `SmartIngestor`** showing the recursion and LLM fallback logic.
3.  Python code for `NoiseGenerator` and `NoiseVectorizer`.
4.  A high-level `main.py` flow.