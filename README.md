# Omnilex Agentic Retrieval Competition Starter Repo

Official starter repo for Kaggle competiton https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/host/launch-checklist

## Quick Start

### Installation

(Tested with Ubuntu-24.04 in WSL)

```bash
# Clone the repository
git clone https://github.com/Omnilex-AI/Omnilex-Agentic-Retrieval-Competition.git
cd Omnilex-Agentic-Retrieval-Competition

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # for testing/linting

# Install package in development mode
pip install -e .
```

### Download Data

Get it from Kaggle into `data` directory

### Run Baselines

Two baseline notebooks are provided:

1. **Direct Generation** (`notebooks/01_direct_generation_baseline.ipynb`)
   - Prompts LLM to directly generate citations
   - Simple but prone to hallucination

2. **Agentic Retrieval** (`notebooks/02_agentic_retrieval_baseline.ipynb`)
   - Uses ReAct-style agent with semantic search + reranking tools
   - Grounded in actual legal documents

Both notebooks work in VSCode and can be submitted to Kaggle.

### Validate Submission

```bash
python scripts/validate_submission.py submission.csv
```

### Optional: Run Inference On Modal GPU

`omnilex.llm.load_model()` can call a deployed Modal `L40S` backend instead of the local `llama-cpp-python` runtime.

```bash
# 1. Activate the repo environment
source .venv/bin/activate

# 2. Export your Modal credentials from .env for CLI commands
set -a
source .env
set +a

# 3. Verify local authentication
modal token info

# 4. Upload your GGUF file into the Modal volume
python scripts/upload_modal_model.py models/mistral-7b-instruct-v0.2.Q4_K_M.gguf

# 5. Deploy the remote L40S backend once
modal deploy src/omnilex/llm/modal_backend.py
```

After that, local Python can switch to the remote backend:

```python
from omnilex.llm import generate, load_model

llm = load_model(
    model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    backend="modal",
    n_ctx=4096,
    chat_format="mistral-instruct",
)

response = generate(llm, "Name one famous Swiss legal code.", max_tokens=64)
print(response)
```

If you want existing code to switch without editing each `load_model(...)` call, set:

```bash
export OMNILEX_LLM_BACKEND=modal
```

Notes:

- `load_model(..., backend="modal")` returns a proxy object that forwards calls to a deployed Modal class named `RemoteLlama` in the `omnilex-remote-llm` app.
- The proxy auto-loads `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` from `.env` when they are not already present in the current Python process.
- The Modal backend expects the model file to exist in the `omnilex-llm-models` volume under `/models/<filename>`.
- The remote image installs `llama-cpp-python==0.3.16` with CUDA support for the `L40S`.

For a quick GPU connectivity check before deploying the full backend:

```bash
modal run scripts/modal_gpu_check.py
```

## Data Format

See Kaggle

## Project Structure

```
├── src/omnilex/           # Core library
│   ├── citations/         # Citation parsing & normalization
│   ├── evaluation/        # Metrics & scoring
│   ├── retrieval/         # Semantic search, reranking, and BM25 compatibility
│   └── llm/               # LLM loading & prompts
├── notebooks/             # Baseline notebooks
├── utils/                 # Data & utility scripts
├── tests/                 # Test suite
└── data/                  # Data directory
```

## Requirements

- Python >= 3.10
- llama-cpp-python (for local LLM inference)
- sentence-transformers (for semantic retrieval + reranking; installs the transformer backend)
- rank-bm25 (kept for debugging and backwards compatibility)
- pandas, numpy, scikit-learn

For Kaggle submissions, you may need to (depending on your solution):

1. Upload your GGUF model as a Kaggle dataset
2. Upload pre-built semantic indices as a Kaggle dataset
3. Package the `omnilex` library

## License

Apache 2.0 - See [LICENSE](LICENSE)

## Contact

For public questions about the competition please use the "Discussion" tab or open an issue on this repository. For private questions reahc out to host on Kaggle or ari.jordan@omnilex.ai
