# Structure Finder

Structure Finder is a universal AI-assisted paper-to-STL pipeline for metamaterial papers. It extracts a reconstruction plan from the PDF, uses local generators where they are reliable, and asks GPT to write a paper-specific builder for structures that need custom code. If that builder fails, the error log is sent back to GPT for repair.

It is intended to support many metamaterial classes:

- TPMS
- re-entrant auxetics
- chiral/tetrachiral cells
- honeycomb and hybrid lattices
- rotating-unit structures
- truss/beam lattices
- 2D cutout plates
- tubular/shell structures
- image-traced structures

## Install

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Web UI

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Deploy

For a private demo, Streamlit Community Cloud is the quickest path:

1. Push these files to a GitHub repository.
2. Do not commit `.env`, `.streamlit/secrets.toml`, PDFs, generated STL files, or run folders.
3. In Streamlit Community Cloud, create an app from the repository with `app.py` as the entrypoint.
4. Add this secret in Advanced settings:

```toml
OPENAI_API_KEY = "sk-..."
```

For heavier or more controlled deployment, use the included Dockerfile on Render, Railway, Fly.io, AWS, or another container host:

```powershell
docker build -t structure-finder .
docker run --rm -p 8501:8501 -e OPENAI_API_KEY="sk-..." structure-finder
```

Then open `http://localhost:8501`.

Security note: this app can ask GPT to generate Python builder code and run it locally. Keep public deployments private/authenticated, or add a stronger sandbox before accepting untrusted public uploads.

## CLI

```powershell
.\.venv\Scripts\python.exe paper_to_stl.py paper.pdf --out output --model gpt-5.5 --reasoning-effort high
```

With a hidden API-key prompt:

```powershell
.\.venv\Scripts\python.exe paper_to_stl.py paper.pdf --out output --ask-api-key --model gpt-5.5 --reasoning-effort high
```

If the selected model is unavailable to the deployed API key, the app automatically falls back through `gpt-5.2`, `gpt-5.1`, `gpt-5`, then `gpt-4.1`.

Outputs include:

- `reconstruction_plan.json`
- `reconstruction_report.md`
- `generated_builder_attempt_*.py`
- `custom_builder_log_attempt_*.txt`
- `stl/*.stl`
- `metadata/*.json`
- `paper_to_stl_output_bundle.zip`

The local TPMS and image-trace paths use higher sampling plus light smoothing to reduce rough marching-cubes surfaces while preserving measured extents.

## Accuracy Note

Exact CAD reproduction is only possible when the paper provides complete equations, dimensions, parameters, and construction rules. If the paper omits CAD constraints, the generated STL must be treated as an evidence-based approximation.
