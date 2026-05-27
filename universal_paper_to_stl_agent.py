#!/usr/bin/env python3
"""
universal_paper_to_stl_agent.py

Universal paper-PDF -> GPT-generated geometry builder -> STL files.

Key idea:
A single hard-coded STL generator will never support every metamaterial topology.
Instead, this script asks GPT to write a paper-specific Python builder script using
the evidence in the uploaded PDF, then runs that builder locally. If the builder
fails, the script sends the error back to GPT and asks for a repaired version.

This is closer to what happens in ChatGPT UI: the model reasons from the paper,
writes custom code for that paper, and iterates.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


SYSTEM_PROMPT = r"""
You are an expert metamaterial CAD/STL reconstruction agent.

Task:
Given a scientific paper PDF, extract the metamaterial design rules from text, equations,
tables, captions, and figures, then write a complete Python builder script that generates
downloadable STL files.

Very important:
- Do not hallucinate missing CAD details.
- Do not claim exact reconstruction unless the paper provides enough equations, dimensions,
  parameters, and construction rules to reproduce the geometry.
- If design rules are incomplete, generate the best evidence-based approximation and label it
  as parameter_rich, figure_traced, or topology_inspired.
- If geometry cannot be responsibly generated, write a builder script that creates no STL for
  that structure but writes metadata explaining why.

The generated builder script must:
1. Be self-contained except for importing geometry_helpers.py from the same folder.
2. Use only these allowed packages:
   numpy, scipy, trimesh, skimage, PIL, cv2, fitz, shapely, mapbox_earcut,
   json, pathlib, math, argparse, os, sys.
3. Accept:
      python generated_builder.py --out output_folder --pdf input_pdf
4. Create:
      output_folder/stl/*.stl
      output_folder/metadata/*.json
      output_folder/preview/*.png if helpful
5. For each STL, write metadata JSON with:
      source_paper
      structure_name
      reconstruction_level
      confidence_0_to_1
      directly_used
      visually_inferred
      assumptions
      missing_information
      geometry_parameters
      generated_stl
6. Use geometry_helpers primitives whenever useful:
      cylinder_between, combine_meshes, export_mesh, tpms_to_mesh,
      voxel_mask_to_mesh, render_pdf_page, crop_relative, image_to_binary_mask,
      extrude_mask_to_mesh, box_mesh
7. Never use external internet or download files.
8. Never require manual interaction.
9. Prefer robust generation over beautiful code.

Write accurate, runnable Python code.
"""

USER_PROMPT = r"""
Refer to the uploaded paper.

Can you identify the design rules mentioned in this paper using text, equations, tables,
figure captions, labelled dimensions, and visible figures, and design the structure as closely
as possible using the extracted design rules?

Generate a paper-specific Python builder script that creates STL files.

The script should support any structure type present in the paper, not only TPMS:
- TPMS implicit surfaces
- re-entrant auxetic lattices
- chiral/tetrachiral lattices
- honeycomb and hybrid lattices
- rotating-unit structures
- truss/beam lattices
- 2D cutout plates extruded into 3D
- tubular structures
- shell/plate lattices
- image-traced structures when only figures are available

Use this reconstruction hierarchy:
1. exact_from_equation: equations and parameters are sufficient.
2. parameter_rich: dimensions/topology are given but full CAD constraints are missing.
3. figure_traced: mainly traced from figure/image.
4. topology_inspired: visual concept only.
5. insufficient: no responsible STL generation possible.

The Python builder should create all STL files that can be generated responsibly and should
also write metadata explaining exactly what was directly extracted, inferred, assumed, and missing.

Return only JSON with:
{
  "paper_title": "...",
  "high_level_plan": "...",
  "expected_structures": ["..."],
  "limitations": ["..."],
  "builder_code": "complete python code here"
}
"""

REPAIR_PROMPT = r"""
The previous generated Python builder failed or produced incomplete output.

Repair the builder code. Keep the same goal, same evidence-based reconstruction rules, and same output format.
Use the error log below and return only JSON with the same keys:
{
  "paper_title": "...",
  "high_level_plan": "...",
  "expected_structures": ["..."],
  "limitations": ["..."],
  "builder_code": "complete repaired python code here"
}

Error/output log:
"""

HELPER_API_NOTES = r"""
Available geometry_helpers.py API notes:
- export_mesh(mesh, path)
- voxel_mask_to_mesh(mask, spacing, origin=(0,0,0))
- scalar_field_to_mesh(field, xmin, xmax, ymin, ymax, zmin, zmax, level=0.0, solid_below=True)
- tpms_to_mesh("gyroid", size_mm=(x,y,z), cells=(nx,ny,nz), iso=0.0, half_thickness=0.15)
- tpms_to_mesh(field, xmin, xmax, ymin, ymax, zmin, zmax, level=0.0)
- cylinder_between(p0, p1, radius)
- strut_lattice_mesh(nodes, edges, radius)
- box_mesh(size, center=(0,0,0))
- render_pdf_page(pdf_path, page_number, dpi=400)
- image_to_binary_mask(img, solid_is_dark=True)
- extrude_mask_to_mesh(mask, width_mm, height_mm, depth_mm)

If you compute a custom scalar field as a 3D numpy array, call:
    mesh = tpms_to_mesh(field, xmin, xmax, ymin, ymax, zmin, zmax, level=0.0)
or:
    mesh = scalar_field_to_mesh(field, xmin, xmax, ymin, ymax, zmin, zmax, level=0.0)

Do not pass both a positional iso/level and cells=... when using a custom scalar field.
"""

JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "paper_title": {"type": "string"},
        "high_level_plan": {"type": "string"},
        "expected_structures": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "builder_code": {"type": "string"},
    },
    "required": ["paper_title", "high_level_plan", "expected_structures", "limitations", "builder_code"],
}


def load_local_env(env_path: Path = Path(".env")) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_api_key(api_key: Optional[str], ask_api_key: bool) -> Optional[str]:
    if api_key:
        return api_key.strip()
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"].strip()
    if ask_api_key:
        return getpass.getpass("Enter OpenAI API key (hidden; not saved): ").strip()
    return None


def call_model_json(
    client: OpenAI,
    model: str,
    file_id: str,
    prompt: str,
    reasoning_effort: str = "none",
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": file_id},
                    {"type": "input_text", "text": f"{prompt}\n\n{HELPER_API_NOTES}"},
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "paper_specific_builder",
                "strict": True,
                "schema": JSON_SCHEMA,
            }
        },
    }
    if reasoning_effort and reasoning_effort.lower() != "none":
        kwargs["reasoning"] = {"effort": reasoning_effort}

    try:
        resp = client.responses.create(**kwargs)
    except Exception:
        kwargs.pop("reasoning", None)
        resp = client.responses.create(**kwargs)
    return json.loads(resp.output_text)


def write_builder(out_dir: Path, code: str, attempt: int) -> Path:
    builder_path = out_dir / f"generated_builder_attempt_{attempt}.py"
    builder_path.write_text(code, encoding="utf-8")
    return builder_path


def run_builder(builder_path: Path, pdf_path: Path, out_dir: Path, timeout_s: int) -> tuple[bool, str]:
    builder_path = builder_path.resolve()
    pdf_path = pdf_path.resolve()
    out_dir = out_dir.resolve()
    run_out = out_dir / "builder_run"
    run_out.mkdir(parents=True, exist_ok=True)

    helper_src = Path(__file__).with_name("geometry_helpers.py")
    helper_dst = builder_path.parent / "geometry_helpers.py"
    if helper_src.exists() and helper_src.resolve() != helper_dst.resolve():
        shutil.copy2(helper_src, helper_dst)

    cmd = [
        sys.executable,
        str(builder_path),
        "--out",
        str(run_out),
        "--pdf",
        str(pdf_path),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(builder_path.parent),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
    )
    log = proc.stdout

    stl_dir = run_out / "stl"
    has_stl = stl_dir.exists() and any(stl_dir.glob("*.stl"))

    if has_stl:
        for sub in ["stl", "metadata", "preview"]:
            src = run_out / sub
            dst = out_dir / sub
            if src.exists():
                dst.mkdir(parents=True, exist_ok=True)
                for p in src.glob("*"):
                    if p.is_file():
                        shutil.copy2(p, dst / p.name)

    return (proc.returncode == 0 and has_stl), log


def make_report(out_dir: Path, result: Dict[str, Any], success: bool, final_log: str) -> None:
    lines = []
    lines.append("# Paper-to-STL generated reconstruction report\n")
    lines.append(f"**Paper title:** {result.get('paper_title', '')}\n")
    lines.append("## High-level plan\n")
    lines.append(result.get("high_level_plan", "") + "\n")
    lines.append("## Expected structures\n")
    for x in result.get("expected_structures", []):
        lines.append(f"- {x}")
    lines.append("\n## Limitations\n")
    for x in result.get("limitations", []):
        lines.append(f"- {x}")
    lines.append("\n## Run status\n")
    lines.append(f"Success: `{success}`\n")
    lines.append("## Final builder log\n")
    lines.append("```text")
    lines.append(final_log[-6000:])
    lines.append("```")
    lines.append("\n## Accuracy note\n")
    lines.append(
        "This pipeline supports many metamaterial types by asking GPT to write a custom builder script. "
        "However, exact CAD reproduction is only possible when the paper provides complete design rules. "
        "For figure-only or partial-CAD papers, the output must be treated as an evidence-based approximation."
    )
    (out_dir / "design_report.md").write_text("\n".join(lines), encoding="utf-8")


def zip_outputs(out_dir: Path) -> Path:
    zip_path = out_dir / "output_bundle.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in out_dir.rglob("*"):
            if p.is_file() and p.name != zip_path.name:
                z.write(p, arcname=p.relative_to(out_dir))
    return zip_path


def run_agent(
    pdf_path: Path,
    out_dir: Path,
    model: str = "gpt-4.1",
    api_key: Optional[str] = None,
    ask_api_key: bool = False,
    reasoning_effort: str = "none",
    max_repairs: int = 2,
    timeout_s: int = 300,
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[bool, Path]:
    def log(message: str) -> None:
        if progress:
            progress(message)
        else:
            print(message)

    load_local_env()
    out_dir.mkdir(parents=True, exist_ok=True)

    if OpenAI is None:
        raise RuntimeError("Install openai: pip install openai")
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    key = resolve_api_key(api_key, ask_api_key)
    if not key:
        raise RuntimeError("No API key. Use --ask-api-key, --api-key, or OPENAI_API_KEY.")

    client = OpenAI(api_key=key)

    log("[INFO] Uploading PDF...")
    with pdf_path.open("rb") as fh:
        uploaded = client.files.create(file=fh, purpose="user_data")

    log("[INFO] Asking GPT to generate paper-specific STL builder...")
    result = call_model_json(client, model, uploaded.id, USER_PROMPT, reasoning_effort)
    (out_dir / "model_plan_attempt_0.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    success = False
    final_log = ""
    current = result

    for attempt in range(max_repairs + 1):
        builder = write_builder(out_dir, current["builder_code"], attempt)
        log(f"[INFO] Running builder attempt {attempt}: {builder.name}")
        try:
            success, final_log = run_builder(builder, pdf_path, out_dir, timeout_s)
        except subprocess.TimeoutExpired as exc:
            success = False
            final_log = f"Timeout after {timeout_s} seconds.\n{exc}"
        except Exception as exc:
            success = False
            final_log = f"Exception while running builder:\n{repr(exc)}"

        (out_dir / f"builder_log_attempt_{attempt}.txt").write_text(final_log, encoding="utf-8")

        if success:
            log("[INFO] Builder succeeded.")
            break

        if attempt < max_repairs:
            log("[WARN] Builder failed. Asking GPT to repair...")
            repair_text = (
                REPAIR_PROMPT
                + "\n\n"
                + final_log[-12000:]
                + "\n\nOutput tree:\n"
                + "\n".join(str(p.relative_to(out_dir)) for p in sorted(out_dir.rglob("*")) if p.is_file())[-6000:]
                + "\n\nPrevious builder code:\n"
                + current["builder_code"][-20000:]
            )
            current = call_model_json(client, model, uploaded.id, repair_text, reasoning_effort)
            (out_dir / f"model_plan_attempt_{attempt + 1}.json").write_text(
                json.dumps(current, indent=2), encoding="utf-8"
            )

    make_report(out_dir, current, success, final_log)
    bundle = zip_outputs(out_dir)
    return success, bundle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Universal GPT paper-to-STL agent.")
    p.add_argument("pdf", type=Path, help="Input PDF paper.")
    p.add_argument("--out", type=Path, default=Path("universal_paper_to_stl_output"))
    p.add_argument("--model", default="gpt-4.1", help="PDF-capable OpenAI model.")
    p.add_argument("--api-key", default=None)
    p.add_argument("--ask-api-key", action="store_true")
    p.add_argument("--reasoning-effort", default="none", help="none/low/medium/high if supported.")
    p.add_argument("--max-repairs", type=int, default=2)
    p.add_argument("--timeout-s", type=int, default=300)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    success, bundle = run_agent(
        pdf_path=args.pdf,
        out_dir=args.out,
        model=args.model,
        api_key=args.api_key,
        ask_api_key=args.ask_api_key,
        reasoning_effort=args.reasoning_effort,
        max_repairs=args.max_repairs,
        timeout_s=args.timeout_s,
    )

    print("\nDone.")
    print("Success:", success)
    print("Output:", args.out)
    print("Bundle:", bundle)
    if not success:
        print("No STL was generated successfully. Check design_report.md and builder logs.")


if __name__ == "__main__":
    main()
