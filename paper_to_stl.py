#!/usr/bin/env python3
"""
paper_to_stl.py

Universal AI-assisted metamaterial paper -> STL pipeline.

Workflow:
1. Upload PDF to the OpenAI API.
2. Extract a strict reconstruction_plan.json with evidence, parameters, assumptions, and missing information.
3. Try local deterministic generators for TPMS implicit surfaces and image-traced 2D cutout extrusion.
4. For other structure types, ask GPT to write a paper-specific Python builder using geometry_helpers.py.
5. Run the builder locally.
6. If it fails, send the error back and repair the builder.
7. Export STL files, metadata, reports, and a zip bundle.

Recommended run:
    python paper_to_stl.py paper.pdf --out output --ask-api-key --model gpt-5 --reasoning-effort high

Use .env:
    OPENAI_API_KEY=sk-...

Important:
This script can generate exact STL only when the paper provides enough design rules.
If CAD details are missing, it labels the output as approximate/parameter-rich/figure-traced/topology-inspired.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    from skimage import measure, morphology
except Exception:
    measure = None
    morphology = None

try:
    import trimesh
except Exception:
    trimesh = None


DEFAULT_MODEL = "gpt-5.5"
DEFAULT_REASONING_EFFORT = "high"
MODEL_FALLBACKS = ["gpt-5.2", "gpt-5.1", "gpt-5", "gpt-4.1"]

SYSTEM_PROMPT = """
You are an expert metamaterial-geometry reconstruction assistant.

Your task is to read the uploaded scientific paper and extract all information needed to regenerate the reported metamaterial structures as STL files.

Use only evidence from the paper:
- text,
- equations,
- parameter tables,
- figure captions,
- labelled dimensions,
- visible figures,
- supplementary material if included in the PDF.

Do not hallucinate missing CAD details.

Always separate:
1. directly reported information,
2. visually inferred information,
3. assumptions required for STL generation,
4. missing information that prevents exact CAD reproduction.

For every important design rule or parameter, provide a source reference such as:
- Eq. 1,
- Table 1,
- Fig. 3(a),
- Section 2. Structural design,
- caption of Fig. 4.

Classify each structure using one reconstruction level:
- exact_from_equation: complete equations, dimensions, and construction rules are given.
- parameter_rich: topology and dimensions are mostly given, but CAD/sketch constraints are missing.
- figure_traced: the geometry must mainly be reconstructed from a figure or image.
- topology_inspired: only the visual concept/topology is available.
- insufficient: not enough information for responsible STL generation.

Never claim exact reconstruction unless the paper provides enough information to reproduce the geometry without hidden assumptions.

Return only valid JSON following the provided schema.
"""

DEFAULT_PROMPT = """
Refer to the uploaded paper.

Identify the metamaterial design rules mentioned in the paper using the text, equations, parameter tables, dimensions, figures, and captions.

Your goal is to create a reconstruction plan that can be used to generate downloadable STL files.

Extract:

1. Structure classification:
   TPMS, re-entrant auxetic, chiral/tetrachiral, honeycomb, rotating-unit, lattice/truss,
   shell/plate lattice, tubular structure, hybrid structure, image-traced 2D cutout, or other.

2. Geometry-generating equations:
   implicit equations, level-set equations, coordinate transformations, parametric curves,
   relative-density equations if they help define geometry, and shape/mapping functions.

3. Geometry parameters:
   For each parameter, extract symbol/name, value, unit, meaning, source reference, and whether it is
   directly reported, visually inferred, assumed, or unavailable.

4. Construction workflow:
   how the unit cell is generated, repeated, thickened, combined, mapped, sliced, extruded,
   or converted into the final 3D structure.

5. STL generation method:
   Choose one:
   - tpms_implicit
   - parametric_surface
   - strut_graph_lattice
   - curved_strut_lattice
   - two_d_cutout_extrusion
   - tubular_mapping
   - image_trace_extrusion
   - custom_builder_required
   - manual_review

6. Reconstruction confidence:
   exact_from_equation, parameter_rich, figure_traced, topology_inspired, or insufficient.

7. Missing information:
   original CAD/STL/STEP file, sketch constraints, exact node coordinates, arc centre coordinates,
   spline definitions, fillet radii, Boolean operations, hidden 3D features, mesh/export settings,
   or scale information.

8. Assumptions required to generate STL files.

If the paper provides complete mathematical design rules, create an exact equation-based reconstruction plan.

If the paper does not provide complete design rules, create the best possible parameter-inferred or figure-traced reconstruction plan, but clearly state that the STL will be approximate.
"""

CUSTOM_BUILDER_SYSTEM_PROMPT = """
You are an expert Python CAD/STL code generator for metamaterial geometries.

You are given:
1. the original paper PDF,
2. reconstruction_plan.json extracted from the paper.

Write a complete Python builder script to generate STL files for structures that require custom code.

Rules:
- Use only the paper evidence and reconstruction_plan.json.
- Do not hallucinate missing CAD details.
- If a structure is approximate, use approximate in the STL filename and metadata.
- The builder must be self-contained except importing geometry_helpers.py from the same folder.
- The builder must accept:
      python generated_builder.py --pdf input.pdf --out output_folder
- The builder must write:
      output_folder/stl/*.stl
      output_folder/metadata/*.json
      output_folder/generation_report.md
- For every STL, validate using trimesh and report extents, vertices, faces, watertight status, and connected components.
- Use actual solid printable geometry: no zero-thickness curves.
- Preserve intentional metamaterial pores/voids.
- Generate final-quality meshes, not coarse previews. For implicit or voxel geometry, use high enough sampling to avoid visibly blocky surfaces.
- Smooth only faceted marching-cubes or image-traced surfaces when the paper structure is smooth/curved, using light smoothing that preserves the bounding box and pore topology.
- Use helper functions from geometry_helpers.py when useful. Use export_mesh(..., smooth_iterations=0) by default; pass a small value only for smooth curved voxel/implicit outputs and report it in metadata.
- For filled implicit solids, use scalar_field_to_mesh(...). For TPMS/shell-type level sets, use scalar_shell_to_mesh(...).
- Do not access the internet.
Return only JSON with key builder_code.
"""

CUSTOM_BUILDER_USER_PROMPT = """
Use the uploaded PDF and reconstruction_plan.json to write a paper-specific Python STL builder.

Generate only the structures whose generator_type is not handled by the local pipeline, or whose local method needs custom handling:
parametric_surface, strut_graph_lattice, curved_strut_lattice, two_d_cutout_extrusion,
tubular_mapping, custom_builder_required, or TPMS with family custom/unknown/requires_mapping.

If the plan says a structure is insufficient or manual_review, do not generate STL for it; write metadata explaining why.

Return only JSON:
{
  "builder_code": "complete Python code here"
}
"""

REPAIR_BUILDER_PROMPT_TEMPLATE = """
The previous generated builder failed or produced no STL files.

Repair the builder code. Use the uploaded PDF and reconstruction_plan.json. Keep the same evidence-based rules.

Builder log:
{log}

Previous builder code:
{code}

Return only JSON:
{{
  "builder_code": "complete repaired Python code here"
}}
"""


CUSTOM_BUILDER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"builder_code": {"type": "string"}},
    "required": ["builder_code"],
}


DESIGN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "paper_title": {"type": "string"},
        "doi_or_identifier": {"type": "string"},
        "reconstruction_summary": {"type": "string"},
        "overall_confidence_0_to_1": {"type": "number"},
        "can_generate_stl": {"type": "boolean"},
        "reason_if_not_generatable": {"type": "string"},
        "global_missing_information": {"type": "array", "items": {"type": "string"}},
        "global_assumptions": {"type": "array", "items": {"type": "string"}},
        "structures": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "metamaterial_type": {
                        "type": "string",
                        "enum": [
                            "TPMS",
                            "re_entrant_auxetic",
                            "chiral",
                            "tetrachiral",
                            "honeycomb",
                            "rotating_unit",
                            "lattice_truss",
                            "shell_plate_lattice",
                            "tubular",
                            "hybrid",
                            "image_traced_cutout",
                            "other",
                            "unknown",
                        ],
                    },
                    "generator_type": {
                        "type": "string",
                        "enum": [
                            "tpms_implicit",
                            "parametric_surface",
                            "strut_graph_lattice",
                            "curved_strut_lattice",
                            "two_d_cutout_extrusion",
                            "tubular_mapping",
                            "image_trace_extrusion",
                            "custom_builder_required",
                            "manual_review",
                        ],
                    },
                    "reconstruction_level": {
                        "type": "string",
                        "enum": [
                            "exact_from_equation",
                            "parameter_rich",
                            "figure_traced",
                            "topology_inspired",
                            "insufficient",
                        ],
                    },
                    "confidence_0_to_1": {"type": "number"},
                    "evidence_sources": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "source_id": {"type": "string"},
                                "source_type": {
                                    "type": "string",
                                    "enum": [
                                        "equation",
                                        "table",
                                        "figure",
                                        "caption",
                                        "main_text",
                                        "supplementary",
                                        "visual_inference",
                                    ],
                                },
                                "location": {"type": "string"},
                                "evidence_summary": {"type": "string"},
                            },
                            "required": ["source_id", "source_type", "location", "evidence_summary"],
                        },
                    },
                    "equations_used": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "equation_label": {"type": "string"},
                                "equation_text": {"type": "string"},
                                "role_in_geometry": {"type": "string"},
                                "source_ref": {"type": "string"},
                            },
                            "required": ["equation_label", "equation_text", "role_in_geometry", "source_ref"],
                        },
                    },
                    "parameter_records": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string"},
                                "symbol": {"type": "string"},
                                "value": {"type": ["number", "string", "boolean", "null"]},
                                "unit": {"type": "string"},
                                "meaning": {"type": "string"},
                                "source_ref": {"type": "string"},
                                "provenance": {
                                    "type": "string",
                                    "enum": ["directly_reported", "visually_inferred", "assumed", "not_available"],
                                },
                                "confidence_0_to_1": {"type": "number"},
                            },
                            "required": [
                                "name",
                                "symbol",
                                "value",
                                "unit",
                                "meaning",
                                "source_ref",
                                "provenance",
                                "confidence_0_to_1",
                            ],
                        },
                    },
                    "dimensions_mm": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "width_x": {"type": ["number", "null"]},
                            "height_y": {"type": ["number", "null"]},
                            "depth_z": {"type": ["number", "null"]},
                            "unit_cell_size_x": {"type": ["number", "null"]},
                            "unit_cell_size_y": {"type": ["number", "null"]},
                            "unit_cell_size_z": {"type": ["number", "null"]},
                            "unit_count_x": {"type": ["number", "null"]},
                            "unit_count_y": {"type": ["number", "null"]},
                            "unit_count_z": {"type": ["number", "null"]},
                            "wall_or_strut_thickness": {"type": ["number", "null"]},
                            "face_sheet_thickness": {"type": ["number", "null"]},
                            "outer_radius": {"type": ["number", "null"]},
                            "inner_radius": {"type": ["number", "null"]},
                            "tube_height": {"type": ["number", "null"]},
                        },
                        "required": [
                            "width_x",
                            "height_y",
                            "depth_z",
                            "unit_cell_size_x",
                            "unit_cell_size_y",
                            "unit_cell_size_z",
                            "unit_count_x",
                            "unit_count_y",
                            "unit_count_z",
                            "wall_or_strut_thickness",
                            "face_sheet_thickness",
                            "outer_radius",
                            "inner_radius",
                            "tube_height",
                        ],
                    },
                    "tpms_parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "family": {
                                "type": "string",
                                "enum": ["primitive", "gyroid", "diamond", "iwp", "schwarz_p", "custom", "unknown"],
                            },
                            "custom_expression": {"type": "string"},
                            "level_low": {"type": ["number", "null"]},
                            "level_high": {"type": ["number", "null"]},
                            "iso_value": {"type": ["number", "null"]},
                            "sheet_half_thickness": {"type": ["number", "null"]},
                            "period_x": {"type": ["number", "null"]},
                            "period_y": {"type": ["number", "null"]},
                            "period_z": {"type": ["number", "null"]},
                            "requires_mapping": {"type": "boolean"},
                            "mapping_description": {"type": "string"},
                        },
                        "required": [
                            "family",
                            "custom_expression",
                            "level_low",
                            "level_high",
                            "iso_value",
                            "sheet_half_thickness",
                            "period_x",
                            "period_y",
                            "period_z",
                            "requires_mapping",
                            "mapping_description",
                        ],
                    },
                    "topology_description": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "unit_cell_description": {"type": "string"},
                            "connectivity_description": {"type": "string"},
                            "array_or_repetition_rule": {"type": "string"},
                            "boundary_or_face_sheet_rule": {"type": "string"},
                            "hybrid_combination_rule": {"type": "string"},
                        },
                        "required": [
                            "unit_cell_description",
                            "connectivity_description",
                            "array_or_repetition_rule",
                            "boundary_or_face_sheet_rule",
                            "hybrid_combination_rule",
                        ],
                    },
                    "figure_trace_info": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "needs_figure_tracing": {"type": "boolean"},
                            "best_page_number": {"type": ["number", "null"]},
                            "figure_label": {"type": "string"},
                            "crop_box_relative": {
                                "type": "array",
                                "minItems": 4,
                                "maxItems": 4,
                                "items": {"type": "number"},
                            },
                            "solid_is_dark": {"type": "boolean"},
                            "scale_reference": {"type": "string"},
                        },
                        "required": [
                            "needs_figure_tracing",
                            "best_page_number",
                            "figure_label",
                            "crop_box_relative",
                            "solid_is_dark",
                            "scale_reference",
                        ],
                    },
                    "construction_steps": {"type": "array", "items": {"type": "string"}},
                    "assumptions": {"type": "array", "items": {"type": "string"}},
                    "missing_information": {"type": "array", "items": {"type": "string"}},
                    "stl_generation_plan": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "method": {"type": "string"},
                            "description": {"type": "string"},
                            "requires_custom_code": {"type": "boolean"},
                            "expected_output_files": {"type": "array", "items": {"type": "string"}},
                            "mesh_validation_targets": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": [
                            "method",
                            "description",
                            "requires_custom_code",
                            "expected_output_files",
                            "mesh_validation_targets",
                        ],
                    },
                },
                "required": [
                    "name",
                    "metamaterial_type",
                    "generator_type",
                    "reconstruction_level",
                    "confidence_0_to_1",
                    "evidence_sources",
                    "equations_used",
                    "parameter_records",
                    "dimensions_mm",
                    "tpms_parameters",
                    "topology_description",
                    "figure_trace_info",
                    "construction_steps",
                    "assumptions",
                    "missing_information",
                    "stl_generation_plan",
                ],
            },
        },
    },
    "required": [
        "paper_title",
        "doi_or_identifier",
        "reconstruction_summary",
        "overall_confidence_0_to_1",
        "can_generate_stl",
        "reason_if_not_generatable",
        "global_missing_information",
        "global_assumptions",
        "structures",
    ],
}


LOCAL_GENERATORS = {"tpms_implicit", "image_trace_extrusion"}
CUSTOM_GENERATORS = {
    "parametric_surface",
    "strut_graph_lattice",
    "curved_strut_lattice",
    "two_d_cutout_extrusion",
    "tubular_mapping",
    "custom_builder_required",
}
MANUAL_GENERATORS = {"manual_review"}


def emit(progress: Optional[Callable[[str], None]], message: str) -> None:
    if progress:
        progress(message)
    else:
        print(message)


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


def clean_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
    safe = "_".join(part for part in safe.split("_") if part)
    return safe[:100] or "structure"


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_number(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        allowed = "".join(ch if ch.isdigit() or ch in ".-+eE" else " " for ch in value)
        for token in allowed.split():
            try:
                return float(token)
            except ValueError:
                pass
    return default


def call_responses_json(
    client: Any,
    model: str,
    system_prompt: str,
    user_content: List[Dict[str, Any]],
    schema: Dict[str, Any],
    schema_name: str,
    reasoning_effort: Optional[str] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    candidates = [model.strip() or DEFAULT_MODEL]
    for fallback in MODEL_FALLBACKS:
        if fallback not in candidates:
            candidates.append(fallback)

    last_error: Optional[Exception] = None
    for candidate in candidates:
        request_args: Dict[str, Any] = {
            "model": candidate,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
        }
        if reasoning_effort and reasoning_effort not in {"none", "auto"}:
            request_args["reasoning"] = {"effort": reasoning_effort}

        try:
            emit(progress, f"[INFO] Calling OpenAI model: {candidate}")
            response = client.responses.create(**request_args)
            data = json.loads(response.output_text)
            data["_model_used"] = candidate
            return data
        except Exception as exc:
            last_error = exc
            if "reasoning" in request_args:
                try:
                    request_args.pop("reasoning", None)
                    emit(progress, f"[WARN] Retrying {candidate} without reasoning effort.")
                    response = client.responses.create(**request_args)
                    data = json.loads(response.output_text)
                    data["_model_used"] = candidate
                    data["_reasoning_fallback_used"] = True
                    return data
                except Exception as retry_exc:
                    last_error = retry_exc

            message = str(last_error).lower()
            model_related = any(
                token in message
                for token in [
                    "model",
                    "does not exist",
                    "not found",
                    "unsupported",
                    "invalid",
                    "access",
                    "permission",
                ]
            )
            if candidate != candidates[-1] and model_related:
                emit(progress, f"[WARN] Model {candidate} failed; trying fallback model.")
                continue
            raise last_error

    if last_error:
        raise last_error
    raise RuntimeError("OpenAI request failed before any model was attempted.")


def upload_file(client: Any, path: Path, purpose: str = "user_data") -> Any:
    with path.open("rb") as fh:
        return client.files.create(file=fh, purpose=purpose)


def extract_reconstruction_plan(
    pdf_path: Path,
    out_dir: Path,
    model: str,
    api_key: str,
    reasoning_effort: str,
    prompt: str,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Run: pip install -r requirements.txt")
    client = OpenAI(api_key=api_key)

    emit(progress, "[INFO] Uploading PDF for plan extraction...")
    pdf_file = upload_file(client, pdf_path)

    emit(progress, "[INFO] Extracting reconstruction plan...")
    plan = call_responses_json(
        client=client,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_content=[
            {"type": "input_file", "file_id": pdf_file.id},
            {"type": "input_text", "text": prompt},
        ],
        schema=DESIGN_SCHEMA,
        schema_name="metamaterial_reconstruction_plan",
        reasoning_effort=reasoning_effort,
        progress=progress,
    )
    write_json(out_dir / "reconstruction_plan.json", plan)
    return plan


def require_mesh_packages() -> None:
    missing = []
    if trimesh is None:
        missing.append("trimesh")
    if measure is None or morphology is None:
        missing.append("scikit-image")
    if missing:
        raise RuntimeError(f"Missing package(s): {', '.join(missing)}")


def tpms_field(family: str, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    fam = family.lower()
    if fam in {"primitive", "schwarz_p", "p"}:
        return np.cos(x) + np.cos(y) + np.cos(z)
    if fam == "gyroid":
        return np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)
    if fam == "diamond":
        return (
            np.sin(x) * np.sin(y) * np.sin(z)
            + np.sin(x) * np.cos(y) * np.cos(z)
            + np.cos(x) * np.sin(y) * np.cos(z)
            + np.cos(x) * np.cos(y) * np.sin(z)
        )
    if fam == "iwp":
        return (
            2 * (np.cos(x) * np.cos(y) + np.cos(y) * np.cos(z) + np.cos(z) * np.cos(x))
            - (np.cos(2 * x) + np.cos(2 * y) + np.cos(2 * z))
        )
    raise ValueError(f"Unsupported local TPMS family: {family}")


def mask_to_mesh(mask: np.ndarray, spacing: Tuple[float, float, float], origin=(0.0, 0.0, 0.0)):
    require_mesh_packages()
    if mask.sum() == 0:
        raise RuntimeError("Empty solid mask; cannot generate mesh.")
    vol = np.pad(mask.astype(np.float32), 1, mode="constant", constant_values=0)
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.5, spacing=spacing)
    verts += np.array(origin, dtype=float) - np.array(spacing, dtype=float)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=True)


def _clean_mesh_faces(mesh: Any) -> None:
    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    elif hasattr(mesh, "unique_faces"):
        mesh.update_faces(mesh.unique_faces())
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    elif hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()


def smooth_mesh(mesh: Any, iterations: int = 4, preserve_bounds: bool = True) -> Any:
    require_mesh_packages()
    if iterations <= 0 or len(mesh.vertices) == 0:
        return mesh

    old_bounds = np.asarray(mesh.bounds, dtype=float)
    smoothed = mesh.copy()
    try:
        trimesh.smoothing.filter_taubin(smoothed, lamb=0.5, nu=-0.53, iterations=int(iterations))
    except Exception:
        try:
            trimesh.smoothing.filter_laplacian(smoothed, lamb=0.35, iterations=int(iterations), volume_constraint=True)
        except Exception:
            return mesh

    if preserve_bounds and np.all(np.isfinite(old_bounds)):
        new_bounds = np.asarray(smoothed.bounds, dtype=float)
        old_size = old_bounds[1] - old_bounds[0]
        new_size = new_bounds[1] - new_bounds[0]
        scale = np.divide(old_size, new_size, out=np.ones(3), where=new_size > 1e-9)
        new_center = (new_bounds[0] + new_bounds[1]) / 2.0
        old_center = (old_bounds[0] + old_bounds[1]) / 2.0
        smoothed.vertices = (smoothed.vertices - new_center) * scale + old_center

    _clean_mesh_faces(smoothed)
    smoothed.fix_normals()
    return smoothed


def fit_mesh_to_bounds(mesh: Any, size: Tuple[float, float, float], origin=(0.0, 0.0, 0.0)) -> Any:
    target_size = np.asarray(size, dtype=float)
    target_origin = np.asarray(origin, dtype=float)
    bounds = np.asarray(mesh.bounds, dtype=float)
    current_size = bounds[1] - bounds[0]
    scale = np.divide(target_size, current_size, out=np.ones(3), where=current_size > 1e-9)
    mesh.vertices = (mesh.vertices - bounds[0]) * scale + target_origin
    return mesh


def export_validated_mesh(mesh: Any, out_stl: Path, smooth_iterations: int = 0) -> Dict[str, Any]:
    require_mesh_packages()
    out_stl.parent.mkdir(parents=True, exist_ok=True)
    _clean_mesh_faces(mesh)
    trimesh.repair.fix_normals(mesh)
    if smooth_iterations > 0:
        mesh = smooth_mesh(mesh, iterations=smooth_iterations)
    if not mesh.is_watertight:
        try:
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_normals(mesh)
        except Exception:
            pass

    mesh.export(out_stl)
    loaded = trimesh.load_mesh(out_stl, force="mesh")
    try:
        components = len(loaded.split(only_watertight=False))
    except Exception:
        components = -1
    return {
        "file": str(out_stl),
        "vertices": int(len(loaded.vertices)),
        "faces": int(len(loaded.faces)),
        "extents_mm": [float(v) for v in loaded.extents],
        "watertight": bool(loaded.is_watertight),
        "connected_components": int(components),
        "smoothing_iterations": int(smooth_iterations),
    }


def generate_tpms_local(structure: Dict[str, Any], out_dir: Path, resolution_per_cell: int, max_grid: int) -> Optional[Path]:
    tp = structure["tpms_parameters"]
    family = tp.get("family", "unknown")
    if family in {"unknown", "custom"} or tp.get("requires_mapping"):
        raise RuntimeError(
            "Local TPMS generator only supports primitive, gyroid, diamond, iwp, schwarz_p without mapping. "
            "Use custom builder for custom expressions or mapped TPMS."
        )

    dims = structure["dimensions_mm"]
    ux = (
        get_number(dims.get("unit_cell_size_x"), None)
        or get_number(dims.get("unit_cell_size_y"), None)
        or get_number(dims.get("unit_cell_size_z"), None)
        or 10.0
    )
    uy = get_number(dims.get("unit_cell_size_y"), ux) or ux
    uz = get_number(dims.get("unit_cell_size_z"), ux) or ux

    cx = max(1, int(round(get_number(dims.get("unit_count_x"), 1) or 1)))
    cy = max(1, int(round(get_number(dims.get("unit_count_y"), 1) or 1)))
    cz = max(1, int(round(get_number(dims.get("unit_count_z"), 1) or 1)))

    width = get_number(dims.get("width_x"), cx * ux) or cx * ux
    height = get_number(dims.get("height_y"), cy * uy) or cy * uy
    depth = get_number(dims.get("depth_z"), cz * uz) or cz * uz

    nx = min(max_grid, max(44, cx * resolution_per_cell + 1))
    ny = min(max_grid, max(44, cy * resolution_per_cell + 1))
    nz = min(max_grid, max(44, cz * resolution_per_cell + 1))

    period_x = get_number(tp.get("period_x"), 2 * np.pi) or 2 * np.pi
    period_y = get_number(tp.get("period_y"), 2 * np.pi) or 2 * np.pi
    period_z = get_number(tp.get("period_z"), 2 * np.pi) or 2 * np.pi

    x = np.linspace(0, period_x * cx, nx)
    y = np.linspace(0, period_y * cy, ny)
    z = np.linspace(0, period_z * cz, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    field = tpms_field(family, X, Y, Z)

    low = get_number(tp.get("level_low"), None)
    high = get_number(tp.get("level_high"), None)
    iso = get_number(tp.get("iso_value"), 0.0) or 0.0
    half = abs(get_number(tp.get("sheet_half_thickness"), 0.15) or 0.15)

    if low is not None and high is not None and high > low:
        mask = (field >= low) & (field <= high)
    else:
        mask = np.abs(field - iso) <= half

    fs = get_number(dims.get("face_sheet_thickness"), 0.0) or 0.0
    if fs > 0 and depth > 0:
        z_mm = np.linspace(0, depth, nz)
        sheet = (z_mm <= fs) | (z_mm >= depth - fs)
        mask[:, :, sheet] = True

    spacing = (width / max(1, nx - 1), height / max(1, ny - 1), depth / max(1, nz - 1))
    mesh = mask_to_mesh(mask, spacing)
    mesh = fit_mesh_to_bounds(mesh, (width, height, depth))
    name = clean_name(structure["name"])
    if structure["reconstruction_level"] != "exact_from_equation":
        name += "_approximate"
    out_stl = out_dir / "stl" / f"{name}.stl"
    stats = export_validated_mesh(mesh, out_stl, smooth_iterations=4)

    metadata = {
        "structure_name": structure["name"],
        "generator_type": "tpms_implicit",
        "reconstruction_level": structure["reconstruction_level"],
        "confidence_0_to_1": structure["confidence_0_to_1"],
        "directly_used": structure.get("evidence_sources", []),
        "assumptions": structure.get("assumptions", []),
        "missing_information": structure.get("missing_information", []),
        "tpms_parameters": tp,
        "grid_shape": [int(nx), int(ny), int(nz)],
        "mesh_validation": stats,
    }
    write_json(out_dir / "metadata" / f"{name}.json", metadata)
    return out_stl


def render_pdf_page(pdf_path: Path, page_number: int, dpi: int):
    if fitz is None or Image is None:
        raise RuntimeError("PyMuPDF and Pillow are required for image tracing.")
    doc = fitz.open(str(pdf_path))
    idx = max(0, min(int(page_number) - 1, len(doc) - 1))
    page = doc[idx]
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


def crop_relative(img: Any, box: List[float]):
    w, h = img.size
    x0, y0, x1, y1 = [max(0.0, min(1.0, float(v))) for v in box]
    if x1 <= x0 or y1 <= y0:
        return img
    return img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))


def image_to_mask(img: Any, solid_is_dark: bool, max_pixels: int, min_object_pixels: int) -> np.ndarray:
    if cv2 is None or morphology is None:
        raise RuntimeError("opencv-python and scikit-image are required for image tracing.")

    gray = np.array(img.convert("L"))
    h, w = gray.shape
    scale = min(1.0, float(max_pixels) / float(max(h, w)))
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = th == 0 if solid_is_dark else th == 255
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=max(1, min_object_pixels))
    mask = morphology.remove_small_holes(mask.astype(bool), area_threshold=max(1, min_object_pixels))
    mask = morphology.binary_closing(mask, morphology.disk(1))
    return mask.astype(bool)


def generate_image_trace_local(
    pdf_path: Path,
    structure: Dict[str, Any],
    out_dir: Path,
    dpi: int,
    max_trace_pixels: int,
    z_layers: int,
    min_object_pixels: int,
    save_debug: bool,
    trace_page_override: Optional[int],
    crop_override: Optional[List[float]],
    width_override: Optional[float],
    height_override: Optional[float],
    depth_override: Optional[float],
) -> Optional[Path]:
    fig = structure["figure_trace_info"]
    if not fig.get("needs_figure_tracing", False) and trace_page_override is None:
        raise RuntimeError("Plan does not request figure tracing. Use custom builder or override trace page/crop.")

    page = int(trace_page_override or fig.get("best_page_number") or 1)
    crop = crop_override or fig.get("crop_box_relative") or [0.0, 0.0, 1.0, 1.0]
    solid_is_dark = bool(fig.get("solid_is_dark", True))

    dims = structure["dimensions_mm"]
    width = width_override or get_number(dims.get("width_x"), 50.0) or 50.0
    height = height_override or get_number(dims.get("height_y"), 50.0) or 50.0
    depth = depth_override or get_number(dims.get("depth_z"), 3.0) or 3.0

    page_img = render_pdf_page(pdf_path, page, dpi=dpi)
    crop_img = crop_relative(page_img, crop)
    mask = image_to_mask(crop_img, solid_is_dark, max_trace_pixels, min_object_pixels)

    if save_debug:
        preview_dir = out_dir / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        crop_img.save(preview_dir / f"{clean_name(structure['name'])}_crop.png")
        Image.fromarray((mask.astype(np.uint8) * 255)).save(preview_dir / f"{clean_name(structure['name'])}_mask.png")

    mask_xy = mask.T
    nx, ny = mask_xy.shape
    nz = max(4, int(z_layers))
    vol = np.repeat(mask_xy[:, :, None], nz, axis=2)
    spacing = (width / max(1, nx - 1), height / max(1, ny - 1), depth / max(1, nz - 1))
    mesh = mask_to_mesh(vol, spacing)
    mesh = fit_mesh_to_bounds(mesh, (width, height, depth))

    name = clean_name(structure["name"])
    if structure["reconstruction_level"] != "exact_from_equation":
        name += "_approximate"
    out_stl = out_dir / "stl" / f"{name}.stl"
    stats = export_validated_mesh(mesh, out_stl, smooth_iterations=3)

    metadata = {
        "structure_name": structure["name"],
        "generator_type": "image_trace_extrusion",
        "reconstruction_level": structure["reconstruction_level"],
        "confidence_0_to_1": structure["confidence_0_to_1"],
        "figure_trace_info": {**fig, "used_page": page, "used_crop_box_relative": crop},
        "assumptions": structure.get("assumptions", []),
        "missing_information": structure.get("missing_information", []),
        "dimensions_used_mm": {"width_x": width, "height_y": height, "depth_z": depth},
        "mask_shape": [int(nx), int(ny), int(nz)],
        "mesh_validation": stats,
    }
    write_json(out_dir / "metadata" / f"{name}.json", metadata)
    return out_stl


def run_local_generators(
    pdf_path: Path,
    plan: Dict[str, Any],
    out_dir: Path,
    args: argparse.Namespace,
) -> Tuple[List[Path], List[str]]:
    generated: List[Path] = []
    messages: List[str] = []

    for structure in plan.get("structures", []):
        gen = structure.get("generator_type")
        if gen not in LOCAL_GENERATORS:
            continue
        if structure.get("reconstruction_level") == "insufficient":
            messages.append(f"[SKIP] {structure.get('name')}: insufficient.")
            continue
        try:
            if gen == "tpms_implicit":
                stl = generate_tpms_local(structure, out_dir, args.resolution_per_cell, args.max_grid)
                if stl:
                    generated.append(stl)
                    messages.append(f"[OK] Local TPMS generated: {stl.name}")
            elif gen == "image_trace_extrusion":
                stl = generate_image_trace_local(
                    pdf_path=pdf_path,
                    structure=structure,
                    out_dir=out_dir,
                    dpi=args.dpi,
                    max_trace_pixels=args.max_trace_pixels,
                    z_layers=args.z_layers,
                    min_object_pixels=args.min_object_pixels,
                    save_debug=args.save_debug,
                    trace_page_override=args.trace_page,
                    crop_override=args.crop,
                    width_override=args.width_mm,
                    height_override=args.height_mm,
                    depth_override=args.depth_mm,
                )
                if stl:
                    generated.append(stl)
                    messages.append(f"[OK] Local image trace generated: {stl.name}")
        except Exception as exc:
            messages.append(f"[LOCAL FAILED] {structure.get('name')}: {exc}")

    return generated, messages


def structure_needs_custom_builder(structure: Dict[str, Any]) -> bool:
    gen = structure.get("generator_type")
    if structure.get("reconstruction_level") == "insufficient":
        return False
    if gen in CUSTOM_GENERATORS:
        return True
    if gen == "tpms_implicit":
        tp = structure.get("tpms_parameters", {})
        return tp.get("family") in {"custom", "unknown"} or bool(tp.get("requires_mapping"))
    return False


def plan_needs_custom_builder(plan: Dict[str, Any], force_custom: bool = False) -> bool:
    if force_custom:
        return True
    return any(structure_needs_custom_builder(s) for s in plan.get("structures", []))


def write_temp_plan_for_builder(plan: Dict[str, Any], out_dir: Path) -> Path:
    plan_path = out_dir / "reconstruction_plan_for_builder.json"
    write_json(plan_path, plan)
    return plan_path


def call_custom_builder_model(
    pdf_path: Path,
    plan_path: Path,
    model: str,
    api_key: str,
    reasoning_effort: str,
    repair_log: Optional[str] = None,
    previous_code: Optional[str] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> str:
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Run: pip install -r requirements.txt")
    client = OpenAI(api_key=api_key)

    emit(progress, "[INFO] Uploading files for custom builder...")
    pdf_file = upload_file(client, pdf_path)
    plan_file = upload_file(client, plan_path)

    user_text = CUSTOM_BUILDER_USER_PROMPT
    if repair_log is not None and previous_code is not None:
        user_text = REPAIR_BUILDER_PROMPT_TEMPLATE.format(log=repair_log[-12000:], code=previous_code[-24000:])

    result = call_responses_json(
        client=client,
        model=model,
        system_prompt=CUSTOM_BUILDER_SYSTEM_PROMPT,
        user_content=[
            {"type": "input_file", "file_id": pdf_file.id},
            {"type": "input_file", "file_id": plan_file.id},
            {"type": "input_text", "text": user_text},
        ],
        schema=CUSTOM_BUILDER_SCHEMA,
        schema_name="paper_specific_stl_builder",
        reasoning_effort=reasoning_effort,
        progress=progress,
    )
    return result["builder_code"]


def run_builder_script(
    builder_path: Path,
    pdf_path: Path,
    out_dir: Path,
    timeout_s: int,
) -> Tuple[bool, str, List[Path]]:
    builder_path = builder_path.resolve()
    pdf_path = pdf_path.resolve()
    out_dir = out_dir.resolve()

    helper_src = Path(__file__).with_name("geometry_helpers.py")
    if helper_src.exists():
        shutil.copy2(helper_src, builder_path.parent / "geometry_helpers.py")

    builder_out = out_dir / "custom_builder_output"
    builder_out.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(builder_path), "--pdf", str(pdf_path), "--out", str(builder_out)]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(builder_path.parent),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
        log = proc.stdout
        success_code = proc.returncode == 0
    except subprocess.TimeoutExpired as exc:
        return False, f"Timeout after {timeout_s}s\n{exc}", []
    except Exception:
        return False, traceback.format_exc(), []

    stls = list((builder_out / "stl").glob("*.stl")) if (builder_out / "stl").exists() else []
    success = success_code and len(stls) > 0

    if stls:
        for sub in ["stl", "metadata", "preview"]:
            src = builder_out / sub
            dst = out_dir / sub
            if src.exists():
                dst.mkdir(parents=True, exist_ok=True)
                for p in src.iterdir():
                    if p.is_file():
                        target = dst / p.name
                        if target.exists():
                            target = dst / f"custom_{p.name}"
                        shutil.copy2(p, target)

    if (builder_out / "generation_report.md").exists():
        shutil.copy2(builder_out / "generation_report.md", out_dir / "custom_generation_report.md")

    return success, log, stls


def run_custom_builder_loop(
    pdf_path: Path,
    plan: Dict[str, Any],
    out_dir: Path,
    model: str,
    api_key: str,
    reasoning_effort: str,
    max_repairs: int,
    timeout_s: int,
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple[List[Path], List[str]]:
    generated: List[Path] = []
    messages: List[str] = []
    plan_path = write_temp_plan_for_builder(plan, out_dir)

    previous_code: Optional[str] = None
    repair_log: Optional[str] = None

    for attempt in range(max_repairs + 1):
        emit(progress, f"[INFO] Requesting custom builder attempt {attempt}...")
        code = call_custom_builder_model(
            pdf_path=pdf_path,
            plan_path=plan_path,
            model=model,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
            repair_log=repair_log,
            previous_code=previous_code,
            progress=progress,
        )
        previous_code = code
        builder_path = out_dir / f"generated_builder_attempt_{attempt}.py"
        builder_path.write_text(code, encoding="utf-8")

        emit(progress, f"[INFO] Running custom builder attempt {attempt}...")
        success, log, stls = run_builder_script(builder_path, pdf_path, out_dir, timeout_s)
        (out_dir / f"custom_builder_log_attempt_{attempt}.txt").write_text(log, encoding="utf-8")

        if success:
            copied_stls = sorted((out_dir / "stl").glob("*.stl"))
            generated.extend(copied_stls)
            messages.append(f"[OK] Custom builder generated {len(stls)} STL file(s).")
            return generated, messages

        messages.append(f"[CUSTOM FAILED] Attempt {attempt}: no successful STL. See log.")
        repair_log = log

    return generated, messages


def write_main_report(
    plan: Dict[str, Any],
    out_dir: Path,
    local_messages: List[str],
    custom_messages: List[str],
) -> Path:
    lines: List[str] = []
    lines.append("# Paper-to-STL reconstruction report\n")
    lines.append(f"**Paper:** {plan.get('paper_title', 'Unknown')}")
    if plan.get("doi_or_identifier"):
        lines.append(f"**Identifier:** {plan['doi_or_identifier']}")
    lines.append(f"**Overall confidence:** {plan.get('overall_confidence_0_to_1', 0):.2f}\n")
    lines.append(plan.get("reconstruction_summary", "") + "\n")

    lines.append("## Global assumptions")
    for x in plan.get("global_assumptions", []):
        lines.append(f"- {x}")
    lines.append("\n## Global missing information")
    for x in plan.get("global_missing_information", []):
        lines.append(f"- {x}")

    lines.append("\n## Structures")
    for s in plan.get("structures", []):
        lines.append(f"\n### {s.get('name', 'Unnamed')}")
        lines.append(f"- Type: `{s.get('metamaterial_type', '')}`")
        lines.append(f"- Generator: `{s.get('generator_type', '')}`")
        lines.append(f"- Reconstruction level: `{s.get('reconstruction_level', '')}`")
        lines.append(f"- Confidence: {s.get('confidence_0_to_1', 0):.2f}")
        lines.append("- Evidence sources:")
        for e in s.get("evidence_sources", []):
            lines.append(f"  - {e.get('location','')}: {e.get('evidence_summary','')}")
        lines.append("- Key assumptions:")
        for a in s.get("assumptions", []):
            lines.append(f"  - {a}")
        lines.append("- Missing information:")
        for m in s.get("missing_information", []):
            lines.append(f"  - {m}")

    lines.append("\n## Generation messages")
    for m in local_messages + custom_messages:
        lines.append(f"- {m}")

    stls = sorted((out_dir / "stl").glob("*.stl")) if (out_dir / "stl").exists() else []
    lines.append("\n## STL files")
    if stls:
        for stl in stls:
            lines.append(f"- `{stl.name}`")
    else:
        lines.append("- No STL files generated.")

    lines.append("\n## Mesh quality note")
    lines.append(
        "Local voxel and implicit generators use higher sampling and light Taubin/Laplacian smoothing before export. "
        "The smoothing is intended to reduce marching-cubes faceting while preserving the bounding box and intentional pores."
    )
    lines.append("\n## Accuracy note")
    lines.append(
        "Exact CAD reproduction is only justified for structures labelled exact_from_equation. "
        "All other outputs are evidence-based approximations and should be validated before publication, simulation, or fabrication."
    )

    report_path = out_dir / "reconstruction_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def zip_output(out_dir: Path) -> Path:
    zip_path = out_dir / "paper_to_stl_output_bundle.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in out_dir.rglob("*"):
            if p.is_file() and p.name != zip_path.name:
                z.write(p, arcname=p.relative_to(out_dir))
    return zip_path


def run_pipeline(
    pdf_path: Path,
    out_dir: Path,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    ask_api_key: bool = False,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    prompt: str = DEFAULT_PROMPT,
    prompt_file: Optional[Path] = None,
    plan_json: Optional[Path] = None,
    force_custom_builder: bool = False,
    resolution_per_cell: int = 30,
    max_grid: int = 180,
    trace_page: Optional[int] = None,
    crop: Optional[List[float]] = None,
    width_mm: Optional[float] = None,
    height_mm: Optional[float] = None,
    depth_mm: Optional[float] = None,
    dpi: int = 500,
    max_trace_pixels: int = 900,
    z_layers: int = 30,
    min_object_pixels: int = 80,
    save_debug: bool = False,
    max_repairs: int = 2,
    timeout_s: int = 300,
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, Path]:
    load_local_env(Path(".env"))
    if not pdf_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {pdf_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stl").mkdir(exist_ok=True)
    (out_dir / "metadata").mkdir(exist_ok=True)

    if prompt_file:
        prompt = prompt_file.read_text(encoding="utf-8")

    resolved_key = resolve_api_key(api_key, ask_api_key)

    if plan_json:
        emit(progress, "[INFO] Loading existing reconstruction plan...")
        plan = json.loads(plan_json.read_text(encoding="utf-8"))
        write_json(out_dir / "reconstruction_plan.json", plan)
    else:
        if not resolved_key:
            raise RuntimeError("No OpenAI API key. Use .env, OPENAI_API_KEY, --api-key, or --ask-api-key.")
        plan = extract_reconstruction_plan(
            pdf_path=pdf_path,
            out_dir=out_dir,
            model=model,
            api_key=resolved_key,
            reasoning_effort=reasoning_effort,
            prompt=prompt,
            progress=progress,
        )

    args = argparse.Namespace(
        resolution_per_cell=resolution_per_cell,
        max_grid=max_grid,
        trace_page=trace_page,
        crop=crop,
        width_mm=width_mm,
        height_mm=height_mm,
        depth_mm=depth_mm,
        dpi=dpi,
        max_trace_pixels=max_trace_pixels,
        z_layers=z_layers,
        min_object_pixels=min_object_pixels,
        save_debug=save_debug,
    )
    local_generated, local_messages = run_local_generators(pdf_path, plan, out_dir, args)

    custom_messages: List[str] = []
    if plan_needs_custom_builder(plan, force_custom_builder):
        if not resolved_key:
            custom_messages.append("[SKIP] Custom builder needed, but no API key was provided.")
        else:
            _, custom_messages = run_custom_builder_loop(
                pdf_path=pdf_path,
                plan=plan,
                out_dir=out_dir,
                model=model,
                api_key=resolved_key,
                reasoning_effort=reasoning_effort,
                max_repairs=max_repairs,
                timeout_s=timeout_s,
                progress=progress,
            )

    write_main_report(plan, out_dir, local_messages, custom_messages)
    bundle = zip_output(out_dir)
    success = any((out_dir / "stl").glob("*.stl"))
    return success, bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Universal AI-assisted metamaterial paper-to-STL pipeline.")
    parser.add_argument("pdf", type=Path, help="Input paper PDF")
    parser.add_argument("--out", type=Path, default=Path("paper_to_stl_output"), help="Output directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model")
    parser.add_argument("--api-key", default=None, help="OpenAI API key. Env var or --ask-api-key is safer.")
    parser.add_argument("--ask-api-key", action="store_true", help="Prompt for API key securely.")
    parser.add_argument(
        "--reasoning-effort",
        default=DEFAULT_REASONING_EFFORT,
        choices=["auto", "none", "minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort for models that support it.",
    )
    parser.add_argument("--prompt-file", type=Path, default=None, help="Optional custom extraction prompt file.")
    parser.add_argument("--plan-json", type=Path, default=None, help="Reuse an existing reconstruction_plan.json.")
    parser.add_argument("--force-custom-builder", action="store_true", help="Always ask GPT to write a custom builder.")
    parser.add_argument("--resolution-per-cell", type=int, default=30)
    parser.add_argument("--max-grid", type=int, default=180)
    parser.add_argument("--trace-page", type=int, default=None)
    parser.add_argument("--crop", nargs=4, type=float, default=None, help="Relative crop x0 y0 x1 y1")
    parser.add_argument("--width-mm", type=float, default=None)
    parser.add_argument("--height-mm", type=float, default=None)
    parser.add_argument("--depth-mm", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=500)
    parser.add_argument("--max-trace-pixels", type=int, default=900)
    parser.add_argument("--z-layers", type=int, default=30)
    parser.add_argument("--min-object-pixels", type=int, default=80)
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--timeout-s", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    success, bundle = run_pipeline(
        pdf_path=args.pdf,
        out_dir=args.out,
        model=args.model,
        api_key=args.api_key,
        ask_api_key=args.ask_api_key,
        reasoning_effort=args.reasoning_effort,
        prompt_file=args.prompt_file,
        plan_json=args.plan_json,
        force_custom_builder=args.force_custom_builder,
        resolution_per_cell=args.resolution_per_cell,
        max_grid=args.max_grid,
        trace_page=args.trace_page,
        crop=args.crop,
        width_mm=args.width_mm,
        height_mm=args.height_mm,
        depth_mm=args.depth_mm,
        dpi=args.dpi,
        max_trace_pixels=args.max_trace_pixels,
        z_layers=args.z_layers,
        min_object_pixels=args.min_object_pixels,
        save_debug=args.save_debug,
        max_repairs=args.max_repairs,
        timeout_s=args.timeout_s,
    )

    print("\nDone.")
    print(f"Success: {success}")
    print(f"Output directory: {args.out}")
    print(f"Plan: {args.out / 'reconstruction_plan.json'}")
    print(f"Report: {args.out / 'reconstruction_report.md'}")
    print(f"Bundle: {bundle}")
    stls = sorted((args.out / "stl").glob("*.stl"))
    if stls:
        print("STL files:")
        for stl in stls:
            print(f"  - {stl}")
    else:
        print("No STL files were generated. Check reconstruction_report.md and logs.")


if __name__ == "__main__":
    main()
