from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import trimesh

from paper_to_stl import load_local_env, run_pipeline


APP_DIR = Path(__file__).resolve().parent
RUNS_DIR = APP_DIR / "structure_finder_runs"
MAX_PREVIEW_FACES = 80_000


def mesh_dimensions(stl_path: Path) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray, np.ndarray]:
    mesh = trimesh.load_mesh(stl_path, force="mesh")
    bounds = np.asarray(mesh.bounds)
    mins = bounds[0]
    maxs = bounds[1]
    size = maxs - mins
    return mesh, mins, maxs, size


def mesh_stats(stl_path: Path) -> dict:
    mesh, mins, maxs, size = mesh_dimensions(stl_path)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "bounds_min": [float(v) for v in mins],
        "bounds_max": [float(v) for v in maxs],
        "extents_mm": [float(v) for v in size],
        "file_size_mb": stl_path.stat().st_size / (1024 * 1024),
    }


def sampled_preview_mesh(mesh: trimesh.Trimesh, max_faces: int = MAX_PREVIEW_FACES) -> tuple[trimesh.Trimesh, dict]:
    face_count = int(len(mesh.faces))
    if face_count <= max_faces:
        return mesh, {
            "preview_faces": face_count,
            "original_faces": face_count,
            "simplified": False,
            "method": "full_mesh",
        }

    try:
        simplified = mesh.simplify_quadric_decimation(face_count=max_faces, aggression=5)
        if len(simplified.faces) > 0 and len(simplified.vertices) > 0:
            return simplified, {
                "preview_faces": int(len(simplified.faces)),
                "original_faces": face_count,
                "simplified": True,
                "method": "quadric_decimation",
            }
    except Exception:
        pass

    rng = np.random.default_rng(0)
    selected = np.sort(rng.choice(face_count, size=max_faces, replace=False))
    faces = np.asarray(mesh.faces)[selected]
    unique_vertices, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    preview = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices)[unique_vertices],
        faces=inverse.reshape((-1, 3)),
        process=False,
    )
    return preview, {
        "preview_faces": int(len(preview.faces)),
        "original_faces": face_count,
        "simplified": True,
        "method": "deterministic_face_sampling",
    }


def dimension_trace(start: np.ndarray, end: np.ndarray, label: str, color: str) -> list[go.Scatter3d]:
    mid = (start + end) / 2.0
    return [
        go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode="lines",
            line=dict(color=color, width=7),
            hoverinfo="skip",
            showlegend=False,
        ),
        go.Scatter3d(
            x=[mid[0]],
            y=[mid[1]],
            z=[mid[2]],
            mode="text",
            text=[label],
            textfont=dict(color=color, size=15),
            hoverinfo="skip",
            showlegend=False,
        ),
    ]


def bounding_box_traces(mins: np.ndarray, maxs: np.ndarray) -> list[go.Scatter3d]:
    x0, y0, z0 = mins
    x1, y1, z1 = maxs
    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    traces = []
    for a, b in edges:
        traces.append(
            go.Scatter3d(
                x=[corners[a, 0], corners[b, 0]],
                y=[corners[a, 1], corners[b, 1]],
                z=[corners[a, 2], corners[b, 2]],
                mode="lines",
                line=dict(color="#222222", width=2),
                opacity=0.45,
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return traces


def stl_figure(stl_path: Path, show_measurements: bool = False) -> tuple[go.Figure, dict]:
    mesh, mins, maxs, size = mesh_dimensions(stl_path)
    preview_mesh, preview_info = sampled_preview_mesh(mesh)
    vertices = np.asarray(preview_mesh.vertices)
    faces = np.asarray(preview_mesh.faces)
    pad = max(float(size.max()) * 0.12, 1.0)
    dim_origin = np.array([mins[0], mins[1] - pad, mins[2] - pad])
    x_end = dim_origin + np.array([size[0], 0, 0])
    y_start = np.array([maxs[0] + pad, mins[1], mins[2] - pad])
    y_end = y_start + np.array([0, size[1], 0])
    z_start = np.array([maxs[0] + pad, maxs[1] + pad, mins[2]])
    z_end = z_start + np.array([0, 0, size[2]])

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="#00a6d6",
                opacity=1.0,
                flatshading=False,
                lighting=dict(ambient=0.55, diffuse=0.8, specular=0.35, roughness=0.55, fresnel=0.1),
                lightposition=dict(x=120, y=180, z=220),
                hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>",
            )
        ]
    )
    if show_measurements:
        for trace in bounding_box_traces(mins, maxs):
            fig.add_trace(trace)
        for trace in dimension_trace(dim_origin, x_end, f"X {size[0]:.2f} mm", "#d00000"):
            fig.add_trace(trace)
        for trace in dimension_trace(y_start, y_end, f"Y {size[1]:.2f} mm", "#2b9348"):
            fig.add_trace(trace)
        for trace in dimension_trace(z_start, z_end, f"Z {size[2]:.2f} mm", "#5a189a"):
            fig.add_trace(trace)

    fig.update_layout(
        height=640,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="X mm", backgroundcolor="#f4f6f8", gridcolor="#d8dee4", zerolinecolor="#9aa4ad"),
            yaxis=dict(title="Y mm", backgroundcolor="#f4f6f8", gridcolor="#d8dee4", zerolinecolor="#9aa4ad"),
            zaxis=dict(title="Z mm", backgroundcolor="#f4f6f8", gridcolor="#d8dee4", zerolinecolor="#9aa4ad"),
            bgcolor="#ffffff",
            camera=dict(eye=dict(x=1.45, y=1.55, z=1.15)),
        ),
        paper_bgcolor="#ffffff",
    )
    return fig, preview_info


def download_button(path: Path, label: str, mime: str = "application/octet-stream") -> None:
    if path.exists():
        st.download_button(label, data=path.read_bytes(), file_name=path.name, mime=mime)


def file_size_label(path: Path) -> str:
    size_mb = path.stat().st_size / (1024 * 1024)
    return f"{path.name} ({size_mb:.1f} MB)"


def latest_json(path: Path) -> dict | None:
    plan_path = path / "reconstruction_plan.json"
    if plan_path.exists():
        return json.loads(plan_path.read_text(encoding="utf-8"))
    candidates = sorted(path.glob("model_plan_attempt_*.json"))
    if not candidates:
        return None
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


def main() -> None:
    st.set_page_config(page_title="Structure Finder", page_icon="STL", layout="wide")
    load_local_env(APP_DIR / ".env")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    st.title("Structure Finder")

    with st.sidebar:
        st.header("Input")
        uploaded_pdf = st.file_uploader("Paper PDF", type=["pdf"])
        model = st.text_input("OpenAI model", value="gpt-5.5")
        reasoning_effort = st.selectbox("Reasoning effort", ["none", "low", "medium", "high"], index=3)
        max_repairs = st.number_input("Repair attempts", min_value=0, max_value=5, value=2, step=1)
        timeout_s = st.number_input("Builder timeout seconds", min_value=30, max_value=1800, value=300, step=30)
        run = st.button("Generate STL", type="primary", use_container_width=True)

    if "last_run_dir" not in st.session_state:
        st.session_state.last_run_dir = None

    st.caption(
        "The pipeline extracts a reconstruction plan, uses local generators where appropriate, asks GPT for custom builders "
        "when needed, repairs failures, then previews any STL files produced."
    )

    if uploaded_pdf is None:
        st.info("Upload a PDF paper to generate a paper-specific STL builder.")
        return

    if run:
        run_id = uuid.uuid4().hex[:10]
        out_dir = RUNS_DIR / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = out_dir / uploaded_pdf.name
        pdf_path.write_bytes(uploaded_pdf.getvalue())

        messages: list[str] = []

        def progress(message: str) -> None:
            messages.append(message)
            st.write(message)

        with st.status("Running Structure Finder...", expanded=True) as status:
            try:
                success, bundle = run_pipeline(
                    pdf_path=pdf_path,
                    out_dir=out_dir,
                    model=model.strip() or "gpt-5.5",
                    reasoning_effort=reasoning_effort,
                    max_repairs=int(max_repairs),
                    timeout_s=int(timeout_s),
                    progress=progress,
                )
                st.session_state.last_run_dir = str(out_dir)
                if success:
                    status.update(label="Finished: STL generated", state="complete")
                else:
                    status.update(label="Finished: no successful STL", state="error")
                st.write(f"Bundle: {bundle}")
            except Exception as exc:
                error_log = out_dir / "app_error.txt"
                import traceback

                error_log.write_text(traceback.format_exc(), encoding="utf-8")
                status.update(label="Failed", state="error")
                st.error(str(exc))
                st.caption(f"Full error log: {error_log}")
                return

    if st.session_state.last_run_dir is None:
        return

    out_dir = Path(st.session_state.last_run_dir)
    stl_paths = sorted((out_dir / "stl").glob("*.stl"))
    metadata_paths = sorted((out_dir / "metadata").glob("*.json"))
    builder_paths = sorted(out_dir.glob("generated_builder_attempt_*.py"))
    log_paths = sorted(out_dir.glob("custom_builder_log_attempt_*.txt")) or sorted(out_dir.glob("builder_log_attempt_*.txt"))
    bundle_path = out_dir / "paper_to_stl_output_bundle.zip"
    if not bundle_path.exists():
        bundle_path = out_dir / "output_bundle.zip"
    report_path = out_dir / "reconstruction_report.md"
    if not report_path.exists():
        report_path = out_dir / "design_report.md"
    plan = latest_json(out_dir)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Outputs")
        download_button(bundle_path, "Download output bundle", "application/zip")
        download_button(report_path, "Download reconstruction_report.md", "text/markdown")
        if stl_paths:
            with st.expander("Generated STL files"):
                for stl_path in stl_paths:
                    st.write(file_size_label(stl_path))

        if report_path.exists():
            with st.expander("Design report", expanded=True):
                st.markdown(report_path.read_text(encoding="utf-8"))

        if plan:
            with st.expander("GPT builder plan"):
                st.json({k: v for k, v in plan.items() if k != "builder_code"})

        if metadata_paths:
            with st.expander("STL metadata"):
                for path in metadata_paths:
                    st.write(path.name)
                    st.json(json.loads(path.read_text(encoding="utf-8")))

        if builder_paths:
            with st.expander("Generated builder code"):
                selected_builder = st.selectbox("Builder attempt", builder_paths, format_func=lambda p: p.name)
                st.code(selected_builder.read_text(encoding="utf-8"), language="python")

        if log_paths:
            with st.expander("Builder logs"):
                selected_log = st.selectbox("Log attempt", log_paths, format_func=lambda p: p.name)
                st.text(selected_log.read_text(encoding="utf-8")[-12000:])

    with right:
        st.subheader("STL preview")
        if not stl_paths:
            st.warning("No STL file was generated. Check the design report and builder logs.")
            return

        selected_stl = st.selectbox("STL file", stl_paths, format_func=lambda p: p.name)
        stats = mesh_stats(selected_stl)
        if not stats["watertight"]:
            st.warning("This STL is not watertight. Check metadata and report before fabrication.")
        with st.expander("Mesh diagnostics"):
            st.json(stats)
        download_button(selected_stl, f"Download selected STL ({file_size_label(selected_stl)})", "model/stl")
        fig, preview_info = stl_figure(selected_stl, show_measurements=True)
        if preview_info["simplified"]:
            st.info(
                f"Preview reduced from {preview_info['original_faces']:,} to "
                f"{preview_info['preview_faces']:,} faces. The downloaded STL is unchanged."
            )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
