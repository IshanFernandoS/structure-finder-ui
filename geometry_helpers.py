"""
geometry_helpers.py

Reusable geometry primitives for GPT-generated metamaterial STL builders.
The generated builder script may import these functions.

Dependencies:
numpy, trimesh, scikit-image, pillow, opencv-python, pymupdf
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import trimesh
from skimage import measure, morphology
import zipfile

try:
    import cv2
    import fitz
    from PIL import Image
    fitz.TOOLS.mupdf_display_errors(False)
    fitz.TOOLS.mupdf_display_warnings(False)
except Exception:
    cv2 = None
    fitz = None
    Image = None


def clean_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
    safe = "_".join(part for part in safe.split("_") if part)
    return safe[:100] or "structure"


def combine_meshes(meshes: Sequence[trimesh.Trimesh]) -> trimesh.Trimesh:
    meshes = [m for m in meshes if m is not None and len(m.vertices) > 0 and len(m.faces) > 0]
    if not meshes:
        raise ValueError("No valid meshes to combine.")
    return trimesh.util.concatenate(meshes)


def _clean_mesh_faces(mesh: trimesh.Trimesh) -> None:
    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    elif hasattr(mesh, "unique_faces"):
        mesh.update_faces(mesh.unique_faces())

    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    elif hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())

    mesh.remove_unreferenced_vertices()

    # Minimal safe cleanup: merge duplicate/near-duplicate vertices.
    # This reduces file size without intentionally simplifying geometry.
    try:
        mesh.merge_vertices()
    except Exception:
        pass


def smooth_mesh(
    mesh: trimesh.Trimesh,
    iterations: int = 3,
    preserve_bounds: bool = True,
) -> trimesh.Trimesh:
    """Lightly smooth faceted voxel/marching-cubes meshes while preserving extents."""
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


def simplify_mesh_for_export(mesh: trimesh.Trimesh, max_faces: int | None = 800_000) -> tuple[trimesh.Trimesh, dict]:
    original_faces = int(len(mesh.faces))
    report = {
        "original_faces_before_export_cap": original_faces,
        "max_export_faces": int(max_faces) if max_faces else None,
        "decimated_for_size": False,
        "decimation_error": "",
    }
    if not max_faces or original_faces <= max_faces:
        return mesh, report

    try:
        reduced = mesh.simplify_quadric_decimation(face_count=int(max_faces), aggression=7)
        if len(reduced.faces) > 0 and len(reduced.vertices) > 0:
            _clean_mesh_faces(reduced)
            reduced.fix_normals()
            report["decimated_for_size"] = True
            return reduced, report
    except Exception as exc:
        report["decimation_error"] = str(exc)

    return mesh, report


def fit_mesh_to_bounds(
    mesh: trimesh.Trimesh,
    size: Tuple[float, float, float],
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> trimesh.Trimesh:
    target_size = np.asarray(size, dtype=float)
    target_origin = np.asarray(origin, dtype=float)
    bounds = np.asarray(mesh.bounds, dtype=float)
    current_size = bounds[1] - bounds[0]
    scale = np.divide(target_size, current_size, out=np.ones(3), where=current_size > 1e-9)
    mesh.vertices = (mesh.vertices - bounds[0]) * scale + target_origin
    return mesh


def mesh_validation_report(mesh: trimesh.Trimesh) -> dict:
    try:
        components = len(mesh.split(only_watertight=False))
    except Exception:
        components = -1
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "extents_mm": [float(v) for v in mesh.extents],
        "watertight": bool(mesh.is_watertight),
        "connected_components": int(components),
    }


def export_mesh(
    mesh: trimesh.Trimesh,
    path: str | Path,
    smooth_iterations: int = 0,
    max_faces: int | None = 800_000,
    zip_output: bool = True,
) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    _clean_mesh_faces(mesh)
    mesh.fix_normals()
    mesh, size_report = simplify_mesh_for_export(mesh, max_faces=max_faces)

    # Keep default smoothing as 0 for paper reconstruction.
    if smooth_iterations > 0:
        mesh = smooth_mesh(mesh, iterations=smooth_iterations)

    if not mesh.is_watertight:
        try:
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_normals(mesh)
        except Exception:
            pass

    # Explicit binary STL export.
    # Binary STL is much smaller than ASCII STL.
    mesh.export(path, file_type="stl")

    loaded = trimesh.load_mesh(path, force="mesh")
    report = mesh_validation_report(loaded)
    report["file"] = str(path)
    report["smoothing_iterations"] = int(smooth_iterations)
    report["stl_size_mb"] = round(path.stat().st_size / (1024 * 1024), 3)
    report.update(size_report)

    # Create compressed zip next to the STL.
    # This does not change geometry at all.
    if zip_output:
        zip_path = path.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(path, arcname=path.name)
        report["zip_file"] = str(zip_path)
        report["zip_size_mb"] = round(zip_path.stat().st_size / (1024 * 1024), 3)

    return report


def box_mesh(size: Tuple[float, float, float], center: Tuple[float, float, float] = (0, 0, 0)) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=size)
    mesh.apply_translation(center)
    return mesh


def cylinder_between(
    p0: Sequence[float],
    p1: Sequence[float],
    radius: float,
    sections: int = 32,
) -> trimesh.Trimesh:
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length <= 1e-9:
        sphere = trimesh.creation.icosphere(radius=radius, subdivisions=2)
        sphere.apply_translation(p0)
        return sphere
    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    direction = vec / length
    transform = trimesh.geometry.align_vectors([0, 0, 1], direction)
    cyl.apply_transform(transform)
    cyl.apply_translation((p0 + p1) / 2.0)
    return cyl


def sphere_at(center: Sequence[float], radius: float, subdivisions: int = 2) -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(radius=radius, subdivisions=subdivisions)
    mesh.apply_translation(center)
    return mesh


def polyline_tube(
    points: Sequence[Sequence[float]],
    radius: float,
    sections: int = 24,
    add_spheres: bool = True,
) -> trimesh.Trimesh:
    pts = [np.asarray(p, dtype=float) for p in points]
    meshes: List[trimesh.Trimesh] = []
    for a, b in zip(pts[:-1], pts[1:]):
        meshes.append(cylinder_between(a, b, radius=radius, sections=sections))
    if add_spheres:
        for p in pts:
            meshes.append(sphere_at(p, radius=radius, subdivisions=2))
    return combine_meshes(meshes)


def strut_lattice_mesh(
    nodes: Sequence[Sequence[float]],
    edges: Sequence[Tuple[int, int]],
    radius: float,
    sections: int = 24,
    add_node_spheres: bool = True,
) -> trimesh.Trimesh:
    meshes: List[trimesh.Trimesh] = []
    for i, j in edges:
        meshes.append(cylinder_between(nodes[i], nodes[j], radius=radius, sections=sections))
    if add_node_spheres:
        used = sorted(set([i for e in edges for i in e]))
        for idx in used:
            meshes.append(sphere_at(nodes[idx], radius=radius, subdivisions=2))
    return combine_meshes(meshes)


def voxel_mask_to_mesh(
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> trimesh.Trimesh:
    if mask.sum() == 0:
        raise ValueError("Empty mask.")
    vol = np.pad(mask.astype(np.float32), 1, mode="constant", constant_values=0)
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.5, spacing=spacing)
    verts += np.asarray(origin, dtype=float) - np.array(spacing)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=True)


def mask_to_mesh(
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> trimesh.Trimesh:
    return voxel_mask_to_mesh(mask, spacing, origin=origin)


def tpms_field(family: str, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    family = family.lower()
    if family in ("primitive", "schwarz_p", "p"):
        return np.cos(x) + np.cos(y) + np.cos(z)
    if family == "gyroid":
        return np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)
    if family == "diamond":
        return (
            np.sin(x) * np.sin(y) * np.sin(z)
            + np.sin(x) * np.cos(y) * np.cos(z)
            + np.cos(x) * np.sin(y) * np.cos(z)
            + np.cos(x) * np.cos(y) * np.sin(z)
        )
    if family == "iwp":
        return (
            2 * (np.cos(x) * np.cos(y) + np.cos(y) * np.cos(z) + np.cos(z) * np.cos(x))
            - (np.cos(2 * x) + np.cos(2 * y) + np.cos(2 * z))
        )
    raise ValueError(f"Unsupported TPMS family: {family}")


def _tpms_family_to_mesh(
    family: str,
    size_mm: Tuple[float, float, float],
    cells: Tuple[int, int, int],
    level_low: float | None = None,
    level_high: float | None = None,
    iso: float = 0.0,
    half_thickness: float = 0.15,
    resolution_per_cell: int = 30,
) -> trimesh.Trimesh:
    nx = max(30, int(cells[0] * resolution_per_cell) + 1)
    ny = max(30, int(cells[1] * resolution_per_cell) + 1)
    nz = max(30, int(cells[2] * resolution_per_cell) + 1)

    x = np.linspace(0, 2 * np.pi * cells[0], nx)
    y = np.linspace(0, 2 * np.pi * cells[1], ny)
    z = np.linspace(0, 2 * np.pi * cells[2], nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    f = tpms_field(family, X, Y, Z)

    if level_low is not None and level_high is not None and level_high > level_low:
        mask = (f >= level_low) & (f <= level_high)
    else:
        mask = np.abs(f - iso) <= half_thickness

    spacing = (
        size_mm[0] / (nx - 1),
        size_mm[1] / (ny - 1),
        size_mm[2] / (nz - 1),
    )
    mesh = voxel_mask_to_mesh(mask, spacing)
    return fit_mesh_to_bounds(mesh, size_mm)


def scalar_field_to_mesh(
    field: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    level: float = 0.0,
    solid_below: bool = True,
) -> trimesh.Trimesh:
    field = np.asarray(field, dtype=np.float32)
    if field.ndim != 3:
        raise ValueError("field must be a 3D numpy array")
    nx, ny, nz = field.shape
    spacing = (
        float(xmax - xmin) / max(1, nx - 1),
        float(ymax - ymin) / max(1, ny - 1),
        float(zmax - zmin) / max(1, nz - 1),
    )
    mask = field <= level if solid_below else field >= level
    mesh = voxel_mask_to_mesh(mask, spacing, origin=(xmin, ymin, zmin))
    return fit_mesh_to_bounds(mesh, (xmax - xmin, ymax - ymin, zmax - zmin), origin=(xmin, ymin, zmin))


def scalar_shell_to_mesh(
    field: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    level: float = 0.0,
    half_thickness: float = 0.15,
) -> trimesh.Trimesh:
    field = np.asarray(field, dtype=np.float32)
    if field.ndim != 3:
        raise ValueError("field must be a 3D numpy array")

    nx, ny, nz = field.shape
    spacing = (
        float(xmax - xmin) / max(1, nx - 1),
        float(ymax - ymin) / max(1, ny - 1),
        float(zmax - zmin) / max(1, nz - 1),
    )
    mask = np.abs(field - level) <= half_thickness
    mesh = voxel_mask_to_mesh(mask, spacing, origin=(xmin, ymin, zmin))
    return fit_mesh_to_bounds(mesh, (xmax - xmin, ymax - ymin, zmax - zmin), origin=(xmin, ymin, zmin))


def tpms_to_mesh(*args, **kwargs) -> trimesh.Trimesh:
    """Create a mesh from either a TPMS family or a precomputed scalar field.

    Supported call styles:
      tpms_to_mesh("gyroid", size_mm=(...), cells=(...), ...)
      tpms_to_mesh(field, xmin, xmax, ymin, ymax, zmin, zmax, level=0.0)
      tpms_to_mesh(field, xmin, xmax, ymin, ymax, zmin, zmax, 0.0)

    The second style is intentionally accepted because generated builders often
    compute paper-specific scalar fields before meshing them.
    """
    if args and isinstance(args[0], np.ndarray):
        field = args[0]
        if len(args) < 7:
            raise TypeError("field-style tpms_to_mesh requires field, xmin, xmax, ymin, ymax, zmin, zmax")
        xmin, xmax, ymin, ymax, zmin, zmax = [float(v) for v in args[1:7]]
        level = kwargs.pop("level", None)
        if level is None:
            level = args[7] if len(args) >= 8 else kwargs.pop("iso", 0.0)
        solid_below = bool(kwargs.pop("solid_below", True))
        return scalar_field_to_mesh(field, xmin, xmax, ymin, ymax, zmin, zmax, float(level), solid_below)

    if args:
        family = args[0]
        size_mm = args[1] if len(args) > 1 else kwargs.pop("size_mm")
        cells = args[2] if len(args) > 2 else kwargs.pop("cells")
    else:
        family = kwargs.pop("family")
        size_mm = kwargs.pop("size_mm")
        cells = kwargs.pop("cells")
    return _tpms_family_to_mesh(
        family=family,
        size_mm=size_mm,
        cells=cells,
        level_low=kwargs.pop("level_low", None),
        level_high=kwargs.pop("level_high", None),
        iso=kwargs.pop("iso", 0.0),
        half_thickness=kwargs.pop("half_thickness", 0.15),
        resolution_per_cell=kwargs.pop("resolution_per_cell", 30),
    )


def render_pdf_page(pdf_path: str | Path, page_number: int, dpi: int = 400):
    if fitz is None or Image is None:
        raise RuntimeError("fitz/PIL unavailable. Install pymupdf pillow.")
    doc = fitz.open(str(pdf_path))
    idx = max(0, min(int(page_number) - 1, len(doc) - 1))
    page = doc[idx]
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


def crop_relative(img, crop_box_rel: Sequence[float]):
    w, h = img.size
    x0, y0, x1, y1 = [float(v) for v in crop_box_rel]
    x0 = max(0, min(1, x0))
    y0 = max(0, min(1, y0))
    x1 = max(0, min(1, x1))
    y1 = max(0, min(1, y1))
    return img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))


def image_to_binary_mask(
    img,
    solid_is_dark: bool = True,
    max_pixels: int = 650,
    min_object_size: int = 80,
) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("cv2 unavailable. Install opencv-python.")
    gray = np.array(img.convert("L"))
    h, w = gray.shape
    scale = min(1.0, max_pixels / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = th == 0 if solid_is_dark else th == 255

    try:
        mask = morphology.remove_small_objects(mask.astype(bool), max_size=min_object_size)
    except TypeError:
        mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_object_size)
    try:
        mask = morphology.remove_small_holes(mask.astype(bool), max_size=min_object_size)
    except TypeError:
        mask = morphology.remove_small_holes(mask.astype(bool), area_threshold=min_object_size)
    if hasattr(morphology, "closing"):
        mask = morphology.closing(mask, morphology.disk(1))
    else:
        mask = morphology.binary_closing(mask, morphology.disk(1))
    return mask.astype(bool)


def extrude_mask_to_mesh(
    mask: np.ndarray,
    width_mm: float,
    height_mm: float,
    depth_mm: float,
    z_layers: int = 16,
) -> trimesh.Trimesh:
    mask_xy = mask.T
    nx, ny = mask_xy.shape
    nz = max(4, int(z_layers))
    vol = np.repeat(mask_xy[:, :, None], nz, axis=2)
    spacing = (
        width_mm / max(1, nx - 1),
        height_mm / max(1, ny - 1),
        depth_mm / max(1, nz - 1),
    )
    mesh = voxel_mask_to_mesh(vol, spacing)
    return fit_mesh_to_bounds(mesh, (width_mm, height_mm, depth_mm))


def arc_points(center, radius, start_angle_deg, end_angle_deg, z=0.0, n=48):
    a0 = math.radians(start_angle_deg)
    a1 = math.radians(end_angle_deg)
    angles = np.linspace(a0, a1, n)
    return [(center[0] + radius * math.cos(a), center[1] + radius * math.sin(a), z) for a in angles]


def write_metadata(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps(data, indent=2), encoding="utf-8")
