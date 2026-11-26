"""Expand SVG <marker> references into explicit shapes before PPTX conversion.

svg2ooxml's native marker handling currently places/rotates arrowheads
incorrectly (they all point in the wrong direction and are duplicated). This
module pre-processes an SVG string: it replaces marker-end/marker-start
attributes with explicitly positioned copies of the marker geometry, so the
converter only has to draw plain paths/shapes.

Key design choice: we *bake* the translate/rotate/translate transform into the
actual vertex coordinates rather than emitting a `transform` attribute.  This
is necessary because svg2ooxml flattens transforms when it groups elements and
only preserves translation, dropping rotations and producing mis-oriented
arrowheads.
"""
from __future__ import annotations

import math
import re
from copy import deepcopy
from typing import Optional
from xml.etree import ElementTree as ET

SVG_NS = "http://www.w3.org/2000/svg"


def _ns_tag(tag: str) -> str:
    return f"{{{SVG_NS}}}{tag}"


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _parse_url_id(value: str) -> Optional[str]:
    """Parse 'url(#arrowTeal)' -> 'arrowTeal'."""
    if not value:
        return None
    m = re.search(r"url\(#([^)]+)\)", value.strip())
    return m.group(1) if m else None


def _get_float(el, attr: str, default: float = 0.0) -> float:
    val = el.get(attr)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _line_endpoint_angle(el, at_start: bool = False) -> tuple[float, float, float]:
    """Return (x, y, angle_deg) for an SVG <line> endpoint.

    If at_start is True, return the start point and the angle from start to end.
    Otherwise return the end point and the same angle.
    """
    x1 = _get_float(el, "x1")
    y1 = _get_float(el, "y1")
    x2 = _get_float(el, "x2")
    y2 = _get_float(el, "y2")
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    if at_start:
        return x1, y1, angle
    return x2, y2, angle


def _path_endpoint_angle(el, at_start: bool = False) -> tuple[float, float, float]:
    """Return (x, y, angle_deg) for an SVG <path> endpoint.

    Uses svgpathtools to evaluate the endpoint derivative if available,
    otherwise falls back to a crude chord approximation of the last segment.
    """
    d = el.get("d", "")
    if not d:
        return 0.0, 0.0, 0.0

    try:
        from svgpathtools import parse_path

        path = parse_path(d)
        if not path:
            raise ValueError("empty path")
        if at_start:
            end = path.point(0.0)
            deriv = path.derivative(0.0)
            if abs(deriv) < 1e-9:
                deriv = path.point(0.001) - path.point(0.0)
            return end.real, end.imag, math.degrees(math.atan2(deriv.imag, deriv.real))
        end = path.point(1.0)
        deriv = path.derivative(1.0)
        if abs(deriv) < 1e-9:
            deriv = path.point(1.0) - path.point(0.999)
        return end.real, end.imag, math.degrees(math.atan2(deriv.imag, deriv.real))
    except Exception:
        pass

    # Fallback: chord of last (or first) M/L/H/V segment
    tokens = re.findall(
        r"[MmLlHhVvCcSsQqTtAaZz]|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?",
        d,
    )
    if not tokens:
        return 0.0, 0.0, 0.0

    x, y = 0.0, 0.0
    cmd = "M"
    prev_ctrl = None
    i = 0
    last_x, last_y = 0.0, 0.0
    first_x, first_y = 0.0, 0.0
    first_set = False
    segments: list[tuple[float, float, float, float]] = []

    def read_n(n: int):
        nonlocal i
        vals = []
        for _ in range(n):
            if i >= len(tokens):
                break
            vals.append(float(tokens[i]))
            i += 1
        return vals

    while i < len(tokens):
        tok = tokens[i]
        i += 1
        if tok.isalpha() or tok in "Zz":
            cmd = tok
        else:
            i -= 1
            vals = read_n(
                {
                    "M": 2,
                    "m": 2,
                    "L": 2,
                    "l": 2,
                    "H": 1,
                    "h": 1,
                    "V": 1,
                    "v": 1,
                    "C": 6,
                    "c": 6,
                    "S": 4,
                    "s": 4,
                    "Q": 4,
                    "q": 4,
                    "T": 2,
                    "t": 2,
                    "A": 7,
                    "a": 7,
                }.get(cmd.upper(), 0)
            )
            if not vals:
                continue

            rel = cmd.islower()
            dx = dy = 0.0
            if rel:
                dx, dy = x, y

            start_x, start_y = x, y
            if cmd.upper() == "M":
                x, y = vals[0] + dx, vals[1] + dy
                if not first_set:
                    first_x, first_y = x, y
                    first_set = True
                last_x, last_y = x, y
            elif cmd.upper() == "L":
                last_x, last_y = x, y
                x, y = vals[0] + dx, vals[1] + dy
                segments.append((start_x, start_y, x, y))
            elif cmd.upper() == "H":
                last_x, last_y = x, y
                x = vals[0] + dx
                segments.append((start_x, start_y, x, y))
            elif cmd.upper() == "V":
                last_x, last_y = x, y
                y = vals[0] + dy
                segments.append((start_x, start_y, x, y))
            elif cmd.upper() == "C":
                last_x, last_y = x, y
                x, y = vals[4] + dx, vals[5] + dy
                prev_ctrl = complex(vals[2] + dx, vals[3] + dy)
                segments.append((start_x, start_y, x, y))
            elif cmd.upper() == "S":
                last_x, last_y = x, y
                x, y = vals[2] + dx, vals[3] + dy
                prev_ctrl = complex(vals[0] + dx, vals[1] + dy)
                segments.append((start_x, start_y, x, y))
            elif cmd.upper() == "Q":
                last_x, last_y = x, y
                x, y = vals[2] + dx, vals[3] + dy
                prev_ctrl = complex(vals[0] + dx, vals[1] + dy)
                segments.append((start_x, start_y, x, y))
            elif cmd.upper() == "T":
                last_x, last_y = x, y
                x, y = vals[0] + dx, vals[1] + dy
                segments.append((start_x, start_y, x, y))
            elif cmd.upper() == "A":
                last_x, last_y = x, y
                x, y = vals[5] + dx, vals[6] + dy
                segments.append((start_x, start_y, x, y))
            elif cmd.upper() == "Z":
                continue

    if at_start and segments:
        x1, y1, x2, y2 = segments[0]
        return x1, y1, math.degrees(math.atan2(y2 - y1, x2 - x1))
    if segments:
        x1, y1, x2, y2 = segments[-1]
        return x2, y2, math.degrees(math.atan2(y2 - y1, x2 - x1))
    return x, y, 0.0


def _transform_point(
    x: float,
    y: float,
    x_end: float,
    y_end: float,
    angle_deg: float,
    ref_x: float,
    ref_y: float,
    scale: float = 1.0,
) -> tuple[float, float]:
    """Apply translate(x_end,y_end) * rotate(angle) * scale * translate(-refX,-refY)."""
    rad = math.radians(angle_deg)
    c = math.cos(rad)
    s = math.sin(rad)
    x0 = (x - ref_x) * scale
    y0 = (y - ref_y) * scale
    xr = c * x0 - s * y0
    yr = s * x0 + c * y0
    return xr + x_end, yr + y_end


def _bake_path_d(
    d: str,
    x_end: float,
    y_end: float,
    angle_deg: float,
    ref_x: float,
    ref_y: float,
    scale: float = 1.0,
) -> str:
    """Bake the marker transform into a path's `d` attribute.

    Uses svgpathtools when available (most accurate). Falls back to a custom
    parser for simple commands if svgpathtools is missing.
    """
    try:
        from svgpathtools import parse_path

        path = parse_path(d)
        if not path:
            return d
        # translate(-refX, -refY); scale; rotate about origin; translate to endpoint
        path = path.translated(complex(-ref_x, -ref_y))
        if abs(scale - 1.0) > 1e-9:
            path = path.scaled(scale, scale)
        path = path.rotated(angle_deg, origin=0)
        path = path.translated(complex(x_end, y_end))
        return path.d()
    except Exception:
        pass

    # Fallback: only handle M, L, H, V, Z with absolute coordinates.
    tokens = re.findall(
        r"[MmLlHhVvZz]|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", d
    )
    out: list[str] = []
    x, y = 0.0, 0.0
    cmd = "M"
    i = 0

    def read_n(n: int):
        nonlocal i
        vals = []
        for _ in range(n):
            if i >= len(tokens):
                break
            vals.append(float(tokens[i]))
            i += 1
        return vals

    while i < len(tokens):
        tok = tokens[i]
        i += 1
        if tok.isalpha():
            cmd = tok
            if cmd.upper() == "Z":
                out.append("Z")
                continue
        else:
            i -= 1
            n = {"M": 2, "L": 2, "H": 1, "V": 1}.get(cmd.upper(), 0)
            vals = read_n(n)
            if not vals:
                i += 1
                continue
            if cmd.upper() == "M":
                x, y = vals[0], vals[1]
                nx, ny = _transform_point(x, y, x_end, y_end, angle_deg, ref_x, ref_y, scale)
                out.append(f"M{nx:.4f},{ny:.4f}")
            elif cmd.upper() == "L":
                x, y = vals[0], vals[1]
                nx, ny = _transform_point(x, y, x_end, y_end, angle_deg, ref_x, ref_y, scale)
                out.append(f"L{nx:.4f},{ny:.4f}")
            elif cmd.upper() == "H":
                x = vals[0]
                nx, ny = _transform_point(x, y, x_end, y_end, angle_deg, ref_x, ref_y, scale)
                out.append(f"L{nx:.4f},{ny:.4f}")
            elif cmd.upper() == "V":
                y = vals[0]
                nx, ny = _transform_point(x, y, x_end, y_end, angle_deg, ref_x, ref_y, scale)
                out.append(f"L{nx:.4f},{ny:.4f}")
    return " ".join(out)


def _bake_polygon_points(
    points_str: str,
    x_end: float,
    y_end: float,
    angle_deg: float,
    ref_x: float,
    ref_y: float,
    close: bool,
    scale: float = 1.0,
) -> str:
    nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", points_str)
    pts = []
    for i in range(0, len(nums), 2):
        if i + 1 >= len(nums):
            break
        x, y = float(nums[i]), float(nums[i + 1])
        nx, ny = _transform_point(x, y, x_end, y_end, angle_deg, ref_x, ref_y, scale)
        pts.append(f"{nx:.4f},{ny:.4f}")
    return " ".join(pts)


def _shape_to_baked_path(
    el: ET.Element,
    x_end: float,
    y_end: float,
    angle_deg: float,
    ref_x: float,
    ref_y: float,
    scale: float = 1.0,
) -> Optional[ET.Element]:
    """Convert a marker child shape to a <path> with baked coordinates."""
    tag = _strip_ns(el.tag)
    new = ET.Element(_ns_tag("path"))

    if tag == "path":
        d = el.get("d", "")
        if not d:
            return None
        new.set("d", _bake_path_d(d, x_end, y_end, angle_deg, ref_x, ref_y, scale))
    elif tag in ("polygon", "polyline"):
        pts = _bake_polygon_points(
            el.get("points", ""), x_end, y_end, angle_deg, ref_x, ref_y, close=tag == "polygon", scale=scale
        )
        if not pts:
            return None
        close_token = " Z" if tag == "polygon" else ""
        new.set("d", f"M {pts.replace(' ', ' L ')}{close_token}")
    elif tag == "rect":
        x = _get_float(el, "x")
        y = _get_float(el, "y")
        w = _get_float(el, "width")
        h = _get_float(el, "height")
        if w <= 0 or h <= 0:
            return None
        corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        baked = [_transform_point(cx, cy, x_end, y_end, angle_deg, ref_x, ref_y, scale) for cx, cy in corners]
        d = "M " + " L ".join(f"{bx:.4f},{by:.4f}" for bx, by in baked) + " Z"
        new.set("d", d)
    elif tag == "circle":
        cx = _get_float(el, "cx")
        cy = _get_float(el, "cy")
        r = _get_float(el, "r")
        if r <= 0:
            return None
        # Approximate circle with a 16-segment polygon
        segs = []
        for k in range(16):
            theta = 2 * math.pi * k / 16
            segs.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
        baked = [_transform_point(px, py, x_end, y_end, angle_deg, ref_x, ref_y, scale) for px, py in segs]
        d = "M " + " L ".join(f"{bx:.4f},{by:.4f}" for bx, by in baked) + " Z"
        new.set("d", d)
    elif tag == "ellipse":
        cx = _get_float(el, "cx")
        cy = _get_float(el, "cy")
        rx = _get_float(el, "rx")
        ry = _get_float(el, "ry")
        if rx <= 0 or ry <= 0:
            return None
        segs = []
        for k in range(16):
            theta = 2 * math.pi * k / 16
            segs.append((cx + rx * math.cos(theta), cy + ry * math.sin(theta)))
        baked = [_transform_point(px, py, x_end, y_end, angle_deg, ref_x, ref_y, scale) for px, py in segs]
        d = "M " + " L ".join(f"{bx:.4f},{by:.4f}" for bx, by in baked) + " Z"
        new.set("d", d)
    elif tag == "line":
        x1 = _get_float(el, "x1")
        y1 = _get_float(el, "y1")
        x2 = _get_float(el, "x2")
        y2 = _get_float(el, "y2")
        p1 = _transform_point(x1, y1, x_end, y_end, angle_deg, ref_x, ref_y, scale)
        p2 = _transform_point(x2, y2, x_end, y_end, angle_deg, ref_x, ref_y, scale)
        new.set("d", f"M{p1[0]:.4f},{p1[1]:.4f} L{p2[0]:.4f},{p2[1]:.4f}")
    else:
        return None

    # Preserve presentation attributes that matter for marker-only arrows.
    for attr in ("fill", "stroke", "opacity", "fill-opacity", "stroke-opacity"):
        val = el.get(attr)
        if val is not None:
            new.set(attr, val)
    # Do not preserve transforms; we have baked them into the geometry.
    return new


def _clone_marker_shapes(
    marker: ET.Element,
    x_end: float,
    y_end: float,
    angle_deg: float,
    stroke_color: Optional[str] = None,
    stroke_width: float = 1.0,
) -> list[ET.Element]:
    """Clone the shapes inside a marker and bake them to the path endpoint.

    The marker's (refX, refY) is placed at the endpoint. The marker is scaled
    according to markerUnits (default "strokeWidth") and rotated so its +X axis
    aligns with the outgoing path direction.
    """
    ref_x = _get_float(marker, "refX", 0.0)
    ref_y = _get_float(marker, "refY", 0.0)
    orient = marker.get("orient", "auto")
    marker_units = marker.get("markerUnits", "strokeWidth")
    scale = 1.0 if marker_units == "userSpaceOnUse" else stroke_width

    if orient == "auto":
        rotation = angle_deg
    else:
        try:
            rotation = float(orient)
        except ValueError:
            rotation = angle_deg

    clones: list[ET.Element] = []
    for child in marker:
        tag = _strip_ns(child.tag)
        if tag not in ("path", "polygon", "polyline", "rect", "circle", "ellipse", "line"):
            continue
        baked = _shape_to_baked_path(child, x_end, y_end, rotation, ref_x, ref_y, scale)
        if baked is None:
            continue
        # If marker shape lacks fill, default to the referencing line's stroke color.
        if baked.get("fill") is None and stroke_color:
            baked.set("fill", stroke_color)
        clones.append(baked)
    return clones


def _resolve_stroke_color(el) -> Optional[str]:
    """Best-effort stroke color extraction from the referencing element."""
    stroke = el.get("stroke")
    if stroke and stroke.lower() != "none":
        return stroke
    style = el.get("style", "")
    m = re.search(r"stroke\s*:\s*([^;]+)", style)
    if m:
        color = m.group(1).strip()
        if color.lower() != "none":
            return color
    return None


def _resolve_stroke_width(el) -> float:
    """Best-effort stroke-width extraction from the referencing element."""
    sw = el.get("stroke-width")
    if sw:
        try:
            return float(sw)
        except ValueError:
            pass
    style = el.get("style", "")
    m = re.search(r"stroke-width\s*:\s*([^;]+)", style)
    if m:
        try:
            return float(m.group(1).strip())
        except ValueError:
            pass
    return 1.0


def expand_markers(svg_content: str) -> str:
    """Expand SVG <marker> references into explicit shapes.

    Returns the modified SVG string. If parsing fails, returns the original.
    """
    try:
        root = ET.fromstring(svg_content)
    except ET.ParseError:
        return svg_content

    # Build marker dictionary
    markers: dict[str, ET.Element] = {}
    defs = root.find(_ns_tag("defs"))
    if defs is not None:
        for child in defs:
            if _strip_ns(child.tag) == "marker" and child.get("id"):
                markers[child.get("id")] = child
    # Also allow markers directly under root (uncommon)
    for child in root:
        if _strip_ns(child.tag) == "marker" and child.get("id"):
            markers[child.get("id")] = child

    if not markers:
        return svg_content

    parent_map = {c: p for p in root.iter() for c in p}

    # Collect operations first to avoid mutating during iteration
    operations: list[tuple[ET.Element, list[ET.Element], str]] = []
    for el in root.iter():
        for attr in ("marker-end", "marker-start"):
            ref = _parse_url_id(el.get(attr, ""))
            if not ref or ref not in markers:
                continue
            marker = markers[ref]
            tag = _strip_ns(el.tag)
            at_start = attr == "marker-start"
            if tag == "line":
                x, y, angle = _line_endpoint_angle(el, at_start=at_start)
            elif tag == "path":
                x, y, angle = _path_endpoint_angle(el, at_start=at_start)
            else:
                continue

            stroke_color = _resolve_stroke_color(el)
            stroke_width = _resolve_stroke_width(el)
            clones = _clone_marker_shapes(marker, x, y, angle, stroke_color, stroke_width)
            operations.append((el, clones, attr))

    # Apply operations
    for el, clones, attr in operations:
        parent = parent_map.get(el)
        if parent is None:
            continue
        idx = list(parent).index(el)
        for i, clone in enumerate(clones):
            parent.insert(idx + 1 + i, clone)
        # Remove the marker attribute so svg2ooxml won't generate its own headEnd
        del el.attrib[attr]

    # Serialize
    ET.register_namespace("", SVG_NS)
    return ET.tostring(root, encoding="unicode")
