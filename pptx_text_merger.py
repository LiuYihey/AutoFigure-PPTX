"""Post-process PPTX slide XML to fix text layout.

svg2ooxml converts each SVG <text> element into a separate PowerPoint text
shape with anchor="t" and wrap="none".  In PowerPoint this produces many
small top-aligned boxes that can look crowded or stick to the top of their
containers.  This module:

1. Merges only tightly-stacked single-line text shapes that share the same
   background container (same horizontal centre, same font size/colour,
   small vertical gap, AND enclosed by the same background rectangle) into
   one multi-paragraph text shape so line spacing is handled by PowerPoint.
2. Sets the vertical anchor to "ctr" for all text shapes so text is centred
   in its box rather than glued to the top.
"""
from __future__ import annotations

import copy
from collections import defaultdict
from typing import Optional
from xml.etree import ElementTree as ET

_NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}

# OOXML uses English Metric Units; 914400 EMU == 1 inch.
_EMU_PER_INCH = 914400

# Tolerances for deciding that two text shapes belong to the same column.
_X_TOLERANCE_EMU = 120000  # ~0.13 inch
_FONT_SIZE_TOLERANCE = 50  # 1/100 pt
_MAX_MERGE_GAP_EMU = 164592  # ~0.18 inch: only merge truly adjacent lines


def _tag(local: str, ns: str = "p") -> str:
    return f"{{{_NS[ns]}}}{local}"


def _int_emu(el, attr: str) -> int:
    val = el.get(attr)
    return int(val) if val is not None else 0


def _get_text_content(sp: ET.Element) -> str:
    parts = []
    for t in sp.iter(_tag("t", "a")):
        parts.append(t.text or "")
    return "".join(parts)


def _get_paragraph_run_props(sp: ET.Element) -> Optional[ET.Element]:
    """Return the first <a:rPr> inside the shape's txBody, if any."""
    tx_body = sp.find(_tag("txBody"))
    if tx_body is None:
        return None
    return tx_body.find(f".//{_tag('rPr', 'a')}")


def _font_size(sp: ET.Element) -> int:
    rpr = _get_paragraph_run_props(sp)
    if rpr is not None:
        sz = rpr.get("sz")
        if sz is not None:
            try:
                return int(sz)
            except ValueError:
                pass
    return 0


def _fill_ref(sp: ET.Element) -> str:
    """Simple signature of the text fill colour to keep matching shapes together."""
    rpr = _get_paragraph_run_props(sp)
    if rpr is None:
        return ""
    fill = rpr.find(_tag("solidFill", "a"))
    if fill is None:
        return ""
    srgb = fill.find(_tag("srgbClr", "a"))
    if srgb is not None:
        return srgb.get("val") or ""
    scheme = fill.find(_tag("schemeClr", "a"))
    if scheme is not None:
        return scheme.get("val") or ""
    return ""


def _shape_position(sp: ET.Element) -> tuple[int, int, int, int]:
    """Return (x, y, w, h) in EMU for a shape."""
    sp_pr = sp.find(_tag("spPr"))
    if sp_pr is None:
        sp_pr = sp
    xf = sp_pr.find(f".//{_tag('xfrm', 'a')}")
    if xf is None:
        return 0, 0, 0, 0
    off = xf.find(_tag("off", "a"))
    ext = xf.find(_tag("ext", "a"))
    if off is None or ext is None:
        return 0, 0, 0, 0
    return (
        _int_emu(off, "x"),
        _int_emu(off, "y"),
        _int_emu(ext, "cx"),
        _int_emu(ext, "cy"),
    )


def _collect_containers(root: ET.Element) -> list[tuple[int, int, int, int]]:
    """Collect background shapes (non-text) that could contain text shapes.

    Returns a list of (x0, y0, x1, y1) tuples in EMU.
    """
    containers: list[tuple[int, int, int, int]] = []
    for sp in root.iter(_tag("sp")):
        tx_body = sp.find(_tag("txBody"))
        if tx_body is not None:
            text = _get_text_content(sp).strip()
            if text:
                continue  # This is a text shape, not a background
        x, y, w, h = _shape_position(sp)
        if w > 0 and h > 0:
            containers.append((x, y, x + w, y + h))
    return containers


def _find_container(
    cx: int, cy: int,
    containers: list[tuple[int, int, int, int]],
) -> int:
    """Return the index of the smallest container enclosing (cx, cy).

    Returns -1 if the point is not inside any container.
    """
    best = -1
    best_area = float("inf")
    for i, (x0, y0, x1, y1) in enumerate(containers):
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            area = (x1 - x0) * (y1 - y0)
            if area < best_area:
                best_area = area
                best = i
    return best


def _paragraph_xml_for_text(text: str, align: str = "ctr") -> str:
    """Build a minimal OOXML paragraph containing one run with the given text."""
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return (
        f'<a:p xmlns:a="{_NS["a"]}" xmlns:p="{_NS["p"]}">'
        f'<a:pPr algn="{align}"/>'
        f'<a:r><a:t>{escaped}</a:t></a:r>'
        f'<a:endParaRPr/>'
        f'</a:p>'
    )


def _set_shape_text(tx_body: ET.Element, paragraphs: list[str], align: str = "ctr") -> None:
    """Replace all paragraphs in txBody with new ones built from plain text lines."""
    for child in list(tx_body):
        if child.tag == _tag("p", "a"):
            tx_body.remove(child)
    for para_text in paragraphs:
        para = ET.fromstring(_paragraph_xml_for_text(para_text, align))
        tx_body.append(para)


def _fix_text_anchor(sp: ET.Element) -> None:
    """Set vertical anchor to centre for a text shape and allow wrapping."""
    body_pr = sp.find(f".//{_tag('bodyPr', 'a')}")
    if body_pr is not None:
        body_pr.set("anchor", "ctr")
        body_pr.set("anchorCtr", "1")
        body_pr.set("wrap", "square")


def _make_merged_shape(shapes: list[ET.Element]) -> ET.Element:
    """Create a new shape that contains the text of all input shapes stacked."""
    template = shapes[0]
    merged = copy.deepcopy(template)

    xs, ys, ws, hs = [], [], [], []
    for sp in shapes:
        x, y, w, h = _shape_position(sp)
        xs.append(x)
        ys.append(y)
        ws.append(w)
        hs.append(h)

    min_y = min(ys)
    max_w = max(ws)
    max_bottom = max(y + h for y, h in zip(ys, hs))
    raw_h = max_bottom - min_y
    # Add extra line spacing (20%) so multi-paragraph text isn't cramped.
    new_h = int(raw_h * 1.25)

    # Center the merged box horizontally on the original column centre.
    centre_xs = [x + w // 2 for x, w in zip(xs, ws)]
    centre_x = sum(centre_xs) // len(centre_xs)
    new_x = centre_x - max_w // 2

    # Keep the vertical centre unchanged after growing the box.
    centre_y = (min_y + max_bottom) // 2
    new_y = centre_y - new_h // 2

    # Update transform.
    xf = merged.find(f".//{_tag('xfrm', 'a')}")
    if xf is not None:
        off = xf.find(_tag("off", "a"))
        ext = xf.find(_tag("ext", "a"))
        if off is not None:
            off.set("x", str(new_x))
            off.set("y", str(new_y))
        if ext is not None:
            ext.set("cx", str(max_w))
            ext.set("cy", str(new_h))

    _fix_text_anchor(merged)

    tx_body = merged.find(_tag("txBody"))
    if tx_body is not None:
        # Join lines with a space so words don't run together.
        texts = []
        for sp in shapes:
            txt = _get_text_content(sp).strip()
            if txt and not txt.endswith((" ", "\n")):
                txt += " "
            texts.append(txt)
        _set_shape_text(tx_body, texts, align="ctr")

    cnvpr = merged.find(f".//{_tag('cNvPr')}")
    if cnvpr is not None:
        cnvpr.set("name", "Merged Text")

    return merged


def merge_stacked_texts(slide_xml: str) -> str:
    """Post-process slide XML to improve text layout in PowerPoint."""
    root = ET.fromstring(slide_xml)

    # Collect background containers so we can group text by its enclosing box.
    containers = _collect_containers(root)

    text_shapes: list[tuple[ET.Element, int, int, int, int, int, str, int]] = []
    for sp in root.iter(_tag("sp")):
        tx_body = sp.find(_tag("txBody"))
        if tx_body is None:
            continue
        text = _get_text_content(sp).strip()
        if not text:
            continue
        x, y, w, h = _shape_position(sp)
        cx = x + w // 2
        cy = y + h // 2
        c_idx = _find_container(cx, cy, containers)
        text_shapes.append((sp, x, y, w, h, _font_size(sp), _fill_ref(sp), c_idx))

    if not text_shapes:
        return slide_xml

    # Group by container, x-center, font size, and colour.
    # c_idx == -1 means "not in any container"; free texts can still merge
    # with each other if they share the same x-centre / font / colour.
    groups: dict[tuple[int, int, int, str], list[tuple[ET.Element, int, int, int, int]]] = defaultdict(list)
    for sp, x, y, w, h, sz, fill, c_idx in text_shapes:
        centre = x + w // 2
        centre_bucket = round(centre / _X_TOLERANCE_EMU)
        sz_bucket = round(sz / _FONT_SIZE_TOLERANCE)
        key = (c_idx, centre_bucket, sz_bucket, fill)
        groups[key].append((sp, x, y, w, h))

    to_remove: set[ET.Element] = set()
    to_insert: list[tuple[ET.Element, ET.Element]] = []

    for group in groups.values():
        if len(group) <= 1:
            continue
        group.sort(key=lambda item: item[2])  # sort by y

        run: list[ET.Element] = [group[0][0]]
        prev_bottom = group[0][2] + group[0][4]
        for sp, x, y, w, h in group[1:]:
            gap = y - prev_bottom
            # Only merge when the boxes are vertically adjacent (small gap or slight overlap).
            if -h <= gap <= _MAX_MERGE_GAP_EMU:
                run.append(sp)
                prev_bottom = y + h
            else:
                if len(run) > 1:
                    merged = _make_merged_shape(run)
                    to_insert.append((run[0], merged))
                    to_remove.update(run)
                run = [sp]
                prev_bottom = y + h

        if len(run) > 1:
            merged = _make_merged_shape(run)
            to_insert.append((run[0], merged))
            to_remove.update(run)

    # Apply anchor fix to shapes that are not being removed (merged ones are already fixed).
    for sp, *_rest in text_shapes:
        if sp not in to_remove:
            _fix_text_anchor(sp)

    if to_insert:
        parent_map = {c: p for p in root.iter() for c in p}
        for original, merged in to_insert:
            parent = parent_map.get(original)
            if parent is None:
                continue
            idx = list(parent).index(original)
            parent.insert(idx, merged)

        for sp in list(to_remove):
            parent = parent_map.get(sp)
            if parent is not None and sp in parent:
                parent.remove(sp)

    # Final pass: vertically center text shapes inside their background containers.
    # svg2ooxml positions text boxes based on the SVG baseline, which may not
    # match the container's vertical centre.  We re-center each text shape that
    # is the ONLY text shape in its container (so we don't disturb multi-line
    # layouts where several text boxes share one container).
    _center_texts_in_containers(root, containers)

    return ET.tostring(root, encoding="unicode")


def _leaf_containers(
    containers: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Return only containers that don't fully enclose any other container.

    This excludes large section boxes (e.g. the outer "Sense" frame) so that
    title text sitting at the top of a section box is not incorrectly pulled
    to the vertical centre of the entire section.
    """
    leaves: list[tuple[int, int, int, int]] = []
    n = len(containers)
    for i in range(n):
        ix0, iy0, ix1, iy1 = containers[i]
        is_leaf = True
        for j in range(n):
            if i == j:
                continue
            jx0, jy0, jx1, jy1 = containers[j]
            if ix0 <= jx0 and jx1 <= ix1 and iy0 <= jy0 and jy1 <= iy1:
                is_leaf = False
                break
        if is_leaf:
            leaves.append(containers[i])
    return leaves


def _center_texts_in_containers(
    root: ET.Element,
    containers: list[tuple[int, int, int, int]],
) -> None:
    """Vertically center text shapes that are alone in their background container.

    Only **leaf** containers (those that don't enclose other containers) are
    considered, so section-title text is left in place.
    """
    leaf_containers = _leaf_containers(containers)

    # Count how many text shapes fall in each leaf container.
    container_text_count: dict[int, int] = defaultdict(int)
    remaining: list[tuple[ET.Element, int, int, int, int, int]] = []

    for sp in root.iter(_tag("sp")):
        tx_body = sp.find(_tag("txBody"))
        if tx_body is None:
            continue
        text = _get_text_content(sp).strip()
        if not text:
            continue
        x, y, w, h = _shape_position(sp)
        cx = x + w // 2
        cy = y + h // 2
        c_idx = _find_container(cx, cy, leaf_containers)
        if c_idx >= 0:
            container_text_count[c_idx] += 1
        remaining.append((sp, y, h, c_idx))

    for sp, y, h, c_idx in remaining:
        if c_idx < 0:
            continue
        # Only center when this text shape is the sole occupant of its container.
        if container_text_count[c_idx] > 1:
            continue
        cx0, cy0, _cx1, cy1 = leaf_containers[c_idx]
        container_h = cy1 - cy0
        if container_h <= 0:
            continue
        new_y = cy0 + (container_h - h) // 2
        if abs(new_y - y) < 10000:  # < ~1 px, skip negligible shifts
            continue
        xf = sp.find(f".//{_tag('xfrm', 'a')}")
        if xf is not None:
            off = xf.find(_tag("off", "a"))
            if off is not None:
                off.set("y", str(new_y))
