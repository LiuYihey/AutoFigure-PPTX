"""Bridge script: convert SVG string to native DrawingML PPTX via svg2ooxml.

Called by server.py through subprocess (C:\\Python313\\python.exe) because
svg2ooxml requires Python >=3.13 (uses PEP 695 `type` statement), while the
main figure conda env is Python 3.10.

Usage:
    python svg2pptx_bridge.py <output_pptx_path>
    # reads SVG string from stdin, writes PPTX to <output_pptx_path>

Exit codes:
    0 = success
    1 = import error (svg2ooxml not installed)
    2 = conversion error
"""
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from pptx_text_merger import merge_stacked_texts
from svg_marker_expander import expand_markers


# svg2ooxml 的输出缺少根级 _rels/.rels，PowerPoint 因此无法识别包入口。
# 这是 OOXML 规范要求的最外层关系文件，必须存在并指向 ppt/presentation.xml。
# 这里定义标准内容，在转换后注入到 zip 中。
_ROOT_RELS_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" '
    'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
    'Target="ppt/presentation.xml"/>'
    '<Relationship Id="rId2" '
    'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" '
    'Target="docProps/core.xml"/>'
    '<Relationship Id="rId3" '
    'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" '
    'Target="docProps/app.xml"/>'
    '</Relationships>'
)


def _merge_stacked_texts_in_pptx(out_path: str) -> None:
    """Merge adjacent single-line text boxes into multi-paragraph shapes.

    svg2ooxml emits one text shape per SVG <text> element.  For multi-line
    labels this creates many small top-aligned boxes.  PowerPoint can render
    them crowded or stuck to the top of their containers.  We post-process
    the slide XML to collapse vertically-stacked text shapes that share the
    same horizontal centre and formatting into one shape with multiple
    paragraphs.
    """
    p = Path(out_path)
    if not p.is_file():
        return

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".pptx", dir=str(p.parent))
    os.close(tmp_fd)
    try:
        with zipfile.ZipFile(p, "r") as zin, \
             zipfile.ZipFile(tmp_name, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "ppt/slides/slide1.xml":
                    data = merge_stacked_texts(data.decode("utf-8")).encode("utf-8")
                zout.writestr(item, data)
        shutil.move(tmp_name, out_path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def _patch_package_relations(out_path: str) -> None:
    """Ensure the PPTX package has the required root-level _rels/.rels part.

    svg2ooxml (as of current version) writes presentation.xml and all internal
    parts but forgets the package-level relationship part that tells PowerPoint
    "this is a presentation, start here".  Without it PowerPoint refuses to
    open the file with an unrecoverable read error.  We post-process the zip
    to inject it.
    """
    p = Path(out_path)
    if not p.is_file():
        return

    needs_patch = False
    with zipfile.ZipFile(p, "r") as zin:
        names = zin.namelist()
        if "_rels/.rels" not in names:
            needs_patch = True

    if not needs_patch:
        return

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".pptx", dir=str(p.parent))
    os.close(tmp_fd)
    try:
        with zipfile.ZipFile(p, "r") as zin, \
             zipfile.ZipFile(tmp_name, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                # Skip any malformed/empty _rels entries just in case
                if item.filename == "_rels/.rels":
                    continue
                zout.writestr(item, zin.read(item.filename))
            # Write the missing root rels as the last entry
            zout.writestr("_rels/.rels", _ROOT_RELS_XML)
        shutil.move(tmp_name, out_path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def main() -> int:
    if len(sys.argv) < 2:
        print("ERROR: missing output path argument", file=sys.stderr)
        return 2

    out_path = sys.argv[1]
    svg_content = sys.stdin.read()

    if not svg_content.strip():
        print("ERROR: empty SVG input on stdin", file=sys.stderr)
        return 2

    try:
        from svg2ooxml.public import SvgToPptxExporter
    except ImportError as e:
        print(f"ERROR: svg2ooxml not installed for this Python. Run: "
              f"\"{sys.executable}\" -m pip install svg2ooxml. ({e})", file=sys.stderr)
        return 1

    try:
        # Pre-process: expand SVG <marker> references into explicit shapes.
        # svg2ooxml's marker handling rotates/places arrowheads incorrectly;
        # expanding them removes the ambiguity and guarantees fidelity.
        processed_svg = expand_markers(svg_content)

        exporter = SvgToPptxExporter()
        exporter.convert_string(processed_svg, out_path)
        # Post-process: inject missing root _rels/.rels so PowerPoint can open it
        _patch_package_relations(out_path)
        # Post-process: merge stacked single-line text boxes so multi-line text
        # blocks become real multi-paragraph shapes.  This avoids PowerPoint
        # rendering issues where many small top-aligned boxes crowd together.
        _merge_stacked_texts_in_pptx(out_path)
        print(f"OK: {out_path} ({os.path.getsize(out_path)} bytes)")
        return 0
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR: conversion failed: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
