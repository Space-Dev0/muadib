#!/usr/bin/env python3
"""Convert a compiled PDF presentation to a PPTX presentation by rendering pages as images.

This preserves LaTeX fonts, TikZ diagrams, math formulas, and Beamer styling with 100% fidelity.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def render_pdf_pdftoppm(pdf_path: Path, output_dir: Path, dpi: int = 200) -> list[Path]:
    """Render PDF pages to PNG using pdftoppm (poppler-utils)."""
    prefix = output_dir / "page"
    cmd = [
        "pdftoppm",
        "-png",
        "-r",
        str(dpi),
        str(pdf_path),
        str(prefix),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"pdftoppm failed: {result.stderr}")

    images = sorted(output_dir.glob("page-*.png"))
    return images


def render_pdf_fallback(pdf_path: Path, output_dir: Path, dpi: int = 200) -> list[Path]:
    """Fallback rendering using pypdfium2 or pdf2image if pdftoppm is unavailable."""
    try:
        import pypdfium2 as pdfium  # type: ignore

        pdf = pdfium.PdfDocument(str(pdf_path))
        images = []
        for i, page in enumerate(pdf):
            image = page.render(scale=dpi / 72).to_pil()
            img_path = output_dir / f"page-{i + 1:03d}.png"
            image.save(img_path)
            images.append(img_path)
        return images
    except ImportError:
        pass

    try:
        from pdf2image import convert_from_path  # type: ignore

        pil_images = convert_from_path(str(pdf_path), dpi=dpi)
        images = []
        for i, image in enumerate(pil_images):
            img_path = output_dir / f"page-{i + 1:03d}.png"
            image.save(img_path)
            images.append(img_path)
        return images
    except ImportError:
        pass

    raise RuntimeError(
        "No PDF renderer found. Please install 'poppler-utils' (pdftoppm) or 'pypdfium2' / 'pdf2image'."
    )


def render_pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = 200) -> list[Path]:
    """Render PDF pages to PNG files."""
    if shutil.which("pdftoppm"):
        return render_pdf_pdftoppm(pdf_path, output_dir, dpi)
    return render_pdf_fallback(pdf_path, output_dir, dpi)


def images_to_pptx(images: list[Path], output_pptx: Path) -> None:
    """Pack images into a PPTX presentation matching aspect ratio."""
    try:
        from PIL import Image
        from pptx import Presentation
        from pptx.util import Inches
    except ImportError as e:
        raise RuntimeError(
            "Missing dependencies for PPTX generation. Install python-pptx and pillow: "
            "pip install python-pptx pillow"
        ) from e

    if not images:
        raise ValueError("No slide images to convert.")

    # Inspect the first image for aspect ratio
    with Image.open(images[0]) as first_img:
        img_w, img_h = first_img.size
        aspect_ratio = img_w / img_h

    prs = Presentation()

    # Determine slide dimensions
    if abs(aspect_ratio - (16 / 9)) < 0.05:
        # Standard widescreen 16:9
        prs.slide_width = Inches(13.333333)
        prs.slide_height = Inches(7.5)
    elif abs(aspect_ratio - (4 / 3)) < 0.05:
        # Standard 4:3
        prs.slide_width = Inches(10.0)
        prs.slide_height = Inches(7.5)
    else:
        prs.slide_width = Inches(13.333333)
        prs.slide_height = Inches(13.333333 / aspect_ratio)

    blank_slide_layout = prs.slide_layouts[6]  # Blank slide

    for img_path in images:
        slide = prs.slides.add_slide(blank_slide_layout)
        slide.shapes.add_picture(
            str(img_path),
            0,
            0,
            width=prs.slide_width,
            height=prs.slide_height,
        )

    output_pptx.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(output_pptx))
    print(f"Successfully generated: {output_pptx} ({len(images)} slides)")


def main():
    parser = argparse.ArgumentParser(description="Convert PDF presentation to PPTX")
    parser.add_argument("pdf_path", type=Path, help="Path to input PDF file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to output PPTX file (defaults to replacing .pdf with .pptx)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Resolution for page rendering (default: 200 DPI)",
    )
    args = parser.parse_args()

    pdf_path = args.pdf_path.resolve()
    if not pdf_path.is_file():
        print(f"Error: PDF file '{pdf_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    output_pptx = args.output.resolve() if args.output else pdf_path.with_suffix(".pptx")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        print(f"Rendering '{pdf_path.name}' at {args.dpi} DPI...")
        images = render_pdf_to_images(pdf_path, temp_dir_path, dpi=args.dpi)
        print(f"Rendered {len(images)} slides. Creating PPTX...")
        images_to_pptx(images, output_pptx)


if __name__ == "__main__":
    main()
