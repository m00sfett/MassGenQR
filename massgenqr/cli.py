"""Command line interface for MassGenQR with robust PDF layout support."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from secrets import choice
from typing import Iterable, List, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imports only for typing
    from PIL import Image
    from reportlab.pdfgen.canvas import Canvas


ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
DEFAULT_ID_LENGTH = 12

DEFAULT_MARGIN_MM = 12.0
DEFAULT_GAP_MM = 2.0
DEFAULT_MIN_QR_MM = 22.0
DEFAULT_LABEL_FONT_PT = 8.0
DEFAULT_LABEL_HEIGHT_MM = 5.0

PAPER_SIZES_MM: dict[str, Tuple[float, float]] = {
    "A4": (210.0, 297.0),
    "Letter": (215.9, 279.4),
}


class LayoutError(RuntimeError):
    """Raised when the desired layout cannot be produced."""


def mm_to_pt(mm: float) -> float:
    """Convert millimetres to PDF points."""

    return mm * 72.0 / 25.4


def pt_to_mm(pt: float) -> float:
    """Convert PDF points to millimetres."""

    return pt * 25.4 / 72.0


@dataclass
class GenerationConfig:
    """User supplied configuration for PDF generation."""

    paper: str
    orientation: str
    margin_mm: float
    gap_mm: float
    min_qr_mm: float
    label_font_pt: float
    label_height_mm: float
    rows: int | None
    cols: int | None
    max_per_page: int | None
    error_correction: str
    quiet_zone_modules: int
    dry_run: bool


@dataclass
class PageLayout:
    """Derived measurements for laying out QR codes on a PDF page."""

    page_width_pt: float
    page_height_pt: float
    margin_pt: float
    gap_pt: float
    label_height_pt: float
    label_font_pt: float
    qr_size_pt: float
    rows: int
    cols: int
    per_page: int

    @property
    def capacity(self) -> int:
        return self.rows * self.cols

    @property
    def qr_size_mm(self) -> float:
        return pt_to_mm(self.qr_size_pt)


@dataclass
class CellPlacement:
    """Placement information for an individual QR code."""

    data: str
    row: int
    column: int


@dataclass
class Page:
    """A single PDF page of QR codes."""

    index: int
    placements: List[CellPlacement]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="massgenqr",
        description="Generate printable QR labels with automatic pagination.",
    )
    parser.add_argument("count", type=int, help="Number of labels to generate.")
    parser.add_argument(
        "--paper",
        choices=sorted(PAPER_SIZES_MM),
        default="A4",
        help="Paper size for the PDF (default: %(default)s).",
    )
    parser.add_argument(
        "--orientation",
        choices=("portrait", "landscape"),
        default="portrait",
        help="Page orientation for the PDF (default: %(default)s).",
    )
    parser.add_argument(
        "--margin-mm",
        type=float,
        default=DEFAULT_MARGIN_MM,
        help="Outer page margin in millimetres (default: %(default)s).",
    )
    parser.add_argument(
        "--gap-mm",
        type=float,
        default=DEFAULT_GAP_MM,
        help="Spacing between QR cells in millimetres (default: %(default)s).",
    )
    parser.add_argument(
        "--min-qr-mm",
        type=float,
        default=DEFAULT_MIN_QR_MM,
        help="Minimum QR code size in millimetres (default: %(default)s).",
    )
    parser.add_argument(
        "--label-font-pt",
        type=float,
        default=DEFAULT_LABEL_FONT_PT,
        help="Font size for QR labels in points (default: %(default)s).",
    )
    parser.add_argument(
        "--label-height-mm",
        type=float,
        default=DEFAULT_LABEL_HEIGHT_MM,
        help="Reserved height for QR labels in millimetres (default: %(default)s).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Force the number of rows per page.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Force the number of columns per page.",
    )
    parser.add_argument(
        "--max-per-page",
        type=int,
        default=None,
        help="Maximum number of QR codes per page.",
    )
    parser.add_argument(
        "--error-correction",
        choices=("L", "M", "Q", "H"),
        default="H",
        help="QR error correction level (default: %(default)s).",
    )
    parser.add_argument(
        "--quiet-zone-modules",
        type=int,
        default=4,
        help="Quiet zone size in QR modules (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print layout information without generating output files.",
    )
    parser.add_argument(
        "--id-length",
        type=int,
        default=DEFAULT_ID_LENGTH,
        help="Length of the random alphanumeric IDs (default: %(default)s).",
    )
    parser.add_argument(
        "--alphabet",
        default=ALPHABET,
        help="Alphabet to draw random IDs from (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory to write the PDF and CSV files into (default: current directory).",
    )
    return parser


def compute_layout(cfg: GenerationConfig) -> PageLayout:
    """Compute layout information for the provided configuration."""

    try:
        page_width_mm, page_height_mm = PAPER_SIZES_MM[cfg.paper]
    except KeyError as exc:  # pragma: no cover - guarded by argparse choices
        raise LayoutError(f"Unsupported paper size: {cfg.paper}") from exc

    if cfg.orientation == "landscape":
        page_width_mm, page_height_mm = page_height_mm, page_width_mm

    margin_pt = mm_to_pt(cfg.margin_mm)
    gap_pt = mm_to_pt(cfg.gap_mm)
    label_height_pt = mm_to_pt(cfg.label_height_mm)
    min_qr_pt = mm_to_pt(cfg.min_qr_mm)

    page_width_pt = mm_to_pt(page_width_mm)
    page_height_pt = mm_to_pt(page_height_mm)

    available_width = page_width_pt - 2 * margin_pt
    available_height = page_height_pt - 2 * margin_pt
    if available_width <= 0 or available_height <= 0:
        raise LayoutError(
            "Margins leave no usable space on the page. Reduce the margin size."
        )

    cell_min_width = min_qr_pt + gap_pt
    cell_min_height = min_qr_pt + label_height_pt + gap_pt

    cols = cfg.cols
    rows = cfg.rows

    if cols is None:
        cols = max(1, int(math.floor((available_width + gap_pt) / cell_min_width)))
    if rows is None:
        rows = max(1, int(math.floor((available_height + gap_pt) / cell_min_height)))

    if cols < 1 or rows < 1:
        raise LayoutError("At least one row and column are required for layout.")

    cell_width = (available_width - (cols - 1) * gap_pt) / cols
    cell_height = (available_height - (rows - 1) * gap_pt) / rows

    if cell_height <= label_height_pt:
        raise LayoutError(
            "Label height leaves no room for QR codes. Reduce the label height or increase spacing."
        )

    qr_size_pt = min(cell_width, cell_height - label_height_pt)

    if qr_size_pt < min_qr_pt:
        raise LayoutError(
            "Requested layout results in QR size {actual:.2f}mm which is below the minimum {minimum:.2f}mm."
            .format(actual=pt_to_mm(qr_size_pt), minimum=cfg.min_qr_mm)
        )

    capacity = rows * cols
    if capacity <= 0:
        raise LayoutError("Layout produced zero capacity per page.")

    per_page = capacity
    if cfg.max_per_page is not None:
        if cfg.max_per_page <= 0:
            raise LayoutError("max-per-page must be greater than zero if provided.")
        per_page = min(per_page, cfg.max_per_page)
        if per_page <= 0:
            raise LayoutError("max-per-page leaves no slots available on each page.")

    return PageLayout(
        page_width_pt=page_width_pt,
        page_height_pt=page_height_pt,
        margin_pt=margin_pt,
        gap_pt=gap_pt,
        label_height_pt=label_height_pt,
        label_font_pt=cfg.label_font_pt,
        qr_size_pt=qr_size_pt,
        rows=rows,
        cols=cols,
        per_page=per_page,
    )


def layout_pages(codes: Sequence[str], layout: PageLayout) -> List[Page]:
    """Arrange codes into pages according to the provided layout."""

    pages: List[Page] = []
    for page_index, chunk in enumerate(chunked(codes, layout.per_page), start=1):
        placements = [
            CellPlacement(data=code, row=idx // layout.cols, column=idx % layout.cols)
            for idx, code in enumerate(chunk)
        ]
        pages.append(Page(index=page_index, placements=placements))
    return pages


def build_qr_image(data: str, layout: PageLayout, cfg: GenerationConfig) -> "Image":
    """Create a high resolution QR code image for the provided data."""

    import qrcode
    from PIL import Image

    error_levels = {
        "L": qrcode.constants.ERROR_CORRECT_L,
        "M": qrcode.constants.ERROR_CORRECT_M,
        "Q": qrcode.constants.ERROR_CORRECT_Q,
        "H": qrcode.constants.ERROR_CORRECT_H,
    }

    qr = qrcode.QRCode(
        border=cfg.quiet_zone_modules,
        error_correction=error_levels[cfg.error_correction],
        box_size=1,
    )
    qr.add_data(data)
    qr.make(fit=True)

    image = qr.make_image(fill_color="black", back_color="white").convert("RGB")

    target_px = max(1, math.ceil(layout.qr_size_pt / 72.0 * 300))
    if image.size[0] != target_px:
        image = image.resize((target_px, target_px), Image.NEAREST)
    return image


def draw_pdf(
    canvas: "Canvas",
    pages: Sequence[Page],
    layout: PageLayout,
    cfg: GenerationConfig,
) -> None:
    """Render the QR codes and labels into the PDF canvas."""

    from reportlab.lib import colors

    total_pages = len(pages)
    min_font_pt = max(2.0, cfg.label_font_pt - 2.0)

    for page_offset, page in enumerate(pages):
        if page_offset:
            canvas.showPage()
        canvas.setFillColor(colors.black)

        placement_map = {(p.row, p.column): p for p in page.placements}
        for row in range(layout.rows):
            for col in range(layout.cols):
                placement = placement_map.get((row, col))
                if placement is None:
                    continue
                x0 = layout.margin_pt + col * (layout.qr_size_pt + layout.gap_pt)
                y0 = layout.margin_pt + (layout.rows - 1 - row) * (
                    layout.qr_size_pt + layout.label_height_pt + layout.gap_pt
                )

                qr_image = build_qr_image(placement.data, layout, cfg)
                canvas.drawInlineImage(
                    qr_image,
                    x0,
                    y0 + layout.label_height_pt,
                    width=layout.qr_size_pt,
                    height=layout.qr_size_pt,
                )

                text, font_size = _fit_label(
                    placement.data,
                    layout.qr_size_pt,
                    layout.label_height_pt,
                    cfg.label_font_pt,
                    min_font_pt,
                )
                canvas.setFont("Courier", font_size)
                text_y = y0 + (layout.label_height_pt - font_size) / 2
                canvas.drawCentredString(x0 + layout.qr_size_pt / 2, text_y, text)

        if total_pages > 1:
            page_number_font = max(6.0, min_font_pt)
            canvas.setFont("Courier", page_number_font)
            footer_y = layout.margin_pt / 2
            canvas.drawCentredString(
                layout.page_width_pt / 2,
                footer_y,
                f"Page {page.index} of {total_pages}",
            )


def _fit_label(
    text: str,
    max_width_pt: float,
    label_height_pt: float,
    initial_font_pt: float,
    min_font_pt: float,
) -> Tuple[str, float]:
    """Determine the label text and font size that fits inside the available space."""

    from reportlab.pdfbase import pdfmetrics

    font_size = max(min_font_pt, min(initial_font_pt, label_height_pt))
    while True:
        width = pdfmetrics.stringWidth(text, "Courier", font_size)
        if width <= max_width_pt:
            return text, font_size
        if font_size <= min_font_pt:
            break
        font_size = max(min_font_pt, font_size - 0.5)

    ellipsis = "…"
    truncated = text
    while truncated:
        candidate = truncated + ellipsis
        width = pdfmetrics.stringWidth(candidate, "Courier", font_size)
        if width <= max_width_pt:
            return candidate, font_size
        truncated = truncated[:-1]
    return ellipsis, font_size


def random_ids(count: int, *, length: int, alphabet: Sequence[str] | str) -> List[str]:
    """Generate a list of unique random identifiers."""

    if count <= 0:
        raise ValueError("count must be positive")
    if length <= 0:
        raise ValueError("id length must be positive")
    seen = set()
    ids: List[str] = []
    alphabet_list = list(alphabet)
    if not alphabet_list:
        raise ValueError("alphabet must not be empty")
    max_identifiers = len(alphabet_list) ** length
    if count > max_identifiers:
        raise ValueError(
            "Cannot generate {count} unique identifiers from an alphabet of size {alphabet_size} "
            "with identifier length {length}. Maximum distinct identifiers: {max_identifiers}."
            .format(
                count=count,
                alphabet_size=len(alphabet_list),
                length=length,
                max_identifiers=max_identifiers,
            )
        )
    while len(ids) < count:
        identifier = "".join(choice(alphabet_list) for _ in range(length))
        if identifier in seen:
            continue
        seen.add(identifier)
        ids.append(identifier)
    return ids


def chunked(iterable: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for index in range(0, len(iterable), size):
        yield iterable[index : index + size]


def write_csv(path: Path, ids: Sequence[str], layout: PageLayout) -> None:
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["id", "page", "row", "column"])
        per_page = layout.per_page
        for index, identifier in enumerate(ids):
            page = index // per_page + 1
            within_page = index % per_page
            row = within_page // layout.cols + 1
            column = within_page % layout.cols + 1
            writer.writerow([identifier, page, row, column])


def ensure_dependencies() -> None:
    try:
        from reportlab.pdfbase import pdfmetrics
    except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
        raise RuntimeError(
            "ReportLab is required. Install MassGenQR with its optional dependencies."
        ) from exc

    try:
        pdfmetrics.getFont("Courier")
    except KeyError as exc:  # pragma: no cover - Courier should always exist
        raise RuntimeError("Courier font is required for output") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        ensure_dependencies()
    except RuntimeError as exc:
        parser.error(str(exc))

    cfg = GenerationConfig(
        paper=args.paper,
        orientation=args.orientation,
        margin_mm=args.margin_mm,
        gap_mm=args.gap_mm,
        min_qr_mm=args.min_qr_mm,
        label_font_pt=args.label_font_pt,
        label_height_mm=args.label_height_mm,
        rows=args.rows,
        cols=args.cols,
        max_per_page=args.max_per_page,
        error_correction=args.error_correction,
        quiet_zone_modules=args.quiet_zone_modules,
        dry_run=args.dry_run,
    )

    try:
        layout = compute_layout(cfg)
    except LayoutError as exc:
        parser.error(str(exc))

    if cfg.dry_run:
        print(
            "rows={rows} cols={cols} per_page={per_page} qr_size_mm={qr:.2f}".format(
                rows=layout.rows,
                cols=layout.cols,
                per_page=layout.per_page,
                qr=layout.qr_size_mm,
            )
        )
        return 0

    try:
        ids = random_ids(args.count, length=args.id_length, alphabet=args.alphabet)
    except ValueError as exc:
        parser.error(str(exc))

    pages = layout_pages(ids, layout)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"massgenqr_{timestamp}"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{base_name}.pdf"
    csv_path = output_dir / f"{base_name}.csv"

    from reportlab.pdfgen import canvas

    pdf_canvas = canvas.Canvas(
        str(pdf_path),
        pagesize=(layout.page_width_pt, layout.page_height_pt),
    )
    draw_pdf(pdf_canvas, pages, layout, cfg)
    pdf_canvas.save()

    write_csv(csv_path, ids, layout)

    print(f"Generated {len(ids)} labels")
    print(f"PDF: {pdf_path}")
    print(f"CSV: {csv_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
