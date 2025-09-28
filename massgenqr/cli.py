"""Command line interface for MassGenQR."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from secrets import choice
from typing import Any, Iterable, List, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imports only for typing
    from reportlab.pdfgen.canvas import Canvas


ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
DEFAULT_ID_LENGTH = 12
DEFAULT_MARGIN_MM = 10.0
DEFAULT_GUTTER_MM = 4.0
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
DEFAULT_TEXT_HEIGHT_MM = 5.0
TARGET_COLUMNS = 6
MAX_COLUMNS = 12
MIN_COLUMNS = 2
MIN_TEXT_HEIGHT_MM = 3.0


@dataclass
class Layout:
    """Grid description for the labels."""

    columns: int
    rows: int
    qr_edge_mm: float
    text_height_mm: float
    margin_mm: float
    gutter_mm: float

    @property
    def per_page(self) -> int:
        return self.columns * self.rows

    @property
    def cell_height_mm(self) -> float:
        return self.qr_edge_mm + self.text_height_mm


class LayoutError(RuntimeError):
    """Raised when the desired layout cannot be produced."""


def mm_to_pt(mm: float) -> float:
    """Convert millimetres to PDF points."""

    return mm * 72.0 / 25.4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="massgenqr",
        description=(
            "Generate printable QR labels on A4 paper along with a CSV index."
        ),
    )
    parser.add_argument(
        "count",
        type=int,
        help="Number of labels to generate.",
    )
    parser.add_argument(
        "--edge-mm",
        type=float,
        default=None,
        help=(
            "Force the edge length of each QR code square in millimetres. Rows and "
            "columns are derived automatically."
        ),
    )
    parser.add_argument(
        "--margin-mm",
        type=float,
        default=DEFAULT_MARGIN_MM,
        help="Override the outer page margin in millimetres (default: %(default)s).",
    )
    parser.add_argument(
        "--gutter-mm",
        type=float,
        default=DEFAULT_GUTTER_MM,
        help="Override the spacing between labels in millimetres (default: %(default)s).",
    )
    parser.add_argument(
        "--orientation",
        choices=("portrait", "landscape"),
        default="portrait",
        help="Page orientation for the PDF (default: %(default)s).",
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


def select_layout(
    *,
    edge_mm: float | None,
    margin_mm: float,
    gutter_mm: float,
    orientation: str,
) -> Layout:
    """Determine an appropriate grid layout."""

    page_width_mm, page_height_mm = (
        (A4_WIDTH_MM, A4_HEIGHT_MM)
        if orientation == "portrait"
        else (A4_HEIGHT_MM, A4_WIDTH_MM)
    )

    available_width = page_width_mm - 2 * margin_mm
    available_height = page_height_mm - 2 * margin_mm
    if available_width <= 0 or available_height <= 0:
        raise LayoutError(
            "Margins leave no usable space on the A4 page. Reduce the margin size."
        )

    if edge_mm is not None:
        qr_edge_mm = edge_mm
        text_height_mm = max(MIN_TEXT_HEIGHT_MM, min(qr_edge_mm * 0.25, 10.0))
        col_count = math.floor((available_width + gutter_mm) / (qr_edge_mm + gutter_mm))
        row_count = math.floor(
            (available_height + gutter_mm)
            / (qr_edge_mm + text_height_mm + gutter_mm)
        )
        if col_count < 1 or row_count < 1:
            raise LayoutError(
                "The requested QR code size does not fit on A4 with the current margins and gutter."
            )
        return Layout(
            columns=col_count,
            rows=row_count,
            qr_edge_mm=qr_edge_mm,
            text_height_mm=text_height_mm,
            margin_mm=margin_mm,
            gutter_mm=gutter_mm,
        )

    best_layout: Layout | None = None
    for columns in range(MIN_COLUMNS, MAX_COLUMNS + 1):
        qr_edge_mm = (available_width + gutter_mm) / columns - gutter_mm
        if qr_edge_mm <= 0:
            continue
        text_height_mm = max(
            MIN_TEXT_HEIGHT_MM, min(qr_edge_mm * 0.25, DEFAULT_TEXT_HEIGHT_MM)
        )
        cell_height = qr_edge_mm + text_height_mm
        rows = math.floor((available_height + gutter_mm) / (cell_height + gutter_mm))
        if rows < 1:
            continue
        layout = Layout(
            columns=columns,
            rows=rows,
            qr_edge_mm=qr_edge_mm,
            text_height_mm=text_height_mm,
            margin_mm=margin_mm,
            gutter_mm=gutter_mm,
        )
        if layout.per_page == 0:
            continue
        if best_layout is None:
            best_layout = layout
            continue
        if layout.per_page > best_layout.per_page:
            best_layout = layout
            continue
        if layout.per_page == best_layout.per_page:
            # Prefer a layout close to the target column count.
            if abs(layout.columns - TARGET_COLUMNS) < abs(
                best_layout.columns - TARGET_COLUMNS
            ):
                best_layout = layout

    if best_layout is None:
        raise LayoutError(
            "Unable to determine a layout. Adjust the margins or gutter to free up more space."
        )
    return best_layout


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


def draw_labels(
    c: "Canvas",
    layout: Layout,
    ids: Sequence[str],
    *,
    orientation: str,
) -> None:
    """Draw the QR labels into the PDF canvas."""

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape

    page_size = landscape(A4) if orientation == "landscape" else A4
    page_width_pt, page_height_pt = page_size
    margin_pt = mm_to_pt(layout.margin_mm)
    gutter_pt = mm_to_pt(layout.gutter_mm)
    qr_size_pt = mm_to_pt(layout.qr_edge_mm)
    text_height_pt = mm_to_pt(layout.text_height_mm)
    cell_height_pt = qr_size_pt + text_height_pt
    cell_width_pt = qr_size_pt

    total_per_page = layout.per_page

    for page_index, page_ids in enumerate(chunked(ids, total_per_page)):
        if page_index:
            c.showPage()
        c.setFillColor(colors.black)
        for idx, identifier in enumerate(page_ids):
            row = idx // layout.columns
            col = idx % layout.columns
            cell_x = margin_pt + col * (cell_width_pt + gutter_pt)
            cell_top = page_height_pt - margin_pt - row * (cell_height_pt + gutter_pt)
            cell_bottom = cell_top - cell_height_pt

            qr_y = cell_bottom + text_height_pt

            qr_image = _build_qr_image(identifier)
            c.drawInlineImage(
                qr_image,
                cell_x,
                qr_y,
                width=qr_size_pt,
                height=qr_size_pt,
            )

            font_size = _fit_font_size(identifier, cell_width_pt, text_height_pt)
            c.setFont("Courier", font_size)
            text_y = cell_bottom + (text_height_pt - font_size) / 2
            c.drawCentredString(cell_x + cell_width_pt / 2, text_y, identifier)


def _build_qr_image(identifier: str) -> Any:
    import qrcode

    qr = qrcode.QRCode(border=1, error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr.add_data(identifier)
    qr.make(fit=True)
    image: PilImage = qr.make_image(fill_color="black", back_color="white")
    return image


def _fit_font_size(text: str, max_width_pt: float, max_height_pt: float) -> float:
    from reportlab.pdfbase import pdfmetrics

    size = min(max_height_pt * 0.9, 20.0)
    while size > 2:
        width = pdfmetrics.stringWidth(text, "Courier", size)
        if width <= max_width_pt:
            return size
        size -= 0.5
    return 2.0


def write_csv(
    path: Path,
    ids: Sequence[str],
    layout: Layout,
) -> None:
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["id", "page", "row", "column"])
        per_page = layout.per_page
        for index, identifier in enumerate(ids):
            page = index // per_page + 1
            within_page = index % per_page
            row = within_page // layout.columns + 1
            column = within_page % layout.columns + 1
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

    try:
        layout = select_layout(
            edge_mm=args.edge_mm,
            margin_mm=args.margin_mm,
            gutter_mm=args.gutter_mm,
            orientation=args.orientation,
        )
    except LayoutError as exc:
        parser.error(str(exc))

    try:
        ids = random_ids(
            args.count, length=args.id_length, alphabet=args.alphabet
        )
    except ValueError as exc:
        parser.error(str(exc))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"massgenqr_{timestamp}"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{base_name}.pdf"
    csv_path = output_dir / f"{base_name}.csv"

    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.pdfgen import canvas

    page_size = landscape(A4) if args.orientation == "landscape" else A4
    c = canvas.Canvas(str(pdf_path), pagesize=page_size)
    draw_labels(c, layout, ids, orientation=args.orientation)
    c.save()

    write_csv(csv_path, ids, layout)

    print(f"Generated {len(ids)} labels")
    print(f"PDF: {pdf_path}")
    print(f"CSV: {csv_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
