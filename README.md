# MassGenQR

MassGenQR is a deliberately simple command-line tool that prints batches of QR
labels for inventory systems. Each label contains only a random alphanumeric ID
(the default length is 12 characters). The generated files share a
timestamp-based base name so it is obvious which PDF and CSV belong together.

The tool focuses on dependable bulk generation of printer-ready sheets:

* **Strict black and white** QR codes sized to fit on A4 paper.
* **Predictable filenames** using the pattern `massgenqr_YYYYMMDD_HHMMSS.*`.
* **Matching CSV** files that list every ID along with its page, row, and
  column position.

## Installation

MassGenQR requires Python 3.10 or newer.

```bash
pip install massgenqr
```

If you are working from a clone of this repository you can install the project
in editable mode:

```bash
pip install -e .
```

## Usage

The command line interface keeps configuration to the essentials. You specify
how many labels you need and MassGenQR handles the layout.

```bash
massgenqr 120
```

Running with `--help` prints the full option reference (mirrored below for
convenience):

```
usage: massgenqr [-h] [--edge-mm EDGE_MM] [--margin-mm MARGIN_MM] [--gutter-mm GUTTER_MM]
                 [--orientation {portrait,landscape}] [--id-length ID_LENGTH] [--alphabet ALPHABET]
                 [--output-dir OUTPUT_DIR]
                 count

Generate printable QR labels on A4 paper along with a CSV index.

positional arguments:
  count                 Number of labels to generate.

options:
  -h, --help            show this help message and exit
  --edge-mm EDGE_MM     Force the edge length of each QR code square in millimetres. Rows and columns are derived
                        automatically.
  --margin-mm MARGIN_MM
                        Override the outer page margin in millimetres (default: 10.0).
  --gutter-mm GUTTER_MM
                        Override the spacing between labels in millimetres (default: 4.0).
  --orientation {portrait,landscape}
                        Page orientation for the PDF (default: portrait).
  --id-length ID_LENGTH
                        Length of the random alphanumeric IDs (default: 12).
  --alphabet ALPHABET   Alphabet to draw random IDs from (default: ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789).
  --output-dir OUTPUT_DIR
                        Directory to write the PDF and CSV files into (default: current directory).
```

### Examples

Generate 200 labels using the default layout:

```bash
massgenqr 200
```

Pick a fixed QR edge length of 28 mm (the tool calculates rows and columns):

```bash
massgenqr 80 --edge-mm 28
```

Switch to landscape orientation with custom margins and gutter:

```bash
massgenqr 120 --orientation landscape --margin-mm 8 --gutter-mm 3
```

## Output files

MassGenQR always writes a matching PDF and CSV using the same timestamp-based
base name inside the chosen output directory (current working directory by
default).

* **PDF** – QR codes arranged in a compact grid on A4 paper. Each cell contains
  a black QR square with the ID centred beneath it in a monospace font. Text is
  automatically scaled to remain readable, and blank space is avoided by
  adjusting the grid to the requested size.
* **CSV** – Contains the generated IDs along with `page`, `row`, and `column`
  values (1-indexed) that describe where the label appears in the PDF. The CSV
  is ready for import into inventory databases or spreadsheets.

If the requested QR size cannot fit within the A4 page while respecting the
margins and gutter, the program stops with a clear error message suggesting you
loosen the constraints.

## Why keep it simple?

MassGenQR intentionally avoids templating systems, metadata, or embedded
payloads. It generates durable identifiers you can attach to anything and then
record elsewhere. Pair the CSV with your inventory system, print the PDF, and
start labelling.
