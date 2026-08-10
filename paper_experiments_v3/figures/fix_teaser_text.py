"""Correct three legacy labels in the vector teaser PDF."""

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "AuthorKit27" / "Figures" / "teaser.pdf"
OUTPUT = ROOT / "AuthorKit27" / "Figures" / "teaser_corrected.pdf"


def overlay(width: float, height: float) -> BytesIO:
    stream = BytesIO()
    pdf = canvas.Canvas(stream, pagesize=(width, height))

    # Bottom-left legend: "begin agents" -> "benign agents".
    pdf.setFillColorRGB(1, 1, 1)
    pdf.rect(203, 188, 49, 13, stroke=0, fill=1)
    pdf.setFillColorRGB(0.88, 0.39, 0.55)
    pdf.setFont("Helvetica-Bold", 6.9)
    pdf.drawString(205.5, 190.5, "benign agents")

    # Panel C stage label.
    pdf.setFillColorRGB(1, 1, 1)
    pdf.rect(451, 169, 34, 16, stroke=0, fill=1)
    pdf.setFillColorRGB(0.08, 0.08, 0.08)
    pdf.setFont("Helvetica", 4.65)
    pdf.drawCentredString(468, 178.2, "Feedback")
    pdf.drawCentredString(468, 172.4, "amplification")

    # Panel C exponent legend; preserve the original zeta glyph and colon.
    pdf.setFillColorRGB(1, 1, 1)
    pdf.rect(570, 178, 59, 11, stroke=0, fill=1)
    pdf.setFillColorRGB(0.08, 0.08, 0.08)
    pdf.setFont("Helvetica", 5.0)
    pdf.drawString(571, 181.1, "feedback amplification")

    pdf.save()
    stream.seek(0)
    return stream


def main() -> None:
    source = PdfReader(str(SOURCE))
    page = source.pages[0]
    width = float(page.mediabox.width)
    height = float(page.mediabox.height)
    page.merge_page(PdfReader(overlay(width, height)).pages[0])

    writer = PdfWriter()
    writer.add_page(page)
    writer._header = b"%PDF-1.5"
    with OUTPUT.open("wb") as handle:
        writer.write(handle)


if __name__ == "__main__":
    main()
