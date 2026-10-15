from pathlib import Path
import fitz  # PyMuPDF

def detect_font_style(font_name: str) -> dict:
    """
    Detect basic font style from the PDF font name.

    """
    if not font_name:
        return {"is_bold": False, "is_italic": False}

    name = font_name.lower()

    is_bold = any(marker in name for marker in [
        "bold",
        "black",
        "heavy"
    ])

    is_italic = any(marker in name for marker in [
        "italic",
        "oblique",
        "kursiv",
        "slanted",
        "-it"
    ])

    return {
        "is_bold": is_bold,
        "is_italic": is_italic
    }


def wrap_text_with_style(text: str, is_bold: bool, is_italic: bool) -> str:
    """
    Wrap text in inline style tags for the marked TXT output.
    """
    if not text:
        return text

    if is_bold and is_italic:
        return f"<b><i>{text}</i></b>"
    elif is_bold:
        return f"<b>{text}</b>"
    elif is_italic:
        return f"<i>{text}</i>"
    else:
        return text
    

def extract_pdf_structure(pdf_path: Path) -> list[dict]:
    """
    Extract the PDF into a structured Python object:
    page -> blocks -> lines -> spans
    """
    doc = fitz.open(pdf_path)
    all_pages = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_dict = page.get_text("dict", sort=True)

        page_record = {
            "page_number": page_index + 1,
            "width": page.rect.width,
            "height": page.rect.height,
            "blocks": []
        }

        for block in page_dict.get("blocks", []):
            # text block
            if block.get("type") != 0:
                continue

            block_record = {
                "bbox": block.get("bbox"),
                "lines": []
            }

            for line in block.get("lines", []):
                line_record = {
                    "bbox": line.get("bbox"),
                    "dir": line.get("dir"),
                    "spans": []
                }

                for span in line.get("spans", []):
                    text = span.get("text", "")
                    font = span.get("font", "")
                    style = detect_font_style(font)

                    span_record = {
                        "text": text,
                        "bbox": span.get("bbox"),
                        "size": span.get("size"),
                        "font": font,
                        "flags": span.get("flags"),
                        "color": span.get("color"),
                        "is_bold": style["is_bold"],
                        "is_italic": style["is_italic"],
                    }

                    line_record["spans"].append(span_record)

                block_record["lines"].append(line_record)

            page_record["blocks"].append(block_record)

        all_pages.append(page_record)

    doc.close()
    return all_pages


def line_to_marked_text(line_record: dict) -> str:
    """
    Convert one extracted line into a style-aware text line.
    """
    parts = []

    for span in line_record.get("spans", []):
        text = span.get("text", "")
        is_bold = span.get("is_bold", False)
        is_italic = span.get("is_italic", False)

        marked = wrap_text_with_style(text, is_bold, is_italic)
        parts.append(marked)

    return "".join(parts).strip()


def pages_to_marked_text(all_pages: list[dict]) -> str:
    """
    Convert the structured extraction into one marked TXT string.

    """
    output_lines = []

    for page in all_pages:
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                line_text = line_to_marked_text(line)

                # keep only non-empty lines
                if line_text.strip():
                    output_lines.append(line_text)

    return "\n".join(output_lines)


def process_one_pdf(pdf_path: Path, txt_path: Path):

    print(f"Processing: {pdf_path.name}")

    all_pages = extract_pdf_structure(pdf_path)

    base_name = pdf_path.stem

    # Save marked TXT
    marked_text = pages_to_marked_text(all_pages)
    marked_txt_path = txt_path / f"{base_name}.txt"
    marked_txt_path.write_text(marked_text, encoding="utf-8")

    print(f"  TXT saved to:  {marked_txt_path}")

    print()
    return all_pages

