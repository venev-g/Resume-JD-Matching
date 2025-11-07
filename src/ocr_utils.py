import os
import io
import logging
import pytesseract
from PIL import Image, UnidentifiedImageError
import fitz  
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    pytesseract.get_tesseract_version()
    logger.info("OCR Utils: Using Tesseract from system PATH.")
except pytesseract.TesseractNotFoundError:
    logger.warning(
        "OCR Utils: Warning - Tesseract executable not found in system PATH. "
        "OCR operations may fail unless Tesseract is installed correctly."
    )
except Exception as e:
    logger.error(f"OCR Utils: Error checking for Tesseract in PATH: {e}")


class OCRError(Exception):
    """Custom exception for OCR failures."""
    pass


def _perform_image_ocr(image_path, lang="eng"):
    """Internal function for image OCR."""
    try:
        img = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(img, lang=lang)
        if not extracted_text.strip():
            logger.warning(f"No text detected in image '{os.path.basename(image_path)}'.")
        return extracted_text
    except FileNotFoundError:
        raise OCRError(f"Image file not found: '{image_path}'")
    except UnidentifiedImageError:
        raise OCRError(f"Cannot identify image file format: '{image_path}'. Is it a valid image?")
    except pytesseract.TesseractNotFoundError:
        raise OCRError(
            "Tesseract executable not found. Ensure it's installed and in PATH."
        )
    except pytesseract.TesseractError as e:
        err_msg = f"Tesseract error processing image '{os.path.basename(image_path)}': {e}"
        if "Failed loading language" in str(e):
            err_msg += f" - Hint: Ensure the language pack for '{lang}' is installed."
        raise OCRError(err_msg)
    except Exception as e:
        raise OCRError(f"Unexpected error processing image '{os.path.basename(image_path)}': {e}")


def _perform_pdf_ocr(pdf_path, lang="eng", dpi=300):
    """Internal function for PDF OCR."""
    all_page_texts = []
    doc = None
    basename = os.path.basename(pdf_path)
    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        if num_pages == 0:
            logger.warning(f"PDF '{basename}' has 0 pages.")
            return ""

        for page_num in range(num_pages):
            page_text_content = ""
            try:
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=dpi, alpha=False)
                try:
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                except ValueError:
                    img_bytes = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_bytes))

                page_text = pytesseract.image_to_string(img, lang=lang)
                if page_text.strip():
                    page_text_content = f"\n--- Page {page_num + 1}/{num_pages} ---\n{page_text}"
                else:
                    page_text_content = ""
                    logger.warning(f"No text detected on page {page_num + 1} of '{basename}'.")
            except pytesseract.TesseractError as e:
                err_msg = f"Tesseract Error on page {page_num + 1} of '{basename}': {e}"
                if "Failed loading language" in str(e):
                    err_msg += f" - Hint: Ensure the language pack for '{lang}' is installed."
                logger.error(err_msg)
                page_text_content = f"\n--- TESSERACT ERROR ON PAGE {page_num + 1} ---"
            except Exception as e:
                logger.error(f"Error processing page {page_num + 1} of '{basename}': {e}")
                page_text_content = f"\n--- ERROR PROCESSING PAGE {page_num + 1} ---"

            if page_text_content:
                all_page_texts.append(page_text_content)

        final_text = "".join(all_page_texts).strip()
        if not final_text:
            logger.warning(f"No text extracted from PDF '{basename}' after processing all pages.")
        return final_text
    except FileNotFoundError:
        raise OCRError(f"PDF file not found: '{pdf_path}'")
    except fitz.PyMuPDFError as e:
        raise OCRError(f"MuPDF error opening/processing PDF '{basename}': {e}. "
                       "It might be corrupted or password-protected.")
    except pytesseract.TesseractNotFoundError:
        raise OCRError(
            "Tesseract executable not found. Ensure it's installed and in PATH."
        )
    except Exception as e:
        raise OCRError(f"Unexpected error processing PDF '{basename}': {e}")
    finally:
        if doc:
            doc.close()


def extract_text_from_file(file_path, lang="eng", dpi=300):
    """
    Extracts text from an image or PDF file.

    Args:
        file_path (str): Path to the input file.
        lang (str): Language for OCR.
        dpi (int): DPI for PDF rendering.

    Returns:
        str: Extracted text.

    Raises:
        OCRError: If OCR fails or file type is unsupported.
        FileNotFoundError: If the file doesn't exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    file_ext = os.path.splitext(file_path)[1].lower()
    logger.info(f"Extracting text from: {os.path.basename(file_path)} (Type: {file_ext})")

    if file_ext == ".pdf":
        return _perform_pdf_ocr(file_path, lang=lang, dpi=dpi)
    elif file_ext in [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif"]:
        return _perform_image_ocr(file_path, lang=lang)
    else:
        raise OCRError(
            f"Unsupported file type: '{file_ext}'. Please use PDF or common image formats."
        )