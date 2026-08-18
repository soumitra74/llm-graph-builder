import logging
from pathlib import Path
import chardet
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredFileLoader
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader

class ListLoader(BaseLoader):
    """
    A wrapper to make a list of Documents compatible with BaseLoader.
    """
    def __init__(self, documents):
        self.documents = documents

    def load(self):
        """
        Returns the list of documents.
        """
        return self.documents

def detect_encoding(file_path):
    """
    Detects the file encoding to avoid UnicodeDecodeError.

    Args:
        file_path (str or Path): Path to the file.

    Returns:
        str: Detected encoding (default "utf-8" if not found).
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read(4096)
        result = chardet.detect(raw_data)
        return result['encoding'] or "utf-8"

def load_document_content(file_path):
    """
    Loads document content from a file, handling PDFs and text encoding.

    Args:
        file_path (str or Path): Path to the file.

    Returns:
        tuple: (loader, encoding_flag)
            loader: Document loader instance.
            encoding_flag (bool): True if non-UTF-8 encoding was used for .txt files.
    """
    file_extension = Path(file_path).suffix.lower()
    encoding_flag = False
    if file_extension == '.pdf':
        loader = PyMuPDFLoader(file_path)
        return loader, encoding_flag
    if file_extension == ".txt":
        encoding = detect_encoding(file_path)
        logging.info("Detected encoding for text file: %s", encoding)
        if encoding.lower() == "utf-8":
            loader = UnstructuredFileLoader(file_path, mode="elements", autodetect_encoding=True)
            return loader, encoding_flag
        with open(file_path, encoding=encoding, errors="replace") as f:
            content = f.read()
        loader = ListLoader([Document(page_content=content, metadata={"source": file_path})])
        encoding_flag = True
        return loader, encoding_flag
    loader = UnstructuredFileLoader(file_path, mode="elements", autodetect_encoding=True)
    return loader, encoding_flag

def _page_text_length(pages):
    return sum(len((page.page_content or "").strip()) for page in pages)


def ocr_pdf_pages(file_path):
    """
    Rasterize PDF pages and OCR them. Used for print-to-PDF files whose text is
    outlined as drawings, so PyMuPDF extracts no characters.
    """
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise Exception(
            "PDF has no extractable text (print-to-PDF / outlined fonts). "
            "Install rapidocr-onnxruntime to OCR pages."
        ) from exc

    import fitz
    import numpy as np

    engine = RapidOCR()
    pdf = fitz.open(file_path)
    pages = []
    try:
        for page_index, page in enumerate(pdf):
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            if pix.n == 1:
                image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)
            else:
                image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 4:
                    image = image[:, :, :3]
            result, _elapsed = engine(image)
            lines = []
            if result:
                for item in result:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        lines.append(str(item[1]))
            text = "\n".join(lines).strip()
            logging.info(
                "OCR page %s/%s produced %s characters for %s",
                page_index + 1,
                pdf.page_count,
                len(text),
                Path(file_path).name,
            )
            pages.append(
                Document(
                    page_content=text,
                    metadata={"source": str(file_path), "file_path": str(file_path), "page": page_index},
                )
            )
    finally:
        pdf.close()
    return pages


def get_documents_from_file_by_path(file_path, file_name):
    """
    Loads documents from a file by its path and returns file name, pages, and extension.

    Args:
        file_path (str or Path): Path to the file.
        file_name (str): Name of the file.

    Returns:
        tuple: (file_name, pages, file_extension)

    Raises:
        Exception: If file does not exist or reading fails.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logging.info('File %s does not exist', file_name)
        raise Exception(f'File {file_name} does not exist')
    logging.info('file %s processing', file_name)
    try:
        loader, encoding_flag = load_document_content(file_path)
        file_extension = file_path.suffix.lower()
        if file_extension == ".pdf" or (file_extension == ".txt" and encoding_flag):
            pages = loader.load()
        else:
            unstructured_pages = loader.load()
            pages = get_pages_with_page_numbers(unstructured_pages)
        if file_extension == ".pdf" and _page_text_length(pages) == 0:
            logging.warning(
                "No extractable text in %s; falling back to OCR for print-to-PDF content",
                file_name,
            )
            pages = ocr_pdf_pages(file_path)
            if _page_text_length(pages) == 0:
                raise Exception(f"OCR produced no text for {file_name}")
    except Exception as exc:
        raise Exception(f'Error while reading the file content or metadata, {exc}')
    return file_name, pages, file_extension

def get_pages_with_page_numbers(unstructured_pages):
    """
    Groups unstructured pages into logical pages with page numbers and metadata.

    Args:
        unstructured_pages (list): List of Document objects.

    Returns:
        list: List of Document objects with page numbers and metadata.
    """
    pages = []
    page_number = 1
    page_content = ''
    metadata = {}
    for idx, page in enumerate(unstructured_pages):
        if 'page_number' in page.metadata:
            if page.metadata['page_number'] == page_number:
                page_content += page.page_content
                metadata = {
                    'source': page.metadata['source'],
                    'page_number': page_number,
                    'filename': page.metadata['filename'],
                    'filetype': page.metadata['filetype']
                }
            if page.metadata['page_number'] > page_number:
                page_number += 1
                pages.append(Document(page_content=page_content))
                page_content = ''
            if page == unstructured_pages[-1]:
                pages.append(Document(page_content=page_content))
        elif page.metadata.get('category') == 'PageBreak' and page != unstructured_pages[0]:
            page_number += 1
            pages.append(Document(page_content=page_content, metadata=metadata))
            page_content = ''
            metadata = {}
        else:
            page_content += page.page_content
            metadata_with_custom_page_number = {
                'source': page.metadata['source'],
                'page_number': 1,
                'filename': page.metadata['filename'],
                'filetype': page.metadata['filetype']
            }
            if page == unstructured_pages[-1]:
                pages.append(Document(page_content=page_content, metadata=metadata_with_custom_page_number))
    return pages
