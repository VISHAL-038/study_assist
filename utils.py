import fitz  # this is PyMuPDF, imported as fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_path):
    """
    Opens a PDF file and extracts all text from every page.
    Returns one big string of all the text combined.
    """
    doc = fitz.open(pdf_path)  # open the PDF
    full_text = ""

    for page in doc:                        # loop through each page
        full_text += page.get_text()        # extract text from the page

    doc.close()
    return full_text


def split_text_into_chunks(text):
    """
    Splits a large block of text into smaller overlapping chunks.
    
    chunk_size=500    → each chunk is ~500 characters long
    chunk_overlap=50  → chunks share 50 chars with the next chunk
                        (so no sentence gets cut off at a boundary)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)  # returns a list of strings
    return chunks