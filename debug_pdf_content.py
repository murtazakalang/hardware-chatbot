
from langchain_community.document_loaders import PDFPlumberLoader
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_pdf_content():
    # Finding the file first
    from pathlib import Path
    pdf_dir = Path("PDF")
    files = list(pdf_dir.glob("*.pdf"))
    
    target_file = None
    for f in files:
        if "HBS" in f.name:
            target_file = f
            break
            
    if not target_file:
        print("Could not find HBS PDF file.")
        return

    print(f"Debugging content from: {target_file}")
    loader = PDFPlumberLoader(str(target_file))
    docs = loader.load()
    
    print(f"Total pages: {len(docs)}")
    
    for i, doc in enumerate(docs):
        content = doc.page_content
        print(f"\n--- PDF Index {i} ---")
        # Check for keywords
        if "shear" in content.lower():
            print("✅ Contains 'shear'")
        if "hbs8200" in content.lower():
            print("✅ Contains 'HBS8200'")
        if "minimum distances" in content.lower():
            print("✅ Contains 'minimum distances'")
            
        # Print footer/header hint if possible (last 50 chars)
        print(f"End of page text: ...{content[-50:].replace(chr(10), ' ')}")

if __name__ == "__main__":
    debug_pdf_content()
