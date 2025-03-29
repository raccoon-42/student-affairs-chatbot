import pdfplumber

pdf_path = "yonerge.pdf"
txt_path = "yonerge.txt"

def extract_pdf(pdf_path, output_file):
    with pdfplumber.open(pdf_path) as pdf:
        with open(output_file, 'w', encoding='utf-8') as file:
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=1)
                file.write(text)
            
            
if __name__ == "__main__":
    extract_pdf(pdf_path, txt_path)