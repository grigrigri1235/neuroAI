import pymupdf4llm
import sys

def convert():
    try:
        input_pdf = "/home/eran.b/neuroAI/og_coconut.pdf"
        output_md = "/home/eran.b/neuroAI/og_coconut.md"
        
        md_text = pymupdf4llm.to_markdown(input_pdf)
        
        with open(output_md, "w", encoding="utf-8") as f:
            f.write(md_text)
            
        print("Conversion successful.")
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    convert()
