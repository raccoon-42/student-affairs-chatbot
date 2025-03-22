from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(file_path, chunk_size=250, chunk_overlap=30):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  
        chunk_overlap=chunk_overlap,  # context intersection parameter
    )
    
    chunks = splitter.split_text(text)
    return chunks

if __name__ == "__main__":
    file_to_chunnnnk = input("Enter txt file name to split into chunks: ")
    chunks = split_text(file_to_chunnnnk) 
    print(f"Text is split into {len(chunks)} chunks.")