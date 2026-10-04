# Files
#  │
#  ├── PDF
#  ├── DOCX
#  ├── PPTX
#  ├── TXT
#  ├── EML
#  ├── JPG/JPEG/PNG
#  │
#  ▼
# Document Parser
#  │
#  ├── PDF Text + IMAGE OCR
#  ├── DOCX Text + Tables + IMAGE OCR
#  ├── PPTX Text + Tables + IMAGE OCR
#  ├── TXT Text
#  ├── Email Body + Attachments + IMAGE OCR
#  ├── JPG/JPEG/PNG
#  │
#  ▼
# Gemini Vision OCR
# (for images mentioned in PDF, DOCX, PPTX, JPG/JPEG/PNG, EML)
#  │
#  ▼
# Merged Content
#  │
#  ▼
# Chunking
#  │
#  ▼
# Gemini Embeddings
#  │
#  ▼
# ChromaDB to store embeddings
#  │
#  ▼
# Retriever searches for relevant chunks
#  │
#  ▼
# Gemini LLM for answering questions based on the retrieved chunks
#  │
#  ▼
# Answer


# | File Type | Text        | Tables | Images OCR               |
# | --------- | ----------- | ------ | ------------------------ |
# | PDF       | ✅           | ✅     | ✅                        |
# | DOCX      | ✅           | ✅     | ✅                        |
# | PPTX      | ✅           | ✅     | ✅                        |
# | TXT       | ✅           | N/A     | N/A                      |
# | EML       | ✅ Body Only | N/A     | ✅ Attachments (optional) |
# | JPG       | N/A           | N/A    | ✅                        |
# | JPEG      | N/A           | N/A    | ✅                        |
# | PNG       | N/A           | N/A    | ✅                        |



# pip install langchain
# pip install langchain-community
# pip install langchain-google-genai
# pip install chromadb

# pip install python-docx
# pip install pdf2image
# pip install pillow

# pip install unstructured
# pip install beautifulsoup4

# pip install pypdf
#pip install python-pptx

#download then unzip poppler then put into C drive
#add to PATH environment variable
#https://github.com/oschwartz10612/poppler-windows/releases/download/v26.09.0-0/Release-26.09.0-0.zip

from importlib.resources import path
import os
import email
import tempfile

from pathlib import Path
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from langchain_community.document_loaders import PyPDFLoader
from docx import Document as DocxDocument

from pdf2image import convert_from_path

from google import genai

from pptx import Presentation

from PIL import Image

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

client = genai.Client()

#############################Process PDF, DOCX, PPTX, TXT, EML, JPG/JPEG/PNG files and extract text, tables, and images using Gemini Vision OCR#############################
#load text file and return as Document
def load_txt(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        text = f.read()

    return Document(
        page_content=text,
        metadata={
            "source": file_path
        }
    )


##############################process email file and extract body text, inline images, and attachments #########################
#load eml file(mail) and extract body text, inline images, and attachments, 
# returning them as Document objects
def load_eml(file_path):

    with open(
        file_path,
        "rb"
    ) as f:

        msg = email.message_from_bytes(
            f.read()
        )

    email_text = []

    for part in msg.walk():

        content_type = (
            part.get_content_type()
        )

        if content_type == "text/plain":

            charset = (
                part.get_content_charset()
                or "utf-8"
            )

            body = (
                part.get_payload(
                    decode=True
                )
                .decode(
                    charset,
                    errors="ignore"
                )
            )

            email_text.append(body)

        elif content_type == "text/html":

            html = (
                part.get_payload(
                    decode=True
                )
            )

            html_text = (
                extract_html_text(
                    html
                )
            )

            email_text.append(
                html_text
            )

    # OCR inline images
    image_text = extract_inline_images(
        msg
    )

    combined_text = "\n".join(
        email_text
    )

    combined_text += (
        "\n\n"
        + image_text
    )

    docs = [

        Document(
            page_content=combined_text,
            metadata={
                "source":
                file_path,
                "type":
                "email"
            }
        )

    ]

    # Process attachments
    docs.extend(
        extract_attachments(
            msg
        )
    )

    return docs

#extract text from html of email body returning the text as a string
def extract_html_text(html):

    soup = BeautifulSoup(html, "html.parser")

    return soup.get_text(
        separator="\n",
        strip=True
    )

#extract inline images from email and process them with Gemini Vision OCR 
# to extract text, returning the text as a string
def extract_inline_images(msg):

    image_texts = []

    for part in msg.walk():

        content_type = part.get_content_type()

        if content_type.startswith("image/"):

            image_data = part.get_payload(
                decode=True
            )

            ext = content_type.split("/")[-1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=f".{ext}"
            ) as tmp:

                tmp.write(image_data)

                img_file = tmp.name

            try:

                ocr_text = process_image(
                    img_file
                )

                image_texts.append(
                    ocr_text
                )

            finally:

                os.remove(
                    img_file
                )

    return "\n".join(
        image_texts
    )

#extract attachments from email and process them based on their file type, 
# returning a list of Document objects
def extract_attachments(msg):

    docs = []

    for part in msg.walk():

        filename = part.get_filename()

        if not filename:
            continue

        attachment_data = part.get_payload(
            decode=True
        )

        ext = os.path.splitext(
            filename
        )[1].lower()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=ext
        ) as tmp:

            tmp.write(
                attachment_data
            )

            path = tmp.name

        try:

            docs.extend(
                load_docx(path)
            )

        except Exception as e:

            print(
                f"Attachment error: {e}"
            )

        finally:

            os.remove(path)

    return docs




#############################process PDF with Gemini Vision OCR to extract text, tables, charts, screenshots, diagrams if included in an image, returning a list of Document objects#################
#load pdf file and extract text using Gemini Vision OCR
def extract_pdf_with_ocr(pdf_file):

    pages = convert_from_path(pdf_file)

    documents = []

    for page_number, image in enumerate(pages):

        image.save(
            "temp_page.jpg",
            "JPEG"
        )

        uploaded = client.files.upload(
            file="temp_page.jpg"
        )

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                uploaded,
                """
                Extract ALL visible text.

                Include:
                - tables
                - charts
                - screenshots
                - images
                - captions

                Output plain text only.
                """
            ]
        )

        documents.append(
            Document(
                page_content=response.text,
                metadata={
                    "source": pdf_file,
                    "page": page_number + 1
                }
            )
        )

    return documents


#############################Main function to process images with Gemini Vision OCR to extract text, tables, charts, screenshots, diagrams if included in an image, returning the text as a string#################
#process image with Gemini Vision OCR to extract text, tables, charts, screenshots,
#  diagrams if include in an image
def process_image(image_file):

    uploaded = client.files.upload(
        file=image_file
    )

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[
            uploaded,
            """
            Extract ALL information.

            Include:
            - text
            - tables
            - charts
            - screenshots
            - diagrams

            Return plain text only.
            """
        ]
    )

    return response.text


##############################Process PPTX file to extract text and images, then process images with Gemini Vision OCR to extract text, returning the text as a string#################
#process PPT file to extract text and images, then process images with Gemini Vision OCR
def process_ppt(file):
    
    ppt_text = extract_ppt_text(file)

    for img in extract_slide_images(file):

        ppt_text += "\n"

        ppt_text += process_image(img)

    return ppt_text

#extract text from pptx file
def extract_ppt_text(file):

    prs = Presentation(file)

    text = []

    for slide in prs.slides:

        for shape in slide.shapes:

            if hasattr(shape, "text"):

                if shape.text:

                    text.append(shape.text)

    return "\n".join(text)

#Extract Images From Slides of a PPTX file and save them as temporary files, returning their paths
def extract_slide_images(pptx_file):

    prs = Presentation(pptx_file)

    image_paths = []

    img_count = 0

    for slide in prs.slides:

        for shape in slide.shapes:

            if shape.shape_type == 13:

                image = shape.image

                ext = image.ext

                img_count += 1

                path = f"temp_{img_count}.{ext}"

                with open(path, "wb") as f:

                    f.write(image.blob)

                image_paths.append(path)

    return image_paths


#############################process docx file and extract text, tables, and images, then process images with Gemini Vision OCR to extract text, returning the text as a string#################
#load docx file and extract text, tables, and images
def load_docx(file_path):

    doc = DocxDocument(file_path)

    paragraphs = "\n".join(
        para.text
        for para in doc.paragraphs
        if para.text.strip()
    )

    tables = extract_tables(doc)

    image_ocr = extract_images(doc)

    combined_text = f"""
{paragraphs}

{tables}

{image_ocr}
"""

    return Document(
        page_content=combined_text,
        metadata={
            "source": file_path,
            "type": "docx"
        }
    )

#extract images from docx file and process them with Gemini Vision OCR
def extract_images(doc):

    image_text = []

    for rel in doc.part.rels.values():

        if "image" in rel.target_ref:

            image = rel.target_part.blob

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".png"
            ) as tmp:

                tmp.write(image)

                tmp_path = tmp.name

            try:

                text = process_image(tmp_path)

                image_text.append(text)

            finally:

                os.remove(tmp_path)

    return "\n".join(image_text)

#extract tables from docx file and return them as text
def extract_tables(doc):

    table_text = []

    for table in doc.tables:

        for row in table.rows:

            cells = []

            for cell in row.cells:
                cells.append(cell.text)

            table_text.append(
                " | ".join(cells)
            )

    return "\n".join(table_text)



#############################load files from a folder and process them based on their file type, returning a list of Document objects#################
#load files from a folder and process them based on their file type
def load_files(folder):

    docs = []

    for file in Path(folder).glob("*"):

        ext = file.suffix.lower()

        if ext == ".pdf":

            docs.extend(
                extract_pdf_with_ocr(
                    str(file)
                )
            )

        elif ext == ".docx":

            docs.append(
                load_docx(
                    str(file)
                )
            )

        elif ext == ".txt":

            docs.append(
                load_txt(
                    str(file)
                )
            )

        elif ext == ".eml":

            docs.extend(
                load_eml(
                    str(file)
                )
            )

        elif ext in [".jpg", ".jpeg", ".png"]:
            docs.append(process_image(str(file)))

        elif ext in [".pptx", ".ppt"]:
            docs.append(process_ppt(str(file)))

    return docs



from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
# )

splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=20
)

documents = load_files("./data_dump")

splits = splitter.split_documents(
    documents
)

print(f"Number of splits: {len(splits)}")
print(f"Content of first split: {splits[0].page_content}")
print(f"Content of second split: {splits[1].page_content}")
print(f"Content of third split: {splits[2].page_content}")


from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)

from langchain_community.vectorstores import Chroma

#this will create a new ChromaDB every time the code is run, if you want to add new documents to the existing ChromaDB, use Chroma.add_documents() instead of Chroma.from_documents()
# db = Chroma.from_documents(
#     splits,
#     embeddings,
#     persist_directory="./chroma_db"
# )


CHROMA_PATH = "./chroma_db"

if os.path.exists(CHROMA_PATH) and len(os.listdir(CHROMA_PATH)) > 0:
    chroma_db_exists = True
else:
    chroma_db_exists = False

if chroma_db_exists:

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    db.add_documents(splits)
    print("Added new documents to existing ChromaDB.")
else:

    db = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
        
    )
    print("Created new ChromaDB and added documents.")    

# from langchain_google_genai import (
#     ChatGoogleGenerativeAI
# )

# llm = ChatGoogleGenerativeAI(
#     model="gemini-3-flash-preview",
#     temperature=0
# )


# def ask(question):

#     docs = db.similarity_search(
#         question,
#         k=5
#     )

#     context = "\n\n".join(
#         d.page_content
#         for d in docs
#     )

#     prompt = f"""
# Answer only using context.

# Context:
# {context}

# Question:
# {question}
# """

#     response = llm.invoke(prompt)

#     if isinstance(response.content, str):
#         return response.content

#     return "\n".join(
#         block["text"]
#         for block in response.content
#         if block["type"] == "text"
#     )








