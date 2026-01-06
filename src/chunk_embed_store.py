# import os 
# import getpass 
# from langchain.chat_models import init_chat_model
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.document_loaders import DataFrameLoader
from clean_data import clean_df_from_path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


#following block of code has been commented due to my use of ollama instead
#inorder to not worry about rate limits

# if not os.environ.get("GOOGLE_API_KEY"):
#     os.environ["GOOGLE_API_KEY"]=getpass.getpass("Enter API Key for Google Gemini:")



#embeddings model
embeddings=OllamaEmbeddings(model="nomic-embed-text:latest")
# embeddings=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

#using in memory vector store for now
# vector_store=InMemoryVectorStore(embeddings)
vector_store=Chroma(
    collection_name="job_listings",
    embedding_function=embeddings,
    persist_directory="./job_listings_db"
)

#i will kinda hardcode this for niw
df=clean_df_from_path("/Users/aayush-aryal/Documents/lfAssignment1/jobs.xlsx")

# loader=DataFrameLoader(df,page_content_column="Job Description")
# docs=loader.load()

docs=[]

for _,row in df.iterrows():
    content = f"""
    Job Title, Company, Location, Date Published, Job ID.
    Job Title: {row['Job Title']}
    Category: {row['Job Category']}
    Tags: {row['Tags']}
    Company Name: {row['Company Name']}
    Publication Date: {row['Publication Date']}
    Job Level: {row['Job Level']}

    Description:
    {row['Job Description']}
    """.strip()
    doc = Document(
        page_content=content,
        metadata={
            "title": row["Job Title"],
            "category": row["Job Category"],
            "tags": row["Tags"],
            "company_name":row["Company Name"],
            "publication_date":row["Publication Date"],
            "job_level":row['Job Level']
        }
    )

    docs.append(doc)

text_spliiter=RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100,
)
final_docs=[]
for doc in docs:
    chunks=text_spliiter.split_documents([doc])
    for i,chunk in enumerate(chunks):
        chunk.metadata["chunk"]=i
        final_docs.append(chunk)

vector_store.add_documents(final_docs)


# all_splits=text_spliiter.split_documents(docs)

# print(f"Split job posts into {len(all_splits)} sub-documents.")

# document_ids=[]
# batch_size = 128

# for i in range(0, len(all_splits), batch_size):
#     document_ids+=vector_store.add_documents(all_splits[i:i+batch_size])

# print(document_ids[:3])
print("Finished adding to vector store!")