# import os 
# import getpass 
# from langchain.chat_models import init_chat_model
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.document_loaders import DataFrameLoader
from clean_data import clean_df_from_path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


#following block of code has been commented due to my use of ollama instead
#inorder to not worry about rate limits

# if not os.environ.get("GOOGLE_API_KEY"):
#     os.environ["GOOGLE_API_KEY"]=getpass.getpass("Enter API Key for Google Gemini:")

#initialize a model
model=ChatOllama(model="gemma3",temperature=0)
# model=init_chat_model("google_genai:gemini-2.5-flash")

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

loader=DataFrameLoader(df,page_content_column="Job Description")
docs=loader.load()

text_spliiter=RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100,
    add_start_index=True
)

all_splits=text_spliiter.split_documents(docs)

print(f"Split job posts into {len(all_splits)} sub-documents.")

document_ids=[]
batch_size = 128

for i in range(0, len(all_splits), batch_size):
    document_ids+=vector_store.add_documents(all_splits[i:i+batch_size])

print(document_ids[:3])
print("Finished adding to vector store!")