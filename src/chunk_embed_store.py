import os 
from langchain_community.document_loaders import DataFrameLoader
from src.clean_data import clean_df_from_path,DATA_PATH
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from pathlib import Path
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import dotenv



dotenv.load_dotenv()

#embeddings model
embeddings=OllamaEmbeddings(model="mxbai-embed-large")
pc=Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index=pc.Index(name="job-listings")
vector_store=PineconeVectorStore(index=index,embedding=embeddings)



#following block of code has been commented due to my use of ollama instead
#inorder to not worry about rate limits

# if not os.environ.get("GOOGLE_API_KEY"):
#     os.environ["GOOGLE_API_KEY"]=getpass.getpass("Enter API Key for Google Gemini:")



def chunk_embed_store():

    # embeddings=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    #using in memory vector store for now
    # vector_store=InMemoryVectorStore(embeddings)
    df=clean_df_from_path(str(DATA_PATH))

    # loader=DataFrameLoader(df,page_content_column="Job Description")
    # docs=loader.load()

    docs=[]

    for _,row in df.iterrows():
        content = f"""
        {row['Job Description']}
        """.strip()
        doc = Document(
            page_content=content,
            metadata={
                "job_id":row["ID"],
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
        chunk_size=1500,
        chunk_overlap=150,
    )
    final_docs=[]
    for doc in docs:
        chunks=text_spliiter.split_documents([doc])
        for i,chunk in enumerate(chunks):
            chunk.metadata["chunk"]=i
            final_docs.append(chunk)

    # vector_store.add_documents(final_docs)


    # all_splits=text_spliiter.split_documents(docs)

    # print(f"Split job posts into {len(all_splits)} sub-documents.")

    document_ids=[]
    batch_size = 1024

    for i in range(0, len(final_docs), batch_size):
        document_ids+=vector_store.add_documents(final_docs[i:i+batch_size])

    # print(document_ids[:3])
    print("Finished adding to vector store!")


if __name__=="__main__":
    chunk_embed_store()