from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.agents import create_agent,AgentState
from langchain_ollama import ChatOllama
from langchain.agents.middleware import after_model
from langchain_openai import OpenAIEmbeddings
from langgraph.runtime import Runtime
from langchain.agents.middleware import dynamic_prompt,ModelRequest
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import RemoveMessage
from langchain_core.prompts import ChatPromptTemplate
from src.chunk_embed_store import vector_store,embeddings
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever


load_dotenv()

# vector_store=Chroma(
#     collection_name="job_listings",
#     embedding_function=embeddings,
#     persist_directory="./job_listings_db",
# )

retreiver=vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":10}
)
compressor=FlashrankRerank()
compression_retriever=ContextualCompressionRetriever(base_compressor=compressor,base_retriever=retreiver)
# model=init_chat_model("google_genai:gemini-2.5-flash")
model=ChatOllama(model="phi3:latest",temperature=0)

# model=init_chat_model("gpt-4.1")

#two step chain
#1 where we always run the search query
#2 use the prompt to generate response

#prompt to condense message
condense_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history to help accurately answer users query. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
    ("placeholder", "{messages}"),
])


condense_chain=condense_prompt| model

@dynamic_prompt
def prompt_with_context(request:ModelRequest)->str:
    last_query=request.state["messages"][-1].text
    messages=request.state["messages"]
    if len(messages)>1:
        print("checking if it has enough content:::::")
        print(messages)
        single_query=condense_chain.invoke({"messages":messages[:-4]})
        print("this is the single queryyyy")
        print(single_query)
        search_query=single_query.content
    else:
        search_query=messages[-1].content
    retrieved_docs=compression_retriever.invoke(search_query) # type: ignore

    if not retrieved_docs:
        retrieved_docs=vector_store.similarity_search(search_query,k=5) # type: ignore
    retrieved_docs=retrieved_docs[:4]
    docs_content="\n\n".join(f"Metadata:{doc.metadata}\nDescription:{doc.page_content}" for doc in retrieved_docs)

    system_message = (f"""
    You are a helpful assistant for job seekers.

    Your task is to answer the user's question using ONLY the provided context and relevant information from the previous conversation when helpful.

    The context consists of job postings. Each job posting may contain:
    - Metadata fields: ID, Position, Company, Location, Date Published
    - Job Description text

    Follow these rules strictly:

    1. Use ONLY the information present in the provided context. Do NOT make up or infer missing details.
    2. Identify and highlight the job postings that are most relevant to the user's question.
    3. When discussing a job, always include key metadata when available:
    - Position
    - Company
    - Location
    - Date Published
    4. Use the job description only to support or explain details already present in the posting.
    5. Always quote the exact pieces of context (including metadata or job description text) that you used to answer the question.
    6. If multiple jobs are relevant, summarize each one clearly and briefly.
    7. If the context does not contain enough information to answer the question, say so explicitly.
    8. Output must be plain text only. Do not use markdown, bullet symbols, or formatting.

    Context:
    {docs_content}
    """)
    print(system_message)
    return system_message



@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove old messages to keep conversation manageable."""
    messages = state["messages"]
    if len(messages) > 6:
        # remove the earliest two messages
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:-2]]} # type: ignore
    return None



def initialize_agent():
    agent=create_agent(model,tools=[],middleware=[prompt_with_context,delete_old_messages],checkpointer=InMemorySaver())
    return agent

# agent=initialize_agent()
# query = "what roles require me to do an extreme backflip every 2 seconds"
# for step in agent.stream(
#     {"messages": [{"role": "user", "content": query}]},
#     stream_mode="values",
# ):
#     step["messages"][-1].pretty_print()