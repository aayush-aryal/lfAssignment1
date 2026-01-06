from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.agents import create_agent,AgentState
from langchain_ollama import ChatOllama
from langchain.agents.middleware import after_model
from langgraph.runtime import Runtime
from langchain.agents.middleware import dynamic_prompt,ModelRequest
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import RemoveMessage





load_dotenv()



embeddings=OllamaEmbeddings(model="nomic-embed-text:latest")

vector_store=Chroma(
    collection_name="job_listings",
    embedding_function=embeddings,
    persist_directory="./job_listings_db",
)

retreiver=vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":5}
)


#two step chain
#1 where we always run the search query
#2 use the prompt to generate response

@dynamic_prompt
def prompt_with_context(request:ModelRequest)->str:
    last_query=request.state["messages"][-1].text
    retrieved_docs=vector_store.similarity_search(last_query,k=3)

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
    if len(messages) > 2:
        # remove the earliest two messages
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:-2]]} # type: ignore
    return None

# model=init_chat_model("google_genai:gemini-2.5-flash")
model=ChatOllama(model="phi4-mini:latest",temperature=0)

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