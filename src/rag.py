from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.agents import create_agent,AgentState
from langchain.agents.middleware import after_model
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from langchain.agents.middleware import dynamic_prompt,ModelRequest
from langchain_ollama import ChatOllama
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
    retrieved_docs=vector_store.similarity_search(last_query,k=5)

    docs_content="\n\n".join(f"Metadata:{doc.metadata}\nDescription:{doc.page_content}" for doc in retrieved_docs)

    system_message = (
    "You are a helpful assistant for job seekers. "
    "Answer the user's question **only using the provided context**. "
    "The context contains job listings with details such as Position, Company, Location, Date Published, Job Description, and other metadata.\n\n"
    "When answering, follow these rules:\n"
    "1. Provide a **concise and accurate answer**.\n"
    "2. Use **metadata fields** (Position, Location, Company, Date Published) whenever relevant.\n"
    "3. Do not make up information that is not in the context.\n"
    "4. Highlight the most relevant jobs for the user's query.\n"
    "5. If multiple jobs are relevant, summarize them clearly and briefly.\n\n"
    f"Context:\n{docs_content}")
    return system_message



@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove old messages to keep conversation manageable."""
    messages = state["messages"]
    if len(messages) > 6:
        # remove the earliest two messages
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:-2]]} # type: ignore
    return None

model=init_chat_model("google_genai:gemini-2.5-flash")

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