from fastapi import FastAPI
from pydantic import BaseModel
from src.rag import initialize_agent


app=FastAPI()
agent=initialize_agent()


class QueryRequest(BaseModel):
    query: str
    thread_id: str



@app.get("/")
def read_root():
    return {"message":"Hello World"}


@app.post("/query")
async def ask_agent(request:QueryRequest):
    response_text=""
    config = {"configurable": {"thread_id": request.thread_id}}
    result = agent.invoke(
    {
        "messages": [{"role": "user", "content":request.query }],
    }, # type: ignore
    config=config) # type: ignore
    return {"response":result} # type: ignore
