# RAG Pipeline Architecture

## 1) Setup

1. Clone the github repository
2. Create a virtual environment
3. Install the required packages mention in the `requirements.txt` using:

```bash
 pip install -r requirements.txt
```

4. Setup the vector store by running the command:

```bash
python src/chunk_embed_store.py.
```

5. Wait till the command fully runs which may take a few minutes
   Run the backend using: uvicorn main:app –reload
   Go to localhost:8000/#docs/ and use the POST endpoint

RAG Pipeline Architecture

## 1) Setup

- Clone the github repository
- Create a virtual environment
- Install the required packages mention in the requirements.txt using **pip install -r requirements.txt**
- Setup the vector store by running the command: python src/chunk_embed_store.py. Wait till the command fully runs which may take a few minutes
- Run the backend using: **uvicorn main:app -reload**
- Go to localhost:8000/#docs/ and use the POST endpoint

## 2) Overview of the Architecture

- **Data Cleaning**

  - First the jobs dataset is loaded into a **pandas** dataframe for fast retrieval and modification to help clean data.
  - The job description is cleaned using **beautifulsoup4** using html parser which helps clean job descriptions and returns only text input.
  - Missing Tags where changed to "Unknown"

- **Chunking, Embedding and Storing**

  - Each row of the job description was stringified and metadata was added using the data from all columns except job description. This was done so the LLM has richer context about the job data
  - For each document, if it exceeded chunk size it was split using the **RecursiveCharacterSplitter** as it is important to fit data into relevant context sizes . Also, each document was split separately so **chunk id** can be added as metadata which might help when multiple chunks of the same document are retrieved
  - After this all the documents were stored in a local, persistent, **ChromaDB** vector database which used an **Ollama** embedding model called **nomic-embed-text**

- **RAG**
  - The RAG pipeline uses the vector store that stores all our documents as a retriever that retrieves 5 documents based on our query
  - For each query there are two steps that takes place using "**@dynamic_prompt**"
    - Retrieve relevant chunks of data based on the query
    - Use a system prompt to give the LLM instructions on the task to perform based on the context and add the retrieved chunks as context
  - There is also a **short term memory** that is stored in memory so the model has context about the last few messages. This uses a thread config where each thread stores memory for a user separately.
  - A deletion mechanism is added so that if user interactions become too long, the earliest two messages are deleted to prevent excessive memory use and slow answers
  - **Phi4-mini** was used as a local model to create an agent to test this architecture due to limited credits from cloud based models such as gemini
- **Access**
  - A POST endpoint was created that can be used to query this agent. It takes a query and thread that the user is using and returns the LLM response

## 3) Usage

Please note only the content part of the response is shown:

Example queries and response:

- Query: "**what jobs mention the use of nlp and llms**"

```bash
"content": "Based on the provided context, there are several job postings that explicitly reference \\"Natural Language Processing (NLP)\\" or Large Language Models (\\"LLMs\\"):\\n\\n\*\*Relevant Job Postings:\*\*\\n\\n1.\\n- \*\*Title:\*\* NLP / LLM Scientist - Applied AI ML Lead\\n- \*\*Company Name:\*\* Machine Learning Centre of Excellence at TELUS Digital\\n- \*\*Publication Date:\*\* June 19, 2025T00:10:25Z\\n\\n2. \\n- \*\*Title:\*\* Machine Learning Manager - GenAI (GenAI)\\n- \*\*Company Name:\*\* Booking.com\\n- \*\*Publication Date:\*\* May 9, 2025T11:36:38Z\\n\\n3.\\n- \*\*Job Level and Category Mentioned in Description Summary for Job Posting #1\*\*:\\n - Senior Level\\n - Data and Analytics category\\n\\nThese job postings are related to roles that involve NLP or LLMs. The first position is a senior-level role at TELUS Digital's Machine Learning Centre of Excellence, which involves applying AI/ML methods including natural language processing (NLP) for various tasks like speech analytics.\\n\\nThe second posting describes the responsibilities and qualifications required by Booking.com's Senior Level ML Manager - GenAI who will work with LLMs tailored to their domain. The job description mentions developing state-of-the-art machine learning models, which likely includes NLP applications given that they are working on large language model (LLM) solutions.\\n\\nPlease note there may be additional relevant postings not included in the provided context or if more listings were available beyond what was shared initially.",

```

- Query: **what jobs mention the use of golang or go?**

```bash
"The following job postings involve Golang:\\n\\n1. \*\*Golang Lead - UK\*\* at Infosys\\n Role: Technology lead/golang lead, located in London, UK.\\n\\n2. MGR BRANCH L2 Sales Manager (Japan) at Atlassian does not mention the use of Go.\\n3. Field Marketing Manager, SLED at GitLab mentions Golang as a required skill for developing software solutions using Go programming language and focuses on performance and reliability."
```

## 4) Drawbacks and Future Enhancements

- **Slow Setup**: The chunk_embed_store.py embeds and stores documents in a vector database. Currently, it takes a few minutes to finish initially which might not be ideal for a production environment. Further research on whether it is due to the use of a local embedding model or my architecture needs to be done.
- **Type of Response**: Sometimes, the model produces additional texts such as : "I could not find any other jobs based on your query". I believe this problem is easily fixable with the use of a more powerful model
- **Retrieved chunks**: When asking non-specific questions, the chunks retrieved are rarely relevant which causes the LLM to not produce a useful output to the users query.

Future enhancements can include looking at a cross-encoder to rank relevant chunks which can then be used to answer the query, but this will increase overhead and make the LLM answers even slower. Different chunking methods and text splitters can be tested to see which one splits the documents the best
