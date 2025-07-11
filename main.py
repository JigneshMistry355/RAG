from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"


####################### 1. select chat model ###########################
llm = ChatOllama(model="qwen:0.5b")
# qwen:0.5b produces 1024 dimensional vector
#llama3.2 produces 3071 dimensional vector
# The index of Pinecone requires dimension 1024 in the starter pack, higher dimensions are paid.

####################### 2. select embedding model ######################
embeddings = OllamaEmbeddings(model="qwen:0.5b")


load_dotenv()
####################### 3. select vector store #########################
pinecone_api_key = os.environ.get('PINECONE_API_KEY')
pc = Pinecone(api_key=pinecone_api_key)


# reference --> https://app.pinecone.io/organizations/-OUjth12PC68piJYNBsa/projects/6a7393e6-3d47-4aec-9e39-293ab550128b/indexes
index_name = "developer-quickstart-py2"
if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map": {"text": "chunk_text"}
        }
    )
index = pc.Index(index_name)
vector_store = PineconeVectorStore(embedding=embeddings, index=index) 

#################################################################################
# building an application that answers questions about website's content
#################################################################################

# 1. indexing

####################### Loading documents #######################################

import bs4
from langchain_community.document_loaders import WebBaseLoader

bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

assert len(docs) == 1
print(f"Total characters: {len(docs[0].page_content)}")
print(docs[0].page_content[:500])


####################### Splitting documents ######################################

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents")

# update metadata
total_documents = len(all_splits)
third = total_documents // 3

for i, document in enumerate(all_splits):
    if i < third:
        document.metadata["section"] = "beginning"
    elif i < 2*third:
        document.metadata["section"] = "middle"
    else:
        document.metadata["section"] = "end"


######################### storing documents #######################################

document_ids = vector_store.add_documents(documents=all_splits)
print(document_ids[:3])


###################################################################################

# 2. Retieval and generation

from langchain import hub

prompt = hub.pull("rlm/rag-prompt")

example_messages = prompt.invoke(
    {
        "context": "(context goes here)",
        "question": "(question goes here)" 
    }
).to_messages()

assert len(example_messages) == 1
print(example_messages[0].content)


###################################################################################

# To use langgraph, we need to define

# 1. The state of our application

from langchain_core.documents import Document
from typing_extensions import TypedDict, List, Annotated

from typing import Literal

class Search(TypedDict):
    """ Search query """

    query: Annotated[str, ..., "Search query to run"]
    section: Annotated[
        Literal["beginning", "medium", "end"], ... , "Section to query"
    ]

class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str


# 2. Nodes (application steps)

def analyze_query(state: State):
    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}

def retrieve(state: State):
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(
        query["query"],
        # filter=lambda doc: doc.metadata.get("section") == query["section"],
        filter={"section": query["section"]},
        )
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# 3. Control flow

from langgraph.graph import StateGraph, START

graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()


#############################################################################
# Test your application
#############################################################################

# result = graph.invoke({"question": "What is Task Decomposition?"})
# print(f'Context: {result["context"]}')
# print(f'Answer: {result["answer"]}')
for step in graph.stream(
    {"question": "What does the end of the post say about Task Decomposition?"},
    stream_mode="updates",
):
    print(f"{step}\n\n----------------\n")