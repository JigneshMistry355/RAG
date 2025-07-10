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


######################### storing documents #######################################

document_ids = vector_store.add_documents(documents=all_splits)
print(document_ids[:3])


###################################################################################

# 2. Retieval and generation

