from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

import re 

# Step 1 Indexing 


# Document Loading 
def get_video_id(url):
    # Works for watch?v=, youtu.be/, shorts/, embed/, live/ — or a bare ID
    match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})", url)
    if match:
        return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    raise ValueError("Couldn't find a video ID in that URL")

url = "https://www.youtube.com/watch?v=QRFyNJnrHxM" 
video_id = get_video_id(url)

try:
    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
    transcript = " ".join(chunk.text for chunk in transcript_list)
    # print(transcript)
except TranscriptsDisabled:
    print("No captions available for this video.")


# Text Splitting 
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
print(len(chunks))

# Embeddings 
embeddings = OllamaEmbeddings(model="embeddinggemma-2:latest")

# vector store 
vector_store = FAISS.from_documents(chunks, embeddings)
# print(vector_store.index_to_docstore_id)


# step 2  - Retrival 

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# print(retriever.invoke('HArness '))

# Step 3 - Augumentation 

llm = ChatOllama(model="gemma4")
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
)

question = "Hoe to automate things ";
retrieved_docs    = retriever.invoke(question)
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
final_prompt = prompt.invoke({"context": context_text, "question": question})

# Step 4 - Generation 

answer = llm.invoke(final_prompt)
print(answer.content)