from langchain_community.document_loaders import DirectoryLoader, AssemblyAIAudioTranscriptLoader
from langchain_community.document_loaders.assemblyai import TranscriptFormat
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from annoy import AnnoyIndex
import gradio as gr
import cohere
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()
co = cohere.Client(os.environ['COHERE_API_KEY'])

# create document chunks of audio file
def create_chunks(audio):
  if audio is None:
    raise gr.Error("Must pass audio file.")
  loader = AssemblyAIAudioTranscriptLoader(file_path=audio, api_key=os.environ['ASSEMBLY_AI_API_KEY'])
  documents = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200,
    length_function=len,
    add_start_index=True
  )
  chunks = text_splitter.split_documents(documents)
  return np.array(chunks)

# create the embeddings of the document chunks
def create_embeddings(chunks):
  response = co.embed(
    texts=[x.page_content for x in chunks],
    model='embed-english-v2.0'
  ).embeddings
  return np.array(response)

# create the search index and add the embeddings to it
def create_search_index(embeddings):

  search_index = AnnoyIndex(embeddings.shape[1], 'angular')

  for i in range(len(embeddings)):
    search_index.add_item(i, embeddings[i])

  # 10 trees
  search_index.build(10)
  return search_index

# retrieve the ranked, relevant context from the search index as a string
def retrieve_context(query, search_index: AnnoyIndex, chunks):

  # embed the query
  query_emb = co.embed(texts=[query], model='embed-english-v2.0').embeddings

  # get 10 nearest neighbors
  similar_item_ids = search_index.get_nns_by_vector(query_emb[0], 10, include_distances=True)
  results = chunks[similar_item_ids[0]]

  # rerank
  reranked_results = co.rerank(
    model='rerank-english-v2.0',
    query=query,
    documents=[x.page_content for x in chunks],
    top_n=3
  )

  # Print the reranked results
  return "\n\n----\n\n".join([c.page_content for c in chunks[[r.index for r in reranked_results.results]]])

# prompt the latest (nightly release) cohere model for the answer, provided context
def retrieve_answer(query, context):
  prompt = f"""
    Excerpts from an audio recording (could be overlapping): 
    {context}
    ----
    Question: {query}
    
    Extract the answer of the question from the text provided. 
    If the text doesn't contain the answer, 
    reply that the answer is not available."""

  prediction = co.generate(
      prompt=prompt,
      max_tokens=150,
      model="command-nightly",
      temperature=0.5,
      num_generations=1
  )

  return prediction.generations[0].text

# process a particular query about a particular audio file
def process_query(query, audio):
  chunks = create_chunks(audio)
  embeddings = create_embeddings(chunks)
  search_index = create_search_index(embeddings)
  context = retrieve_context(query, search_index, chunks)
  response = retrieve_answer(query, context)
  return response, context

def main():
  demo = gr.Interface(
    process_query,
    inputs=["text", gr.Audio(type="filepath", label="Talk about something")],
    outputs=[gr.Text(label="Response"), gr.Text(label="Prompt")],
    title="LectureGPT",
    description="Give me an audio recording of a lecture and I will help you figure it out."
  )

  demo.launch()

if __name__ == "__main__":
  main()