from langchain_community.document_loaders import DirectoryLoader, AssemblyAIAudioTranscriptLoader
from langchain_community.document_loaders.assemblyai import TranscriptFormat
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from annoy import AnnoyIndex
import gradio as gr
import cohere
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
co = cohere.Client(os.environ['COHERE_API_KEY'])

def create_search_index(audio):
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
  # print(len(chunks))
  response = co.embed(texts=[x.page_content for x in chunks], model='embed-english-v2.0').embeddings
  embeddings = np.array(response)

  search_index = AnnoyIndex(embeddings.shape[1], 'angular')

  for i in range(len(embeddings)):
    search_index.add_item(i, embeddings[i])

  search_index.build(10) # 10 trees
  search_index.save('test.ann')
  return np.array(chunks), search_index

  

def process_query(query, audio):

  chunks, search_index = create_search_index(audio)
  print("done creatign index")

  query_emb = co.embed(texts=[query], model='embed-english-v2.0').embeddings

  # get nearest neighbors
  similar_item_ids = search_index.get_nns_by_vector(query_emb[0], 10, include_distances=True)
  results = chunks[similar_item_ids[0]]
  # print(results)

  # rerank
  reranked_results = co.rerank(model='rerank-english-v2.0', query=query, documents=[x.page_content for x in chunks], top_n=3)

  # Print the reranked results
  context = "\n\n----\n\n".join([c.page_content for c in chunks[[r.index for r in reranked_results.results]]])

  # prompt the latest cohere model
  # Prepare the prompt
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

  return prediction.generations[0]



def main():
  demo = gr.Interface(
    process_query,
    inputs=["text", gr.Audio(type="filepath", label="Talk about something")],
    outputs=["text", "text"],
    title="Document QA Chatbot",
    description="Give a pdf document to ask questions about. If not given, will have the context of the 'Alice In Wonderland' book."
  )

  demo.launch()

if __name__ == "__main__":
  main()