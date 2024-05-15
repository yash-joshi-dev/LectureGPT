# Lecture GPT
A simple RAG-based chatbot that can answer questions about an audio recording. Made with the Cohere API, LangChain, and Gradio.

## How it works:
1. With LangChain, I transcribe/load the the audio file with the AssemblyAI document loader and chunk it up.
2. Using the Cohere embedding model (**embed-english-v2.0**), I create embeddings for each chunk add them to an Annoy search index.
3. Again using the Cohere embedding model, I embed the user query and find the top 10 similar items in the search index.
4. Using the Cohere rerank model (**rerank-english-v2.0**), I rerank the the results in order of descending relevance and take the top 3 for context.
5. Finally, I add the context into a prompt with the user's query and use the Cohere LLM (the **command-nightly** experimental model) to generate an answer.

## Running Locally:
1. run `pip install -r requirements.txt`
2. run `python main.py`
