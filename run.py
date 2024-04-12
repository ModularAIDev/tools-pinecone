from dotenv import load_dotenv
from openai import OpenAI
from aic_tools_pinecone import init_toolbox_with_gptembeddings, _query_from_long_term_memory, _store_in_long_term_memory

import os
import pinecone

load_dotenv()


PINECONE_KEY = os.getenv('PINECONE_KEY')
USER_ID = "113685807430823249126"

client = OpenAI()


pinecone.init(api_key=PINECONE_KEY, environment='gcp-starter')
init_toolbox_with_gptembeddings(USER_ID, client, pinecone)
_store_in_long_term_memory("I love burgers")
print(_query_from_long_term_memory("Who am i?", 2))
print(_query_from_long_term_memory("What do I love?", 2))