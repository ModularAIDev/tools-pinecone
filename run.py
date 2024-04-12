from dotenv import load_dotenv
from openai import OpenAI
from aic_tools_pinecone import PineconeToolbox

import os
import pinecone

load_dotenv()


PINECONE_KEY = os.getenv('PINECONE_KEY')
USER_ID = "113685807430823249126"

client = OpenAI()


pinecone.init(api_key=PINECONE_KEY, environment='gcp-starter')
tools = PineconeToolbox.create_toolbox_with_gptembeddings(USER_ID, client, pinecone)
tools._store_in_long_term_memory("I love burgers")
print(tools._query_from_long_term_memory("Who am i?", 2))
print(tools._query_from_long_term_memory("What do I love?", 2))