import streamlit as st
import os
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings

from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()
huggingfacehub_api_token = os.getenv("huggingfacehub_api_token")

# repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
# #repo_id = "meta-llama/Meta-Llama-3-70B-Instruct"
# llm = HuggingFaceEndpoint(repo_id=repo_id,
#                         max_new_tokens=512,
#                         temperature=0.1,
#                         token=huggingfacehub_api_token)

#By using st.set_page_config(), you can customize the appearance of your Streamlit application's web page
st.set_page_config(page_title="LLM: Educate Kids App", page_icon=":robot:")
st.header("Hey, Ask me something & I will give out similar things")

#Initialize the HuggingFaceEmbeddings object
# model_name = "mixedbread-ai/mxbai-embed-large-v1"
# hf_embeddings = HuggingFaceEmbeddings(
#     model_name=model_name,
#     huggingfacehub_api_token=huggingfacehub_api_token,
#     )

hf_embeddings = HuggingFaceEndpointEmbeddings(
    model= "mixedbread-ai/mxbai-embed-large-v1",
    task="feature-extraction",
    huggingfacehub_api_token=huggingfacehub_api_token,
)

texts = ["Hello, world!", "How are you?"]
hf_embeddings.embed_documents(texts)

#The below snippet helps us to import CSV file data for our tasks
from langchain.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path='myData.csv', csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['Words']
})

#Assigning the data inside the csv to our variable here
data = loader.load()

#Display the data
print(data)

db = FAISS.from_documents(data, hf_embeddings)

#Function to receive input from user and store it in a variable
def get_text():
    input_text = st.text_input("You: ", key= input)
    return input_text


user_input=get_text()
submit = st.button('Find similar Things')  

if submit:
    
    #If the button is clicked, the below snippet will fetch us the similar text
    docs = db.similarity_search(query=user_input,k=4)
    print(docs)
    st.subheader("Top Matches:")
    st.text(docs[0].page_content)
    st.text(docs[1].page_content)
    st.text(docs[2].page_content)
    st.text(docs[3].page_content)
    # st.text(docs[1].page_content)

