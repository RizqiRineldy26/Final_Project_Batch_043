import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
import os
import nltk

from supabase import create_client
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import NLTKTextSplitter
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

warnings.filterwarnings("ignore")

# CONFIG MODEL AND API KEY FOR GEMINI AND SUPABASE
CHAT_MODEL = "models/gemini-3.5-flash"
EMBEDDING_MODEL = "gemini-embedding-2-preview"
EMBEDDING_MODEL_HF = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_HF_2 = "Qwen/Qwen3-Embedding-0.6B"
CHROMA_DIR = "./chroma_db"
JOURNAL_BUCKET = "Journal"
JOURNAL_FILE = "jurnal skincare.pdf"
 
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.environ.get("SUPABASE_URL"))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.environ.get("SUPABASE_KEY"))

chat_model = ChatGoogleGenerativeAI(google_api_key=GEMINI_API_KEY, model=CHAT_MODEL)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ACCESS SUPABASE TABLE
@st.cache_data
def load_table():
    all_rows = []
    batch_size = 1000
    start = 0

    while True:
        response = (
            supabase.table("skincare_cleaned")
            .select("*")
            .range(start, start + batch_size - 1)
            .execute()
        )
        batch = response.data
        if not batch:
            break
        all_rows.extend(batch)
        start += batch_size

    return pd.DataFrame(all_rows)

    # response = supabase.table("skincare_cleaned").select("*").execute()
    # return pd.DataFrame(response.data)

df = load_table()

# BUILD RETRIEVER
@st.cache_resource
def build_retriever():
    pdf_url = supabase.storage.from_("Journal").get_public_url("jurnal skincare.pdf")
    loader = PyPDFLoader(pdf_url)
    pages = loader.load_and_split()

    nltk.download("punkt_tab")
    text_splitter = NLTKTextSplitter(separator="\n\n", chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(pages)

    embedding_model_hf = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_HF,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model_hf,
        persist_directory="./chroma_db",
    )

    db_connection = Chroma(
    persist_directory='./chroma_db',
    embedding_function=embedding_model_hf
    )

    return db_connection.as_retriever(search_kwargs={"k": 10})

with st.spinner("Loading the resource..."):
    retriever = build_retriever()

# BUILD PROMPT TEMPLATE
chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
            You are an AI that gives skincare recommendations based on the provided context. 
            Only recommend products and ingredients that appear in the "Matching products" list below. 
            Do not invent or assume ingredients that aren't listed there.
            """
        ),
        HumanMessagePromptTemplate.from_template(
            """
            Give your recommended ingredients first, then give your recommended product from the matching product list that match the ingredients.

            journal context: {context}
            matching product: {product_context}
            skin condition: {skin_condition}
            
            INSTRUCTIONS FOR PRODUCT MATCHING:
            - DO NOT require an exact ingredient match (e.g., if Isotretinoin is mentioned, do not restrict yourself only to products explicitly named 'Isotretinoin').
            - Match products based on SEMANTIC SIMILARITY, intended use, and shared benefits:
            - Compare the product descriptions, target concerns, and active categories (e.g., exfoliants, spot treatments, retinoids, soothing agents).
            - If a exact active ingredient is absent, recommend products from the list that address the user's overall skin condition ({skin_condition}).
            - Clearly explain WHY each product is recommended (e.g., "While this product does not contain Isotretinoin, it contains BHA which addresses similar pore congestion...").
            - Only output "None" if the provided product list is completely empty or completely irrelevant to skincare.
            - Show preview link that shows the image of the product recomendation.
            - Generate the image to show the image base on product link.
            """
        ),
    ]
)

output_parser = StrOutputParser()

def filter_product(skin_conditions):
    filtered = df[df['problem'].isin(skin_conditions)]
    return filtered

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": (lambda x: x['skin_conditions_str']) | retriever | format_docs, 
     "product_context": lambda x:filter_product(x['skin_conditions_list']),
     "skin_condition": lambda x: x['skin_conditions_str']}
    | chat_template
    | chat_model
    | output_parser
)

# load model
model = tf.keras.models.load_model('model_2_sigmoid.keras')

def run():
    # judul
    st.title("Check your skin condition!")

    # upload or take a picture of cat
    enable = st.checkbox('Enable your camera')
    camera = st.camera_input('Take a picture of your face', disabled = not enable)
    upload = st.file_uploader('Choose a file')

    picture = camera if camera is not None else upload

    if picture is not None:
        # resize and expand dimension of image
        bytes_data = picture.getvalue()
        img_tensor = tf.io.decode_image(bytes_data, channels=3)
        img_tensor = tf.image.resize(img_tensor, [150, 150])
        img_tensor = tf.expand_dims(img_tensor,axis=0)

        # predict
        pred_prob = model.predict(img_tensor)
        pred_class = np.argmax(pred_prob[0])
        class_names = ['acne', 'blackheads', 'dark spots', 'pores', 'wrinkles']

        pred_class_name = class_names[pred_class]
        pct_prob = [f'{x * 100:.2f}%' for x in pred_prob[0]]

        # masukkan ke dataframe
        inf_df = pd.DataFrame({
            'skin_problem':class_names,
            'prediction': pct_prob
        })

        # show prediction
        st.write('## The number one concern from your face is:', pred_class_name)
        st.write('#### Scroll down to check the full prediction!')
        st.image(picture)

        # show table prediction
        st.dataframe(inf_df)

        result_str = inf_df.to_string()
        inf_df['prediction'] = (inf_df['prediction'].astype(str).str.strip('%').astype(float))
        skin_conditions_list = inf_df.loc[inf_df['prediction'] > 30, 'skin_problem'].tolist()
        skin_conditions_str = ", ".join(skin_conditions_list)

        st.title("Here is your Skin Care Recommendation!")
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({
                "skin_conditions_str": skin_conditions_str,
                "skin_conditions_list": skin_conditions_list,
                })
            st.markdown(response)

if __name__ == '__main__':
    run()
