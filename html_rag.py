from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM,pipeline, AutoTokenizer, BitsAndBytesConfig
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA 
import torch

htmlLoader = UnstructuredHTMLLoader("/media/nsl3090-3/hdd1/hujaifa/Langchain_RAG/HTML/webpage.html")
documents = htmlLoader.load()
print('Document:::')
print(documents)
# exit()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=10, separators='\n\n\n')
docs = text_splitter.split_documents(documents)

modelPath = "intfloat/multilingual-e5-large"
model_kwargs = {'device':'cuda:0'}
encode_kwargs = {'normalize_embeddings':False}
embeddings = HuggingFaceEmbeddings(
  model_name = modelPath,  
  model_kwargs = model_kwargs,
  encode_kwargs=encode_kwargs
)

db = FAISS.from_documents(docs, embeddings)
question = "バングラデシュで最も人気のあるスポーツは何ですか?"
searchDocs = db.similarity_search(question)
print('Similar Chunk:::')
print(searchDocs[0].page_content)
# exit()

tokenizer = AutoTokenizer.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b-fast-instruct")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b-fast-instruct",
                                            #  quantization_config=bnb_config,
                                            device_map="auto",
                                            torch_dtype=torch.bfloat16,
                                            attn_implementation="flash_attention_2",
                                            max_length = 64)
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device='cuda:1')
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(
    pipeline = pipe,
    model_kwargs={"temperature": 0, "max_length": 64},
)

# template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. 
# {context}
# Question: {question}
# Helpful Answer:"""

template = """[INST] <>
あなたは誠実で優秀な日本人のアシスタントです。
マークダウン形式で以下のコンテキスト情報を元に質問に回答してください。
<>

{context}

{question}
[/INST]"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
print('Result:::')
qa_chain = RetrievalQA.from_chain_type(   
  llm=llm,   
  chain_type="stuff",   
  retriever=db.as_retriever(),   
  chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} 
) 
result = qa_chain ({ "query" : question })['result']
print(f'Question:::{question}')
print('Result:::') 
print(result[result.rfind('[/INST]')+7:])