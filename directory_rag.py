from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS #Chroma
from transformers import AutoModelForCausalLM,pipeline, AutoTokenizer, BitsAndBytesConfig
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA 
import torch

pdfLoader = DirectoryLoader('/media/nsl3090-3/hdd1/hujaifa/Langchain_RAG/TEMP_JP_LAW/', glob='**/*.txt', loader_cls=TextLoader)
documents = pdfLoader.load()
print('Document::')
print(len(documents))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20, separators='\n\n\n')
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

question = "「日本の労働法において、労働者の権利や労働条件に関する法的な保護はどのように定義されていますか？具体的な労働法に基づいて、労働者がどのような権利を有しているか教えてください。」"
searchDocs = db.similarity_search(question)
print('Similar Chunk:::')
print(searchDocs[0].page_content)

tokenizer = AutoTokenizer.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b-fast-instruct")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b-fast-instruct",
                                             quantization_config=bnb_config,
                                            device_map="auto",
                                            # torch_dtype=torch.bfloat16,
                                            attn_implementation="flash_attention_2",
                                            max_length = 64)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(
    pipeline = pipe,
    model_kwargs={"temperature": 0, "max_length": 64},
)


template = """[INST] <>
あなたは誠実で優秀な日本の弁護士です。
次のコンテキスト情報をマークダウン形式で使用して、法的な質問に答えてください。答えがわからない場合は、答えをでっち上げようとせず、わからないとだけ言ってください。 答えはできるだけ簡潔にしてください。
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
print(f'Question::: {question}')
print('Result:::') 
print(result[result.rfind('[/INST]')+7:])