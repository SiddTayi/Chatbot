# Code to create and run the bot. 

from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

import chainlit as cl


DB_FAISS_PATH = 'vectorstores/db_faiss'


custom_prompt_template = """
Use the following information to answer the user's questions. If you don't have an answer, please respond with "No knowledge". 
Do not try to make any assumptions about the answer and do not make any answers.

Context: {context}
Question: {question}


Only returns the actual known answer and nothing else. 

Answer:

"""

def custom_prompt():
    """
    Prompt template for QA Retrieval for each vector stores. -- RAG (Retrieval Augmented Graph)

    Explanation: n LLM (Large Language Model) terms, "QA retrieval" refers to the process of retrieving relevant information from a knowledge base to answer a given question.
    This retrieval step is often performed as part of a retrieval-augmented generation (RAG) approach, where the model first retrieves relevant passages or documents from a knowledge 
    base and then generates an answer based on the retrieved information.

    
    QA Retrieval in LLM:
    1. Retrieval: Matching the query against the knowledge base to idenify the text that are likely to contain the relavant information.
    2. Ranking: Ranking the retrieved passages based on their relevance score.
    3. Generation: Generating an answer based on the retrieved passages. It may use the information from the top ranked passages to generate an answer.
    """
    llm_chain_input_variables = ['context', 'question'] 
    prompt = PromptTemplate(template = custom_prompt_template, input_variables = ['context', 'question'])

    return prompt 


def load_llm():
    # C transformer is a type of transformer built on C, C++ for python - ggml. Why? Bc it is faster.
    ## Alternative: V LLMs 
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type = "llama",
        max_new_tokens = 512, # Max dimension
        temperature = 0.5
    )
    print("LLM LOADED -- FUNC 1 RAN SUCCESSFULLY")
    return llm


# def retrieval_qa_chain(llm, prompt, db):
#     llm_chain_input_variables = ['context', 'question']
#     qa_chain = RetrievalQA.from_chain_type(
#         llm = llm, 
#         chain_type  = "stuff",
#         retriever = db.as_retriever(search_kwargs = {'k': 2}),
#         return_source_documents = True, # use only the data provided and not langchain's knowledge
#         chain_type_kwargs = {'prompt': prompt, 'llm_chain_input_variables': llm_chain_input_variables}
#     ) 
#     return qa_chain

def retrieval_qa_chain(llm, prompt, db):
    print("\n\n----------RAG STARTED----------")
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        #return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    print("RAG FUNC RAN SUCCESSFULLY\n\n")
    return qa_chain


def qa_bot():
    
    print("Qa bot funtion started")
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs = {'device': 'cpu'})

    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization = True)
    llm = load_llm()
    qa_prompt = custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    #print("QA object:", qa)
    print("Qa bot function ran")
    return qa


def final_result(query):
    print("Final result function started")
    qa_result = qa_bot()
    response = qa_result({'query': query})
    print("FINAL RESULT COMPUTED")
    return response

'''

 Fine tuning: 

 1. LORA - 
 2. Define size of your weights - by default it is 32 bit. INT 18.
 

 '''

### CHAINLIT CODE

import chainlit as cl

@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content = 'Starting the bot...')
    await msg.send()
    msg.content = "Hi, Welcome to Sid's world. The ultimate knowledge corpus"
    await msg.update()
    cl.user_session.set("chain", chain)

# Result and responses in chainlit
@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get('chain')
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer = True, answer_prefix_tokens = ["FINAL", "ANSWER"]
    )

    cb.answer_reached = True
    #res = await chain.acall(message.content, callbacks=[cb])
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["result"]
    # sources = res["source_documents"]

    # if sources: 
    #     answer += f'\nSources:' + str(sources)
    # else:
    #     answer += f'\nNo sources found.'

    await cl.Message(content = answer).send()