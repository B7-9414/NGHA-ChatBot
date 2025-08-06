# # chains.py
# from langchain_openai import ChatOpenAI
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# def build_llm():
#     return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

# def build_prompts():
#     search_prompt = ChatPromptTemplate.from_messages([
#         MessagesPlaceholder("chat_history"),
#         ("user", "{input}"),
#         ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation.")
#     ])
#     answer_prompt = ChatPromptTemplate.from_messages([
#         ("system", "Answer the user's questions based on the context below. Always be concise and relevant.\n\n{context}"),
#         MessagesPlaceholder("chat_history"),
#         ("user", "{input}")
#     ])
#     return search_prompt, answer_prompt

# def build_rag_chain(llm, retriever, search_prompt, answer_prompt):
#     retriever_chain = create_history_aware_retriever(llm=llm, retriever=retriever, prompt=search_prompt)
#     stuff_chain = create_stuff_documents_chain(llm=llm, prompt=answer_prompt)
#     return create_retrieval_chain(retriever_chain, stuff_chain)
