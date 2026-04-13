QA_TEMPLATE = """You are a helpful assistant answering questions based ONLY on the provided context.
if the context does not contain enough information to answer , say "I don't have enough information to answer this question."
Do not make up information beyond what is provided in the context.

Context:
{context}

Question: {question}

Answer:"""