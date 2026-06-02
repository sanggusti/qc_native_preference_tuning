SLIGHT_FACTUAL_ERROR = """You are given a response from an LLM system. You are also given the context
which was used to generate the answer. Can you add a slight factual error in the response. Don't change the answer too much.
Don't change the length of the answer. Just slightly add slight factual error. The factual error should not be noticeable easily.
Context: {context}
Answer: {answer}
"""