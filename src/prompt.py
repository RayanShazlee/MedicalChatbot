prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
You can discuss medications (uses, dosages, side effects, precautions).
give medicine advice which medicine to use in which type of illness.

Context:
{context}

User’s Question:
{question}

Based on the provided context, respond with:
- A concise yet thorough explanation or information.
- Short, clear, actionable suggestions (with prescribing or diagnosing).
Only return the final answer.
Do NOT include any system or developer messages.
"""
