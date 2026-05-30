# OpenAI API Demo
# Haochen Li - SMU Computer Engineering

from openai import OpenAI


client = OpenAI(api_key="YOUR_ACTUAL_KEY_HERE")


print("=== Basic Chat ===")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain machine learning in one sentence."}
    ]
)
print(response.choices[0].message.content)

print("\n=== Few-shot Classification ===")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Classify sentiment as Positive or Negative."},
        {"role": "user", "content": """
Examples:
Text: The product is amazing! -> Positive
Text: Terrible experience, very slow. -> Negative
Text: Great customer service! -> Positive

Now classify:
Text: The delivery was fast but packaging was damaged.
"""}
    ]
)
print(response.choices[0].message.content)


print("\n=== Chain of Thought ===")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": """
Think step by step and solve this:
A data scientist processes 500 records per hour.
If she works 8 hours a day for 5 days,
how many records does she process in total?
"""}
    ]
)
print(response.choices[0].message.content)


print("\n=== Simple RAG Demo - SMU Coursework ===")

smu_doc = """
Haochen Li - SMU Graduate Coursework (M.S. Computer Engineering, GPA 3.8):

Fall 2024:
- ECE 7375 Random Processes in Engineering (Grade: A)

Spring 2025:
- ECE 7356 VLSI Design and Lab (Grade: A)
- ECE 7381 Computer Architecture (Grade: A)
- ECE 7385 Microcontroller Architecture and Interfacing (Grade: A-)

Fall 2025:
- CS 7343 Operating Systems (Grade: A-)
- CS 7344 Computer Networks and Distributed Systems (Grade: A-)
- ECE 7387 Digital Systems Design (Grade: A-)

Spring 2026:
- CS 7324 Machine Learning in Python (Grade: A-)
- ECE 7349 Data and Network Security (Grade: A-)
- ECE 8371 Information Theory (Grade: A)

Cumulative GPA: 3.82
"""

questions = [
    "What machine learning courses did Haochen take?",
    "What was Haochen's GPA and strongest subjects?",
    "Is Haochen qualified for an AI Developer role?"
]

for question in questions:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"""
Answer questions based only on this academic record:
{smu_doc}
Be concise and professional.
"""},
            {"role": "user", "content": question}
        ]
    )
    print(f"Q: {question}")
    print(f"A: {response.choices[0].message.content}")
    print()

print("\n=== Done! ===")