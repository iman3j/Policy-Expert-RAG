from Backend.rag_chain import ask_multimodal

query = "Show workflow diagram for leave approval"
texts, images = ask_multimodal(query)

print("Texts:", texts)
print("Images:", images)
