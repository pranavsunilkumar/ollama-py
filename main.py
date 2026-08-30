from langchain_ollama import ChatOllama 
# OllamaLLM-> print(response) or ChatOllama -> use print(response.content)

def create_llm():
    llm = ChatOllama(
        model="qwen3:0.6b",
        temperature=0.2
    )
    return llm

if __name__ == "__main__":
    user_input = input("Enter your question:")
    llm=create_llm()
    response=llm.invoke(user_input)
    print(response.content)