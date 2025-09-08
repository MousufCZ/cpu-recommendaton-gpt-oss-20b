from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="gpt-oss:20b")
memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

conversation.predict(input="Hello, who are you?")
conversation.predict(input="Can you write me a story about a drone explorer?")
