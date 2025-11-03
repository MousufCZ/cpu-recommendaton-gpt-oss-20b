from langchain_ollama import OllamaLLM
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

llm = OllamaLLM(model="alibayram/smollm3:latest")

response = llm.invoke("Write me a short poem about a drone flying over Mars.")
print(response)