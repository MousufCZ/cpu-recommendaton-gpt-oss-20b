from langchain_ollama import OllamaLLM
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Initialize Ollama with the model you want
llm = OllamaLLM(model="alibayram/smollm3:latest")



# Simple prompt
response = llm.invoke("Write me a short poem about a drone flying over Mars.")
print(response)