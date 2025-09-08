from langchain_ollama import OllamaLLM
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Initialize Ollama with the model you want
llm = OllamaLLM(model="gpt-oss:20b")



# Simple prompt
response = llm.invoke("Write me a short poem about a drone flying over Mars.")
print(response)