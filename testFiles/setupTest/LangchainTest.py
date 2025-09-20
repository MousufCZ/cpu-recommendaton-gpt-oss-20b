from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import OllamaLLM
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Initialize Ollama with the model I want
llm = OllamaLLM(model="alibayram/smollm3:latest")

# Define a template
template = "You are a helpful assistant. Translate the following English text to French: {text}"

prompt = PromptTemplate(
    input_variables=["text"],
    template=template,
)

# Create a chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run chain
result = chain.run("The drone is flying over a red rocky landscape.")
print(result)
