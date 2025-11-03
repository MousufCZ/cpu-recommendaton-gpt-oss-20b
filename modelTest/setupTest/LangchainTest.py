from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import OllamaLLM
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

llm = OllamaLLM(model="alibayram/smollm3:latest")

template = "You are a helpful assistant. Translate the following English text to French: {text}"

prompt = PromptTemplate(
    input_variables=["text"],
    template=template,
)

chain = LLMChain(llm=llm, prompt=prompt)

result = chain.run("The drone is flying over a red rocky landscape.")
print(result)
