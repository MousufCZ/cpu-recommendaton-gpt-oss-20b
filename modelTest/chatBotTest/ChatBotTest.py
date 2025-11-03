from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_ollama import OllamaLLM

def main():
    llm = OllamaLLM(model="alibayram/smollm3:latest")
    memory = ConversationBufferMemory(return_messages=True)

    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )

    print("Chatbot ready! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")

            with open("conversation_log.txt", "w", encoding="utf-8") as f:
                for msg in memory.chat_memory.messages:
                    role = "You" if msg.type == "human" else "Bot"
                    f.write(f"{role}: {msg.content}\n")
            print("Conversation saved to conversation_log.txt")
            break

        response = conversation.predict(input=user_input)
        print(f"Test Bot: {response}\n")

if __name__ == "__main__":
    main()