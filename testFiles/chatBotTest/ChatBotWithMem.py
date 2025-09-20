from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_ollama import OllamaLLM
from datetime import datetime
import os

def load_previous_memory(filename):
    """Load a previous conversation from a text file into ConversationBufferMemory."""
    memory = ConversationBufferMemory(return_messages=True)
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("You: "):
                memory.chat_memory.add_user_message(line[len("You: "):].strip())
            elif line.startswith("Bot: "):
                memory.chat_memory.add_ai_message(line[len("Bot: "):].strip())
    return memory

def main():
    # Hardcoded folder path to save conversations
    logs_folder = "/Users/mousuf/ProgProj/oss-hackathon/OssCode/testFiles/chatBotTest/ChatLogs"  # <-- change this to your desired folder
    os.makedirs(logs_folder, exist_ok=True)
    print(f"Conversations will be saved in: {logs_folder}\n")

    # Ask user if they want to continue a previous conversation
    memory = None
    prev_files = [f for f in os.listdir(logs_folder) if f.startswith("conversation_") and f.endswith(".txt")]
    prev_files.sort()
    
    if prev_files:
        print("Previous conversation logs found:")
        for i, f in enumerate(prev_files):
            print(f"{i+1}: {f}")
        choice = input("Enter the number of a file to continue, or press Enter to start fresh: ")
        if choice.isdigit() and 1 <= int(choice) <= len(prev_files):
            filename = os.path.join(logs_folder, prev_files[int(choice)-1])
            memory = load_previous_memory(filename)
            print(f"Loaded conversation from {filename}")
    
    # If no file selected, start with fresh memory
    if memory is None:
        memory = ConversationBufferMemory(return_messages=True)

    # Initialize LLM and conversation chain
    llm = OllamaLLM(model="alibayram/smollm3:latest")
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )

    print("\nChatbot ready! Type 'exit' to quit.\n")

    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            response = conversation.predict(input=user_input)
            print(f"Test Bot: {response}\n")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Exiting chat...")

    finally:
        # Save conversation only once, at the end of the session
        if memory and memory.chat_memory.messages:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(logs_folder, f"conversation_{timestamp}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                for msg in memory.chat_memory.messages:
                    role = "You" if msg.type == "human" else "Bot"
                    f.write(f"{role}: {msg.content}\n")
            print(f"💾 Conversation saved to {filename}")

if __name__ == "__main__":
    main()