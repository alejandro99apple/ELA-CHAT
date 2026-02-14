import tkinter as tk
from tkinter import scrolledtext
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI
import os
from dotenv import load_dotenv


class ChatUI(tk.Tk):
    def __init__(self, retriever=None, client=None):
        super().__init__()
        self.retriever = retriever
        self.client = client
        self.chat_history = []

        self.title("RAG Chat - ELA Assistant")
        self.geometry("720x520")
        self.minsize(930, 520)
        self.resizable(True, True)

        self.configure(bg="#0f0f0f")

        self.has_started = False
        self.spinner_running = False
        self.spinner_frames = ['|', '/', '—', '\\']
        self.spinner_index = 0
        self.spinner_after_id = None
        self.assistant_message_line = None

        self.chat_display = scrolledtext.ScrolledText(
            self,
            wrap=tk.WORD,
            state=tk.DISABLED,
            bg="#0f0f0f",
            fg="#e6e6e6",
            insertbackground="#00d9ff",
            font=("Segoe UI", 11),
            padx=15,
            pady=15,
            bd=0,
        )
        self.chat_display.tag_configure("user", foreground="#00d9ff", font=("Segoe UI", 10, "bold"))
        self.chat_display.tag_configure("assistant", foreground="#10b981", font=("Segoe UI", 10, "bold"))

        self.welcome_label = tk.Label(
            self,
            text=(
                "🎓 Hello! I'm your ALS virtual assistant.\n\n"
                "I'll help you with your thesis by answering all your questions.\n"
                "Where should we start?"
            ),
            bg="#0f0f0f",
            fg="#10b981",
            font=("Segoe UI", 12, "bold"),
            wraplength=550,
            justify="center",
        )
        self.welcome_label.place(relx=0.5, rely=0.45, anchor="center")

        input_frame = tk.Frame(self, bg="#0f0f0f", height=90)
        input_frame.pack_propagate(False)
        self.input_text = tk.Text(
            input_frame,
            height=1,
            wrap=tk.WORD,
            bg="#1a1a1a",
            fg="#e6e6e6",
            insertbackground="#00d9ff",
            font=("Segoe UI", 11),
            bd=1,
            relief=tk.FLAT,
            padx=12,
            pady=10,
        )
        self.input_text.configure(highlightbackground="#10b981", highlightthickness=1)
        self.input_text.bind("<Return>", self._on_enter)
        self.input_text.bind("<Shift-Return>", self._on_shift_enter)

        send_button = tk.Button(
            input_frame,
            text="➤ Send",
            command=self.on_send,
            bg="#10b981",
            fg="#ffffff",
            activebackground="#059669",
            activeforeground="#ffffff",
            bd=0,
            padx=20,
            pady=8,
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
            cursor="hand2"
        )

        input_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=0, pady=0)
        self.input_text.pack(fill=tk.BOTH, side=tk.LEFT, expand=True, padx=(12, 8), pady=12)
        send_button.pack(side=tk.RIGHT, padx=(0, 12), pady=12)

    def _on_enter(self, event):
        self.on_send()
        return "break"

    def _on_shift_enter(self, event):
        return None


    # SEND ----------------------------------------------------- 
    def on_send(self):
        user_text = self.input_text.get("1.0", tk.END).strip()
        if not user_text:
            return

        if not self.has_started:
            self._start_chat()

        self._append_message("You", user_text, "user")
        self.input_text.delete("1.0", tk.END)
        

        
        # History context
        if self.chat_history:
            
            messages = [{
                "role": "user",
                "content": f""" Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."
                Chat history:
                {self.chat_history}

                New question:
                {user_text}
                """
            }]

            completion = self.client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct:groq",
                messages=messages,
                temperature=0.2,
                max_tokens=300,
            )

            user_text = completion.choices[0].message.content.strip().strip('"')
            print(f"Rewritten question: {user_text}")
            relevant_docs = self.retriever.invoke(user_text)
            context = "\n\n".join(doc.page_content for doc in relevant_docs)
            
        else:
            
            # Get relevant documents using RAG retriever
            relevant_docs = self.retriever.invoke(user_text)
            context = "\n\n".join(doc.page_content for doc in relevant_docs)

        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant specialized in answering questions about the provided context. Answer ONLY using the provided context. If the answer is not in the context, say 'I don't know based on the provided documents.'"
            },
            {
                "role": "user",
                "content": f"""
                    
                Question:
                {user_text}

                Context:
                {context}
                """
            }
        ]
            
        completion = self.client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct:groq",
            messages=messages,
            temperature=0.2,
            max_tokens=300,
        )

            
        self.chat_history.append(f"User: {user_text} \nAssistant: {completion.choices[0].message.content}")

        # Placeholder response. Replace with RAG call.
        response_text = completion.choices[0].message.content
        self._append_message("Assistant", response_text, "assistant")

    def _append_message(self, sender, message, tag):
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{sender}: ", tag)
        self.chat_display.insert(tk.END, message + "\n\n")
        self.chat_display.configure(state=tk.DISABLED)
        self.chat_display.yview(tk.END)

    def _start_chat(self):
        self.has_started = True
        self.welcome_label.place_forget()
        self.chat_display.pack(fill=tk.BOTH, expand=True)

def load_environment():
    load_dotenv()
    persistent_directory = "db/chroma_db"
    
    embedding_model = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 7}
    )
    
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ["HF_TOKEN"],
    )
    
    return retriever, client




def main():
    retriever, client = load_environment()
    app = ChatUI(retriever=retriever, client=client)
    app.mainloop()


if __name__ == "__main__":
    main()
