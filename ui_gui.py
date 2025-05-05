import tkinter as tk
from tkinter import scrolledtext

class ChatGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("RAG Chatbot")
        self.window.geometry("1200x600")

        # Question input
        self.label_question = tk.Label(self.window, text="Your Question:")
        self.label_question.pack(pady=5)
        self.entry_question = tk.Entry(self.window, width=120)
        self.entry_question.pack(pady=5)
        self.ask_button = tk.Button(self.window, text="Ask", command=self.process_question)
        self.ask_button.pack(pady=5)

        # Split panels
        self.frame = tk.Frame(self.window)
        self.frame.pack(fill="both", expand=True)

        # Left: retrieved
        self.text_retrieved = scrolledtext.ScrolledText(self.frame, width=70, height=30, bg="black", fg="yellow")
        self.text_retrieved.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        # Right: answer
        self.text_answer = scrolledtext.ScrolledText(self.frame, width=70, height=30, bg="black", fg="lightgreen")
        self.text_answer.pack(side="right", fill="both", expand=True, padx=5, pady=5)

    def process_question(self):
        query = self.entry_question.get()
        if self.query_callback:
            retrieved, answer = self.query_callback(query)
            self.text_retrieved.delete(1.0, tk.END)
            self.text_retrieved.insert(tk.END, retrieved)
            self.text_answer.delete(1.0, tk.END)
            self.text_answer.insert(tk.END, answer)

    def set_query_callback(self, func):
        self.query_callback = func

    def start(self):
        self.window.mainloop()

def display_interface(handle_query_function):
    gui = ChatGUI()
    gui.set_query_callback(handle_query_function)
    gui.start()
