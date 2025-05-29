import tkinter as tk
from tkinter import scrolledtext, PhotoImage, PanedWindow
import os
import subprocess # For opening files on macOS/Linux
import sys # To check platform

class ChatGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Offline RAG Chatbot")
        self.window.geometry("1250x700")

        try:
            script_dir = os.path.dirname(__file__)
            icon_path_png = os.path.join(script_dir, "icon.png")
            icon_path_ico = os.path.join(script_dir, "icon.ico")

            if os.path.exists(icon_path_png):
                img = PhotoImage(file=icon_path_png)
                self.window.tk.call('wm', 'iconphoto', self.window._w, img)
            elif os.path.exists(icon_path_ico):
                self.window.iconbitmap(icon_path_ico)
        except Exception as e:
            print(f"Could not set window icon: {e}")

        self.main_pane = PanedWindow(self.window, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=6)
        self.main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.left_frame = tk.Frame(self.main_pane, relief=tk.SUNKEN, borderwidth=1)
        
        retrieved_label = tk.Label(self.left_frame, text="Retrieved Context:", font=("Arial", 11, "bold"), anchor="w")
        retrieved_label.pack(side=tk.TOP, fill="x", padx=5, pady=(5,2))
        
        self.text_retrieved = scrolledtext.ScrolledText(self.left_frame, width=40, height=25, wrap=tk.WORD,
                                                        bg="#f0f0f0", fg="black", relief=tk.FLAT,
                                                        font=("Arial", 9))
        self.text_retrieved.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=(0,5))
        self.text_retrieved.config(state=tk.DISABLED)
        
        # No general tag_bind for <Button-1> here for "source_link"
        # We only configure its appearance and hover effects.
        self.text_retrieved.tag_configure("source_link_style", foreground="blue", underline=True) # Renamed for clarity
        self.text_retrieved.tag_bind("source_link_style", "<Enter>", lambda e: self.text_retrieved.config(cursor="hand2"))
        self.text_retrieved.tag_bind("source_link_style", "<Leave>", lambda e: self.text_retrieved.config(cursor=""))
        
        self.main_pane.add(self.left_frame)

        self.right_frame = tk.Frame(self.main_pane)
        self.chat_history = scrolledtext.ScrolledText(self.right_frame, width=70, height=25, wrap=tk.WORD,
                                                     bg="#ffffff", fg="black", relief=tk.SUNKEN, borderwidth=1,
                                                     font=("Arial", 10), padx=5, pady=5)
        self.chat_history.pack(fill=tk.BOTH, expand=True, pady=(0,5))
        self.chat_history.config(state=tk.DISABLED)
        
        self.chat_history.tag_configure("user_label", foreground="#005b96", font=("Arial", 10, "bold"))
        self.chat_history.tag_configure("user_text", foreground="#333333", font=("Arial", 10), lmargin1=20, lmargin2=20)
        self.chat_history.tag_configure("bot_label", foreground="#006400", font=("Arial", 10, "bold"))
        self.chat_history.tag_configure("bot_text", foreground="#111111", font=("Arial", 10), lmargin1=20, lmargin2=20)
        self.chat_history.tag_configure("error_label", foreground="#dc3545", font=("Arial", 10, "bold"))
        self.chat_history.tag_configure("error_text", foreground="#721c24", font=("Arial", 10, "italic"), lmargin1=20, lmargin2=20)
        self.chat_history.tag_configure("system_message", foreground="#555555", font=("Arial", 9, "italic"), justify=tk.CENTER)

        self.input_area_frame = tk.Frame(self.right_frame, pady=5)
        self.input_area_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.chat_entry = tk.Entry(self.input_area_frame, font=("Arial", 11), relief=tk.SOLID, borderwidth=1)
        self.chat_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6, padx=(0,10))
        self.chat_entry.bind("<Return>", self.process_chat_input_event)

        self.send_button = tk.Button(self.input_area_frame, text="Send", command=self.process_chat_input, 
                                     font=("Arial", 10, "bold"), width=8, height=1, 
                                     bg="#007bff", fg="white", relief=tk.RAISED, activebackground="#0056b3")
        self.send_button.pack(side=tk.RIGHT)
        
        self.main_pane.add(self.right_frame)
        self.window.update_idletasks()
        try:
            initial_left_width = int(self.window.winfo_width() * 0.35)
            if initial_left_width > 0 :
                 self.main_pane.sash_place(0, initial_left_width, 0)
        except tk.TclError:
            self.window.after(100, lambda: self.main_pane.sash_place(0, int(self.window.winfo_width() * 0.35), 0))

        self.query_callback = None
        self.chat_entry.focus_set()
        self._append_to_chat("System", "Welcome! Ask me anything about your documents.", "system_message", is_system=True)

    # This helper event handler is removed as open_file_path is called directly by unique tag binds
    # def _open_file_path_event(self, event, file_path):
    #     self.open_file_path(file_path)

    def open_file_path(self, file_path_to_open):
        if file_path_to_open and os.path.exists(file_path_to_open):
            print(f"Attempting to open file: {file_path_to_open}")
            try:
                normalized_path = os.path.normpath(file_path_to_open)
                if sys.platform == "win32":
                    os.startfile(normalized_path)
                elif sys.platform == "darwin": 
                    subprocess.run(["open", normalized_path], check=True)
                else: 
                    subprocess.run(["xdg-open", normalized_path], check=True)
            except FileNotFoundError:
                 error_msg = f"Error: Source file not found at path: {normalized_path}"
                 print(error_msg)
                 self._append_to_chat("System", error_msg, "error_text", is_system=True)
            except Exception as e:
                error_msg = f"Error opening file '{normalized_path}': {e}"
                print(error_msg)
                self._append_to_chat("System", error_msg, "error_text", is_system=True)
        elif file_path_to_open:
            error_msg = f"Error: File path does not exist: {file_path_to_open}"
            print(error_msg)
            self._append_to_chat("System", error_msg, "error_text", is_system=True)
        else:
            print("No valid file path provided to open.")

    def _append_to_chat(self, sender, message, tag_text_name, is_system=False):
        self.chat_history.config(state=tk.NORMAL)
        if self.chat_history.index('end-1c') != "1.0": 
            self.chat_history.insert(tk.END, "\n\n")
        if not is_system:
            self.chat_history.insert(tk.END, f"{sender}:\n", (tag_text_name.replace("_text", "_label"),))
            self.chat_history.insert(tk.END, message, (tag_text_name,))
        else: 
            self.chat_history.insert(tk.END, message, (tag_text_name,), "\n")
        self.chat_history.see(tk.END)
        self.chat_history.config(state=tk.DISABLED)

    def process_chat_input_event(self, event=None):
        self.process_chat_input()

    def process_chat_input(self):
        query = self.chat_entry.get()
        if not query.strip(): return

        self._append_to_chat("You", query, "user_text")
        self.chat_entry.delete(0, tk.END)

        if self.query_callback:
            self.send_button.config(state=tk.DISABLED, text="...")
            self.chat_entry.config(state=tk.DISABLED)
            self.window.update_idletasks()
            try:
                retrieved_items_structured, answer = self.query_callback(query)
                
                self.text_retrieved.config(state=tk.NORMAL)
                self.text_retrieved.delete(1.0, tk.END)

                if retrieved_items_structured:
                    for idx, item in enumerate(retrieved_items_structured): # Use idx for unique tag names
                        source_display_text = item.get('source_display', 'Unknown Source')
                        similarity_score = item.get('similarity', 0.0)
                        chunk_text = item.get('text', '')
                        full_path = item.get('full_file_path', None)

                        # Store current end position BEFORE inserting the source line
                        # This will be the start of the clickable text
                        if self.text_retrieved.index(tk.END + "-1c") != "1.0":
                            self.text_retrieved.insert(tk.END, "\n\n")
                        
                        source_line_start_index = self.text_retrieved.index(tk.END + "-1c")
                        
                        source_info_line = f"From {source_display_text} (Similarity: {similarity_score:.2f}):"
                        self.text_retrieved.insert(tk.END, source_info_line)
                        
                        if full_path:
                            source_line_end_index = self.text_retrieved.index(tk.END + "-1c")
                            unique_tag_name = f"source_link_{idx}" # Create a unique tag for each link

                            # Apply the styling tag
                            self.text_retrieved.tag_add("source_link_style", source_line_start_index, source_line_end_index)
                            
                            # Add the unique tag specifically for this link's range
                            self.text_retrieved.tag_add(unique_tag_name, source_line_start_index, source_line_end_index)
                            
                            # Bind the click event ONLY to this unique tag
                            # The lambda correctly captures 'full_path' for this specific binding
                            self.text_retrieved.tag_bind(unique_tag_name, "<Button-1>", 
                                                         lambda e, p=full_path: self.open_file_path(p))
                        
                        self.text_retrieved.insert(tk.END, f"\n{chunk_text}")
                else:
                    self.text_retrieved.insert(tk.END, "No specific context retrieved.")
                
                self.text_retrieved.config(state=tk.DISABLED)
                self._append_to_chat("Bot", answer, "bot_text")
            except Exception as e:
                error_message = f"An error occurred: {e}"
                self._append_to_chat("System", error_message, "error_text", is_system=True)
                print(f"Error during query processing: {error_message}") 
                import traceback
                traceback.print_exc()
            finally:
                self.send_button.config(state=tk.NORMAL, text="Send")
                self.chat_entry.config(state=tk.NORMAL)
                self.chat_entry.focus_set()
        else:
            self._append_to_chat("System", "Query callback not set. Application error.", "error_text", is_system=True)

    def set_query_callback(self, func):
        self.query_callback = func

    def start(self):
        self.window.mainloop()

def display_interface(handle_query_function):
    gui = ChatGUI()
    gui.set_query_callback(handle_query_function)
    gui.start()

if __name__ == '__main__':
    def dummy_handle_query(query):
        import time
        time.sleep(0.5)
        script_dir = os.path.dirname(__file__)
        # Create unique dummy file names for testing to avoid potential conflicts if run multiple times
        dummy_pdf_path = os.path.abspath(os.path.join(script_dir, f"dummy_doc_{time.time()}.pdf"))
        dummy_txt_path = os.path.abspath(os.path.join(script_dir, f"another_src_{time.time()}.txt"))

        if not os.path.exists(dummy_pdf_path):
            with open(dummy_pdf_path, "w") as f: f.write(f"Content from {os.path.basename(dummy_pdf_path)} related to '{query}'.")
        if not os.path.exists(dummy_txt_path):
            with open(dummy_txt_path, "w") as f: f.write(f"Content from {os.path.basename(dummy_txt_path)} also about '{query}'.")
            
        retrieved_items_structured = [
            {'text': "This is the first dummy chunk. It simulates retrieved content based on your query about various topics.", 
             'source_display': f"{os.path.basename(dummy_pdf_path)} (section 1)", 
             'similarity': 0.95, 'full_file_path': dummy_pdf_path},
            {'text': "Another relevant paragraph. This one is designed to show how different sources can be linked and opened.", 
             'source_display': f"{os.path.basename(dummy_txt_path)} (section 5)", 
             'similarity': 0.88, 'full_file_path': dummy_txt_path},
            {'text': "A third piece of context. Sometimes context might be less direct but still offer supporting details.", 
             'source_display': f"{os.path.basename(dummy_pdf_path)} (section 2)", 
             'similarity': 0.75, 'full_file_path': dummy_pdf_path}
        ]
        answer_text = f"Okay, regarding '{query}':\n"
        if "hello" in query.lower(): answer_text += "Hello there! This is a smart RAG chatbot, ready to help."
        elif "how are you" in query.lower(): answer_text += "I'm a set of algorithms, functioning optimally! Ask away."
        else: answer_text += "This is a synthesized answer based on the dummy context. It would normally be more detailed."
        return retrieved_items_structured, answer_text
    
    display_interface(dummy_handle_query)