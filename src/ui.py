import tkinter as tk

def create_ui():
    root = tk.Tk()
    root.title("Modern Black and White UI")

    # Set the background color to black
    root.configure(bg='black')

    # Create a label with white text
    label = tk.Label(root, text="Welcome to the Modern UI", fg='white', bg='black', font=("Helvetica", 16))
    label.pack(pady=20)

    # Create a button with white text and black background
    button = tk.Button(root, text="Click Me", fg='white', bg='black', font=("Helvetica", 14), relief='flat')
    button.pack(pady=10)

    # Run the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    create_ui()
