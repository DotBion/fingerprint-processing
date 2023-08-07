import tkinter as tk
from tkinter import messagebox

def get_input():
    message = messagebox.askstring("Enter a message", "Please enter a message: ")
    print(f"The message entered by the user is: {message}")

if __name__ == "__main__":
    get_input()
