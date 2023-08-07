# import tkinter as tk

# def get_input():
#     window = tk.Tk()
#     label = tk.Label(window, text="Please enter the filename of the file that has changed: ")
#     entry = tk.Entry(window, name="filename")
#     button = tk.Button(window, text="Submit", command=window.destroy)

#     label.pack()
#     entry.pack()
#     button.pack()

#     window.mainloop()
#     fn=entry["filename"]
#     print("fn: ", fn)

#     return entry.get()

# if __name__ == "__main__":
#     file_name = get_input()
#     print(f"The filename of the changed file is: {file_name}")
#------------------------------------------------------------------------
# from tkinter import *
# from functools import partial

# def printDetails(usernameEntry) :
#     usernameText = usernameEntry.get()
#     print("user entered :", usernameText)
#     return

# #window
# tkWindow = Tk()  
# tkWindow.geometry('400x150')  
# tkWindow.title('Python Examples')

# #label
# usernameLabel = Label(tkWindow, text="Enter your name")
# #entry for user input
# usernameEntry = Entry(tkWindow)

# #define callable function with printDetails function and usernameEntry argument
# printDetailsCallable = partial(printDetails, usernameEntry)

# #submit button
# submitButton = Button(tkWindow, text="Submit", command=printDetailsCallable)

# #place label, entry, and button in grid
# usernameLabel.grid(row=0, column=0)
# usernameEntry.grid(row=0, column=1) 
# submitButton .grid(row=1, column=1)  

# #main loop
# tkWindow.mainloop()
#-----------------------------------------------------------------------------
def getNewFileName():
    import tkinter as tk
    window=tk.Tk()
    # setting the windows size
    window.geometry("600x400")
    name_var=tk.StringVar()
    name=""
    def submit():
        name=name_var.get()
        print("The name is : " + name)
        window.destroy()	
    name_label = tk.Label(window, text = 'Username', font=('calibre',10, 'bold'))
    name_entry = tk.Entry(window,textvariable = name_var, font=('calibre',10,'normal'))
    sub_btn=tk.Button(window,text = 'Submit', command = submit)

    name_label.grid(row=0,column=0)
    name_entry.grid(row=0,column=1)
    sub_btn.grid(row=2,column=1)
    print("The name is 2:" + name)
    window.mainloop()
    return name
if __name__ == "__main__":
    file_name = getNewFileName()
    print(f"The filename of the changed file is: {file_name}")
#-----------------------------------------------------------------------------------
# import tkinter as tk
# #def submit():
# #     name=name_var.get()
#  #   print("added")
# #     name_var.set("")	
# def get_input():
#     window = tk.Tk()
#     name_var=tk.StringVar()
#     label = tk.Label(window, text="Please enter the filename of the file that has changed: ")
#     entry = tk.Entry(window, textvariable = name_var)
#     #button = tk.Button(window, text="Submit", command=window.destroy)
#     button = tk.Button(window, text="Submit", command=window.destroy)
    
#     name=name_var.get()
#     print("The name is : " + name)

#     label.pack()
#     entry.pack()
#     button.pack()

#     window.mainloop()
#     fn=entry["filename"]
#     print("fn: ", fn)

#     return name

# if __name__ == "__main__":
#     file_name = get_input()
#     print(f"The filename of the changed file is: {file_name}")