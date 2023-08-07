import os
import sys
import time
import shutil
import tkinter as tk

def getNewFileName():
    window=tk.Tk()
    # setting the windows size
    window.geometry("600x400")
    name_var=tk.StringVar()
    name=""
    def submit():
        name=name_var.get()
        print("The name is : " + name)
        name_var.set("")
        window.destroy()	
    name_label = tk.Label(window, text = 'Username', font=('calibre',10, 'bold'))
    name_entry = tk.Entry(window,textvariable = name_var, font=('calibre',10,'normal'))
    sub_btn=tk.Button(window,text = 'Submit', command = submit)

    name_label.grid(row=0,column=0)
    name_entry.grid(row=0,column=1)
    sub_btn.grid(row=2,column=1)
    print("The name is 2: " + name)
    window.mainloop()
    
def track_file_changes(file_name, destination_folder):
    original_file_mtime = os.path.getmtime(file_name)
    print("file-mtime: ",original_file_mtime)
    while True:
        new_file_mtime = os.path.getmtime(file_name)
        if original_file_mtime != new_file_mtime:
            new_file_name=getNewFileName()
            copy_file(file_name, destination_folder, original_file_mtime)
            original_file_mtime = new_file_mtime
        time.sleep(1)

def copy_file(file_name, destination_folder, new_file_name):
    
    #new_name=input("Enter fingerprint name")
    #new_file_name="fp.bmp"
    new_file_name = str(new_file_name) + ".bmp"
    file_name = sys.stdin.readline().strip()
    print("filename: ",file_name)
    
    shutil.copy(file_name, destination_folder+new_file_name)
    print("Files are copied successfully")

    #handle errors : same name file
    # os.system(f"cp {file_name} {destination_folder}")

if __name__ == "__main__":
    file_name = "C:\\Program Files\Mantra\\MFS100\\Driver\\MFS100Test\\FingerData\\FingerImage.bmp"
    destination_folder = "D:\\fingerprints\\"
    track_file_changes(file_name, destination_folder)
