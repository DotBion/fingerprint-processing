import os
import time
import shutil

def track_file_changes(file_name, destination_folder):
    original_file_mtime = os.path.getmtime(file_name)
    print("file-mtime: ",original_file_mtime)
    while True:
        new_file_mtime = os.path.getmtime(file_name)
        if original_file_mtime != new_file_mtime:
            copy_file(file_name, destination_folder, original_file_mtime)
            original_file_mtime = new_file_mtime
        time.sleep(1)

def copy_file(file_name, destination_folder, new_file_name):
    
    #new_name=input("Enter fingerprint name")
    #new_file_name="fp.bmp"
    new_file_name = str(new_file_name) + ".bmp"
    shutil.copy(file_name, destination_folder+new_file_name)
    print("Files are copied successfully")

    #handle errors : same name file
    # os.system(f"cp {file_name} {destination_folder}")

if __name__ == "__main__":
    file_name = "C:\\Program Files\Mantra\\MFS100\\Driver\\MFS100Test\\FingerData\\FingerImage.bmp"
    destination_folder = "D:\\fingerprints\\"
    track_file_changes(file_name, destination_folder)

