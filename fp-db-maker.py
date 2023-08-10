import os
import time
import shutil

file_name = "C:\\Program Files\Mantra\\MFS100\\Driver\\MFS100Test\\FingerData\\FingerImage.bmp"
destination_parent_dir = "D:\\fingerprints\\"

def update_directory(file_name,destination_parent_dir="D:\\fingerprints\\"):
    dir_name=input('Enter User Name')
    #parent_dir = "D:\\fingerprints\\"
    path = os.path.join(destination_parent_dir, dir_name)
    os.mkdir(path)
    print("Directory '% s' created" % dir_name)
    destination_folder = destination_parent_dir+dir_name+"\\"
    track_file_changes(file_name, destination_folder)

def track_file_changes(file_name, destination_folder):
    i=0
    original_file_mtime = os.path.getmtime(file_name)
    print("file-mtime: ",original_file_mtime)
    while True:
        if i==10:
                update_directory(file_name)
        new_file_mtime = os.path.getmtime(file_name)
        if original_file_mtime != new_file_mtime:
            i+=1
            new_file_name = "impression"+str(i)
            copy_file(file_name, destination_folder, new_file_name)
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
    update_directory(file_name, destination_parent_dir)
    