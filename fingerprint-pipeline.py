import os
import time
import shutil
import subprocess
import cv2
import numpy as np
from walking import walking, walkonce, checkstable, mergeneighbors

file_name = "C:\\Program Files\Mantra\\MFS100\\Driver\\MFS100Test\\FingerData\\FingerImage.bmp"
destination_parent_dir = "D:\\fingerprints\\"

def track_file_changes(file_name, destination_folder):
    original_file_mtime = os.path.getmtime(file_name)
    #print("file-mtime: ",original_file_mtime)
    while True:
        # if i==10:
        #         update_directory(file_name)
        new_file_mtime = os.path.getmtime(file_name)
        if original_file_mtime != new_file_mtime:
            new_file_name = "impression_"+str(new_file_mtime)
            #check nfiq score
            #destination_path = destination_parent_dir + new_file_name use this after copy

            a = subprocess.run(["C:\\Program Files\\NFIQ 2\\bin\\nfiq2",file_name], stdout=subprocess.PIPE)
            print(a.stdout.decode('utf-8'))

            #check singular points co-ord
            im = cv2.imread(file_name,0) #make changes here

            stacked_img = np.stack((im,)*3, axis=-1)

            detect_SP = walking(im)

            if min(detect_SP['core'].shape) !=0:
                for i in range(0, detect_SP['core'].shape[0]):
                    centre = (int(detect_SP['core'][i,0]), int(detect_SP['core'][i,1]))
                    stacked_img = cv2.circle(stacked_img, centre, 10, (0,0,255), 2)

            if min(detect_SP['delta'].shape) !=0:
                for j in range(0, detect_SP['delta'].shape[0]):
                    x = int(detect_SP['delta'][j,0])
                    y = int(detect_SP['delta'][j,1])
                    pts = np.array([[x,y-10], [x-9,y+5], [x+9,y+5]])
                    stacked_img = cv2.polylines(stacked_img, [pts], True, (0,255,0), 2)

            destination_path = "D:\\#fp-codes\\fingerprint-processing\\results\\" + new_file_name
            cv2.imwrite(destination_path, stacked_img) #make changes here

            print(detect_SP)

            #if both exists save file else retake

            
            copy_file(file_name, destination_folder, new_file_name)
            original_file_mtime = new_file_mtime
        time.sleep(1)

















def update_directory(file_name,destination_parent_dir="D:\\fingerprints\\"):
    dir_name=input('Enter User Name: ')
    #parent_dir = "D:\\fingerprints\\"
    path = os.path.join(destination_parent_dir, dir_name)
    os.mkdir(path)
    print("Directory '% s' created" % dir_name)
    destination_folder = destination_parent_dir+dir_name+"\\"
    track_file_changes(file_name, destination_folder)

def track_file_changes(file_name, destination_folder):
    i=0
    original_file_mtime = os.path.getmtime(file_name)
    #print("file-mtime: ",original_file_mtime)
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
    print(new_file_name," added successfully")

    #handle errors : same name file
    # os.system(f"cp {file_name} {destination_folder}")

if __name__ == "__main__":
    update_directory(file_name, destination_parent_dir)
    