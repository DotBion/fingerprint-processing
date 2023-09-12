#256x364 

import os
import time
import shutil
import subprocess
import cv2
import numpy as np
from walking import walking, walkonce, checkstable, mergeneighbors

file_name = "C:\\Program Files\Mantra\\MFS100\\Driver\\MFS100Test\\FingerData\\FingerImage.bmp"
destination_parent_dir = "D:\\fingerprints\\"
nfiq2_path = "C:\\Program Files\\NFIQ 2\\bin\\nfiq2"

def copy_file(file_name, destination_folder, new_file_name):
    
    #new_name=input("Enter fingerprint name")
    #new_file_name="fp.bmp"
    new_file_name = str(new_file_name) + ".bmp"
    shutil.copy(file_name, destination_folder+new_file_name)
    print(new_file_name," added successfully")
    print()
    #handle errors : same name file
    # os.system(f"cp {file_name} {destination_folder}")

def get_nfiq2(file_name):
    print("Calculating NFIQ2 Score")
    a = subprocess.run([nfiq2_path,file_name], stdout=subprocess.PIPE)
    score = a.stdout.decode('utf-8')
    print("Fingerprint NFIQ2 Score:", score)
    #print(a.stdout.decode('utf-8'))
    return score

def get_core_points(file_name, new_file_name):

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

    destination_path = "D:\\#fp-codes\\fingerprint-processing\\results\\" + new_file_name + ".bmp"
    #print("Dp: ", destination_path)
    cv2.imwrite(destination_path, stacked_img) #make changes here

    print("Core point co-ordinates : ")
    for i in detect_SP['core']:
        print(i)     
    return detect_SP['core']


def track_file_changes(file_name, destination_folder):
    original_file_mtime = os.path.getmtime(file_name)
    #print("file-mtime: ",original_file_mtime)
    print("-----Waiting for Fingerprint-----------")
    print()
    while True:
        # if i==10:
        #         update_directory(file_name)
        
        new_file_mtime = os.path.getmtime(file_name)
        if original_file_mtime != new_file_mtime:
            #print("-----Waiting for Fingerprint-----------")
            new_file_name = "impression_"+str(new_file_mtime)
            print("Fingerprint Scan Complete")
            print()
            #check nfiq score
            score = get_nfiq2(file_name)
            if( score > "30") :
                #calculate core-points
                print("Finding Core Points")
                core_points=get_core_points(file_name,new_file_name)
                if (np.any(core_points)):
                    copy_file(file_name,destination_folder,new_file_name)
                    print("-----Waiting for Fingerprint-----------")
                    print()
                else :
                    print("Core-points not found. Re-enter fingerprint")
                    print("-----Waiting for Fingerprint-----------")
                    print()
            else :
                print("Fingerprint quality low. Re-enter fingerprint")
                print("-----Waiting for Fingerprint-----------")
                print()
            
            original_file_mtime = new_file_mtime
            print()
        time.sleep(1)

if __name__ == "__main__":
    track_file_changes(file_name, destination_parent_dir)