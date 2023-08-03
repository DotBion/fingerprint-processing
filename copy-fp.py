# importing the modules
import os
import shutil

# Providing the folder path
origin = 'C:\\Program Files\\Mantra\\MFS100\\Driver\\MFS100Test\\FingerData\\FingerImage.bmp'
target = 'D:\\fingerprints\\'

# Fetching the list of all the files
# files = os.listdir(origin)

#new_name=input("Enter fingerprint name")
new_name="fp.bmp"

shutil.copy(origin, target+new_name)
print("Files are copied successfully")

# Fetching all the files to directory
# for file_name in files:
#    shutil.copy(origin+file_name, target+file_name)
# print("Files are copied successfully")