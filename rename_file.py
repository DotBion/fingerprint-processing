# import os

# old_name = "C:\Program Files\Mantra\MFS100\Driver\MFS100Test\FingerData\FingerImage.bmp"
# new_name = "C:\Program Files\Mantra\MFS100\Driver\MFS100Test\FingerData\FingerImage1.bmp"

# if os.path.isfile(new_name):
#     print("The file already exists")
# else:
#     # Rename the file
#     os.rename(old_name, new_name)

import os

def rename_file_with_admin_privileges(old_file_path, new_file_path):
    os.rename(old_file_path, new_file_path)
    return True
#   else:
#     return False


if __name__ == "__main__":
    # old_file_path = "C:\Program Files\Mantra\MFS100\Driver\MFS100Test\FingerData\FingerImage.bmp"
    # new_file_path = "C:\Program Files\Mantra\MFS100\Driver\MFS100Test\FingerData\FingerImage1.bmp"
    old_file_path = "D:\\fingerprints\\New folder\\pk-rh-thumb.bmp"
    new_file_path = "D:\\fingerprints\\New folder\\pk-rh-thumb1.bmp"

    if rename_file_with_admin_privileges(old_file_path, new_file_path):
        print("File renamed successfully.")
    else:
        print("Failed to rename file.")