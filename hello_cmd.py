# import subprocess

# run=subprocess.run(["C:\\Program Files\\NFIQ 2\\bin\\nfiq2", '"D:\\fingerprints\\1691581877.4867053.bmp"'], capture_output=True, shell=True)
# print(run.stdout)
# print(run.stderr)


# import subprocess
# # You can put the parts of your command in the list below or just use a string directly.
# command_to_execute = ["echo", "Test"]
#C:\\Program Files\\NFIQ 2\\bin\\nfiq2' 'D:\\fingerprints\\1691581877.4867053.bmp
# run = subprocess.run(command_to_execute, capture_output=True, shell=True)

# print(run.stdout) # the output "Test"
# print(run.stderr) # the error part of the output

import subprocess

a = subprocess.run(["C:\\Program Files\\NFIQ 2\\bin\\nfiq2","D:\\fingerprints\\1691581877.4867053.bmp"], stdout=subprocess.PIPE)
print(a.stdout.decode('utf-8'))

# b = subprocess.run(['D:\\fingerprints\\1691581877.4867053.bmp'],input=a.stdout, stdout=subprocess.PIPE)
# print(b.stdout.decode('utf-8'))
