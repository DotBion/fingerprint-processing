# # Importing flask module in the project is mandatory
# # An object of Flask class is our WSGI application.
# from flask import Flask

# # Flask constructor takes the name of
# # current module (__name__) as argument.
# app = Flask(__name__)

# # The route() function of the Flask class is a decorator,
# # which tells the application which URL should call
# # the associated function.
# @app.route('/')
# # ‘/’ URL is bound with hello_world() function.
# def hello_world():
# 	return 'Hello World'

# # def gfg():
# #    return "geeksforgeeks"
# # app.add_url_rule("/", "g2g", gfg)

# # main driver function
# if __name__ == '__main__':

# 	# run() method of Flask class runs the application
# 	# on the local development server.
# 	app.run()

# ---------------------------------------------------
from flask import Flask, redirect, url_for, request
import os
import cv2  
import subprocess
import numpy as np
from walking import walking, walkonce, checkstable, mergeneighbors

nfiq2_path = "C:\\Program Files\\NFIQ 2\\bin\\nfiq2"
core_points = []
app = Flask(__name__)


@app.route('/success/<name>')
def success(name):
    print("asda: ",core_points)
    #name = name.split('/')
    return 'fingerprint scan comlete for %s \ncalculating nfiq2 score : %s ' % name % core_points


@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        user = request.form['nm']
        old_file = "D:\\#fp-codes\\fingerprint-processing\\FingerImage.jpg"
        new_file = "D:\\#fp-codes\\fingerprint-processing\\" + str(user) + ".jpg"
        #os.rename(old_file, new_file)
        core_points = get_core_points(old_file, user)
        #user = user + "/"+str(core_points)
        print(core_points)
        return redirect(url_for('success', name=user))
    
    else:
        user = request.args.get('nm')
        return redirect(url_for('success', name=user))

# def get_nfiq2(file_name):
#     print("Calculating NFIQ2 Score")
#     a = subprocess.run([nfiq2_path,file_name], stdout=subprocess.PIPE)
#     score = a.stdout.decode('utf-8')
#     print("Fingerprint NFIQ2 Score:", score)
#     #print(a.stdout.decode('utf-8'))
#     return score

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

if __name__ == '__main__':
    app.run(debug=True)
# -------------------------------------------------------------
# import flask
# import cv2
# import numpy as np

# app = flask.Flask(__name__)

# @app.route('/fingerprint_recognition', methods=['POST'])
# def fingerprint_recognition():
#   # Get the fingerprint image from the request.
#   fingerprint_image = request.files['fingerprint_image']

#   # Convert the fingerprint image to a NumPy array.
#   fingerprint_image_array = np.fromstring(fingerprint_image.read(), np.uint8)
#   fingerprint_image = cv2.imdecode(fingerprint_image_array, cv2.IMREAD_COLOR)

#   # Display the fingerprint image on the webpage.
#   fingerprint_image_bytes = cv2.imencode('.jpg', fingerprint_image)[1].tostring()
#   return flask.send_file(io.BytesIO(fingerprint_image_bytes), mimetype='image/jpg')

# if __name__ == '__main__':
#   app.run(debug=True)
