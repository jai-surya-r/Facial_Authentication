from flask import Flask, redirect, url_for, jsonify, render_template, Response, request, session
from FaceDBManager import MongoDB
import face_recognition
import io, cv2, os, time, random, torch
from torchvision import transforms
from MesoNet import Meso4
from ResNetLSTM import Model, ImagePredictor, validation_dataset

app = Flask(__name__)

app.secret_key = "SecureFace101"

global TRIGGER, CAMERA, camera, REC_TRIGGER
TRIGGER, REC_TRIGGER, CAMERA = False, False, False

# Load MesoNet model
meso = Meso4()
meso.load_model("Models\Meso4.h5")

# Load and initialize ResNetLSTM model
resnet = Model(2).cuda()
path_to_model = "Models\Model_90_20_FF.pt"
resnet.load_state_dict(torch.load(path_to_model))

TEMP_FOLDER = "Temp"

def camera_operation(switch):
    global camera, CAMERA
    if switch == 1 and CAMERA:
        camera = cv2.VideoCapture(0)
    elif switch == 0:
        CAMERA = False
        camera.release()
        cv2.destroyAllWindows()

def gen_frames():
    global camera, TRIGGER, REC_TRIGGER
    start_time = None
    video_frames = []
    while True:
        _, frame = camera.read()
        if camera.isOpened():
            if(TRIGGER):
                TRIGGER = False
                file_path = os.path.join(TEMP_FOLDER, 'Temp_image.png')
                cv2.imwrite(file_path, frame)
                camera_operation(0)
                print("Image saved")
            
            if(REC_TRIGGER):
                if start_time is None:
                    start_time = time.time()
                current_time = time.time()
                if current_time - start_time <= 3:
                    frame = cv2.putText(cv2.flip(frame,1),"Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),4)
                    frame = cv2.flip(frame,1)
                    video_frames.append(frame)
                else:
                    # Save video frames as mp4
                    file_path = os.path.join(TEMP_FOLDER, 'Temp_video.mp4')
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    out = cv2.VideoWriter(file_path, fourcc, 30.0, (width, height))
                    for frame in video_frames:
                        out.write(frame)
                    out.release()
                    REC_TRIGGER = False
                    start_time = None
                    camera_operation(0)
                    print("Video saved")
            
            try:
                _, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Error: {e}")

def choose_random_frame():
    cap = cv2.VideoCapture("Temp/Temp_video.mp4")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    random_frame_index = random.randint(0, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index)
    success, frame = cap.read()
    cap.release()

    # Check if the frame was read successfully
    if success:
        return frame
    else:
        return None

def Meso_model():
    global meso
    img_path = "Temp/Temp_image.png"
    img = cv2.imread(img_path)
    prediction = meso.predict_single(img)
    if prediction >= 0.99:
        print('Predicted: Real', prediction)
        return "REAL", prediction
    elif prediction <=60 and prediction>=90:
        print('Predicted: Fake', prediction)
        return "FAKE", prediction
    else: 
        print('Predicted: Real', prediction)
        return "REAL", prediction

def Resnet_model():
    global resnet
    im_size = 112
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    train_transforms = transforms.Compose([ transforms.ToPILImage(),
                                            transforms.Resize((im_size,im_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean,std)])
    path_to_videos= ["Temp/Temp_video.mp4"]
    video_dataset = validation_dataset(path_to_videos,sequence_length = 20,transform = train_transforms)
    resnet.eval()
    img_pred = ImagePredictor(resnet)
    for i in range(0,len(path_to_videos)):
        print(path_to_videos[i])
        prediction = img_pred.predict(video_dataset[i])
        if prediction[0] == 1:
            print("REAL")
            return "REAL", prediction[1]
        else:
            print("FAKE")
            return "FAKE", prediction[1]

@app.route("/")
@app.route("/home")
def home():
    return render_template("Home.html")

@app.route("/features")
def features():
    return render_template("Features.html")

@app.route("/admin_login", methods=['POST'])
def admin_login():
    username = request.form.get('username')
    password = request.form.get('password')

    # Check if username and password are correct
    if username == "admin" and password == "12345":
        session['admin'] = True
        return redirect(url_for('admin'))
    return render_template('Home.html', error_message="Invalid username or password")

@app.route("/admin", methods=['GET', 'POST'])
def admin():
    if(admin):
        # Fetch records from the backend
        records = db_manager.fetch_all_records()
        return render_template("Admin.html", records=records)
    return render_template('Home.html', error_message="Admin is not logged in, Please login")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/start_stop", methods=['GET'])
def start_stop():
    global CAMERA
    if CAMERA == True:
        camera_operation(0)
        return jsonify({'Camera': 'Off'})
    else:
        CAMERA = True
        camera_operation(1)
        return jsonify({'Camera': 'On'})

@app.route("/trigger", methods=['GET'])
def trigger():
    global TRIGGER, REC_TRIGGER
    action = request.args.get('action')
    if action == 'image':
        TRIGGER = True
        return jsonify({'trigger': 'Captured photo'})
    elif action == 'video':
        REC_TRIGGER = True
        return jsonify({'trigger': 'Started video recording'})
    else:
        return jsonify({'error': 'Invalid action'})

@app.route("/delete_user", methods = ['POST'])
def delete_user():
    # Retrieve the user's name from the request body
    data = request.json
    name = data.get('name')
    # Delete the user record from the database
    try:
        db_manager.delete_user(name)
        return 'User deleted successfully', 200
    except Exception as e:
        return f'Failed to delete user: {str(e)}', 500

@app.route("/signup", methods = ['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Get user input from the form
        username = request.form['username']
        upload_option = request.form['upload-option']
        # print(request.form)

        if upload_option == 'camera':
            file_path = os.path.join(TEMP_FOLDER, 'Temp_image.png')
            image = face_recognition.load_image_file(file_path)
        
        elif upload_option == 'upload':
            # Read the image file and convert it to face_recognition format
            face_image = request.files['face-image']
            image_bytes = io.BytesIO(face_image.read())
            image = face_recognition.load_image_file(image_bytes)

        # Register the user in MongoDB
        try:
            user_id = db_manager.register_user(username, image)
            return render_template('Home.html', message=f"User successfully registered with id: {user_id}")
        except ValueError as e:
            return f"Registration failed: {str(e)}"
    return render_template("Signup.html")

@app.route("/login", methods = ['GET', 'POST'])
def login():
    chosen_model = request.args.get('model')
    session['model'] = chosen_model
    return render_template('Login.html', chosen_model = chosen_model)

@app.route("/login_process", methods = ['POST'])
def login_process():
    global meso, resnet
    # Retrieve username from the form
    username = request.form.get('username')
    upload = request.form.get('upload-option')
    chosen_model = session.get('model')

    # Check if the chosen model is MesoNet
    if chosen_model == 'MesoNet':
        if upload == 'upload':
            image_file = request.files['face-image']
            image_file.save("Temp/Temp_image.png")
            image = face_recognition.load_image_file(image_file)
            authenticate = db_manager.authenticate(username, image)
        elif upload=='camera':
            file_path = os.path.join(TEMP_FOLDER, 'Temp_image.png')
            image = face_recognition.load_image_file(file_path)
            authenticate = db_manager.authenticate(username, image)

    # Check if the chosen model is ResNetLSTM
    elif chosen_model == 'ResNetLSTM':
        if upload == 'upload':
            video_file = request.files['face-video']
            video_file.save("Temp/Temp_video.mp4")
            frame = choose_random_frame()
            authenticate = db_manager.authenticate(username, frame)
        elif upload == 'camera':
            file_path = os.path.join(TEMP_FOLDER, 'Temp_video.mp4')
            frame = choose_random_frame()
            authenticate = db_manager.authenticate(username, frame)

    # Check if authentication was successful
    if authenticate:
        print("Login successful")
        return redirect(url_for('model_eval'))
    else:
        print("Authentication failed")
        return render_template('Result.html', face_reg = "Failed", df_pred = "Not predicted", confidence = "--")

@app.route("/model_eval")
def model_eval():
    chosen_model = session.get('model')

    if chosen_model == 'MesoNet':
        df_pred, confidence = Meso_model()
        session['model'] = None
        if df_pred == "REAL":
            return render_template('Result.html', face_reg = "Success", df_pred = df_pred, confidence = confidence, result="Successful")
        else:
            return render_template('Result.html', face_reg = "Success", df_pred = df_pred, confidence = confidence, result="Unsuccessful")
    elif chosen_model == 'ResNetLSTM':
        df_pred, confidence = Resnet_model()
        session['model'] = None
        if df_pred == "REAL":
            return render_template('Result.html', face_reg = "Success", df_pred = df_pred, confidence = confidence, result="Successful")
        else:
            return render_template('Result.html', face_reg = "Success", df_pred = df_pred, confidence = confidence, result="Unsuccessful")

@app.route("/logout")
def logout():
    session['admin'] = False
    return redirect(url_for('home'))

if __name__ == "__main__":
    db_manager = MongoDB()
    app.run(debug=True)
