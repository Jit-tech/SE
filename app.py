import base64
import io
import dash
from dash import dcc, html, Output, Input
import numpy as np
import cv2
import dlib
from deepface import DeepFace
from PIL import Image
import os
import gdown

# Function to download the model from Google Drive if not already present
def download_predictor():
    file_id = "1TK2XoVcKTTei3MjFVpuHQuXiFLxN-G0F"
    output = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        print("Downloading shape_predictor_68_face_landmarks.dat from Google Drive...")
        gdown.download(url, output, quiet=False)
        print("Download complete.")

# Ensure model is downloaded
download_predictor()

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# Load model and detectors
predictor_path = "shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(predictor_path)

# Eye aspect ratio-based stress detection
def detect_eyes_and_stress(gray, landmarks):
    def eye_aspect_ratio(eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)

    left_eye = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(36, 42)])
    right_eye = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(42, 48)])
    ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
    return "Stressed" if ear < 0.2 else "Relaxed"

# Analyze uploaded image
def analyze_uploaded_image(content):
    _, content_string = content.split(',')
    img_data = base64.b64decode(content_string)
    img = Image.open(io.BytesIO(img_data)).convert('RGB')
    frame = np.array(img)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_detector(gray)

    output_text = []
    for face in faces:
        landmarks = landmark_predictor(gray, face)
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_img = frame[y:y+h, x:x+w]

        try:
            analysis = DeepFace.analyze(face_img, actions=['emotion', 'age', 'gender'], enforce_detection=False)[0]
            stress = detect_eyes_and_stress(gray, landmarks)
            output_text.append(f"Emotion: {analysis['dominant_emotion']}, Age: {int(analysis['age'])}, Gender: {analysis['gender']}, Stress: {stress}")
        except Exception as e:
            output_text.append(f"Error analyzing face: {e}")

    return " | ".join(output_text)

# Layout
app.layout = html.Div([
    html.H2("Stress & Emotion Detection"),
    dcc.Upload(
        id='upload-image',
        children=html.Button("Upload Image"),
        multiple=False
    ),
    html.Div(id='result'),
    html.Img(id='uploaded-image', style={'maxWidth': '100%'})
])

# Callback
@app.callback(
    Output('result', 'children'),
    Output('uploaded-image', 'src'),
    Input('upload-image', 'contents')
)
def update_output(contents):
    if contents is not None:
        result = analyze_uploaded_image(contents)
        return result, contents
    return "", None

# Run server
if __name__ == '__main__':
    app.run_server(debug=True)
