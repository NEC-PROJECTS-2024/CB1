from flask import Flask, render_template, request, flash, redirect, url_for
from PIL import Image
import numpy as np
import os
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'super_secret_key'  # Secret key for flash messages

# Load your Keras model
loaded_model = tf.keras.models.load_model('hippo.h5', compile=False)

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

# Function to predict class label
def predict_label(img):
    img = img.resize((28, 28))  # Resize the image to the target size
    img_array = np.array(img) / 255.0  # Convert image to numpy array and rescale to values between 0 and 1
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = loaded_model.predict(img_array)
    return predictions[0]  # Return the probabilities for both classes

# Define route for uploading images
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        # Check if the file has a valid extension
        if not allowed_file(file.filename):
            flash('Invalid file type. Allowed file types are: png, jpg, jpeg, gif', 'error')
            return redirect(request.url)

        try:
            # Use PIL to open the image and convert it to RGB format
            img = Image.open(file).convert('RGB')

            # Make predictions
            class_probabilities = predict_label(img)

            # Set the validation threshold
            threshold = 0.1  # You can adjust this threshold value as needed

            # Check if either class probability exceeds the threshold
            if any(prob > threshold for prob in class_probabilities):
                class_label = "Tumour Found"
            else:
                class_label = "Tumour no found"

            return render_template('upload.html', prediction=class_label) 
        except Exception as e:
            flash(f'Error processing image: {e}', 'error')

    return render_template('upload.html')  # Render the upload page if method is GET

# Define route for home page
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("Home.html")

# Define route for about us page
@app.route('/aboutus', methods=['GET', 'POST'])
def about():
    return render_template("AboutUs.html")

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
