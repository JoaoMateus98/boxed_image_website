from flask import Flask, request, render_template, send_file
import secrets
from text_detector import get_boxed_image_url

app = Flask(__name__)

# Set a secret key for session management
app.secret_key = secrets.token_hex(16) # Generates a secure random string
@app.route('/', methods=['GET', 'POST'])
def upload():
    boxed_image_url = None

    if request.method == 'POST':
        title = request.form['title']
        image = request.files['image']

        boxed_image_url = get_boxed_image_url(image, title)
        if boxed_image_url is None:
            return "No text found in the image."

    return render_template('upload.html', processed_image_url=boxed_image_url)

if __name__ == '__main__':
    app.run(debug=True)