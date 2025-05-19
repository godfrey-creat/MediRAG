from flask import Flask, render_template, request
from utils import get_text_embedding, get_image_embedding, retrieve_similar_cases, generate_diagnostic_report
import os

app = Flask(__name__)
UPLOAD_FOLDER = "xray_samples"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    diagnosis, references = None, []
    if request.method == 'POST':
        text = request.form['symptoms']
        image = request.files['xray']
        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(image_path)

        diagnosis, references = generate_diagnostic_report(text, image_path)

    return render_template('index.html', diagnosis=diagnosis, references=references)

if __name__ == '__main__':
    app.run(debug=True)