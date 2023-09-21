from __future__ import division, print_function
import sys
import os
import shutil
import glob
import re
from pathlib import Path
from io import BytesIO
import base64
import requests
import tensorflow
from tensorflow.keras.applications import EfficientNetV2S
from keras.models import Model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle, Paragraph
from io import BytesIO
from reportlab.lib import colors
import replicate
import ultralytics
from ultralytics import YOLO
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from fastai import *
from fastai.vision import *
from flask import Flask, redirect, url_for, render_template, request,Response, session,  jsonify
from flask import make_response
from PIL import Image as PILImage
import datetime
from reportlab.lib.styles import getSampleStyleSheet


app = Flask(__name__)


path_to_model = '../models/weight_file/Skin Cancer1.h5'
path_to_model1 = '../models/weight_file/best.pt'

os.environ["REPLICATE_API_TOKEN"] = "r8_dGU76t0kkMgSFA4NMyqKBVkYpLmiOO52Ml7s8"

PATH_TO_MODELS_DIR = Path('') # by default just use /models in root dir



def encode(img):
    b, g, r = cv2.split(img)
    # Flatten pixel values and concatenate channels
    pixel_values = list(r.flatten()) + list(g.flatten()) + list(b.flatten())
    a = np.array(pixel_values).reshape(-1, 28, 28, 3)
    return a

def setup_model_pth(path_to_model):
    loaded_model = load_model(path_to_model,compile=False)
    return loaded_model

def setup_model_pth1(path_to_model1):
    model = YOLO(path_to_model1)
    return model

learn = setup_model_pth(path_to_model)
learn1 = setup_model_pth1(path_to_model1)


def detect(cls):
    if cls==0:
        dis="Actinic Keratoses and Intraepithelial Carcinoma"
    elif cls==1:
        dis="Basal Cell Carcinoma"
    elif cls==2:
        dis="Benign Keratosis-like Lesions"
    elif cls==3:
        dis="Dermatofibroma"
    elif cls==4:
        dis="Melanocytic Nevi"
    elif cls==5:
        dis="Vascular Carcinoma"
    elif cls==6:
        dis="Melanoma"
    else:
        dis="No Disease Detected"
    return dis


@app.route('/', methods=['GET', "POST"])
def index():
    # Main page
    return render_template('index.html')

@app.route('/model')
def model():
    return render_template('model.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/lama')
def lama():
    return render_template('lama.html')

@app.route('/index_bengali')
def index_bengali():
    return render_template('index_bengali.html')

@app.route('/about_bengali')
def about_bengali():
    return render_template('about_bengali.html')

@app.route('/result_bengali')
def result_bengali():
    return render_template('result_bengali.html')

@app.route('/model_bengali')
def model_bengali():
    return render_template('model_bengali.html')

@app.route('/index_hindi')
def index_hindi():
    return render_template('index_hindi.html')

@app.route('/about_hindi')
def about_hindi():
    return render_template('about_hindi.html')

@app.route('/result_hindi')
def result_hindi():
    return render_template('result_hindi.html')

@app.route('/model_hindi')
def model_hindi():
    return render_template('model_hindi.html')


@app.route('/upload', methods=["POST", "GET"])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        model = request.form['model']
        name = request.form['name']
        age = request.form['age']
        phone = request.form['phone']
        email = request.form['email']
        uploaded_files = request.files.getlist('files[]')

        result=[]
        for i, uploaded_file in enumerate(uploaded_files):
            # Read image file
            print(i)
            img = uploaded_file.read()
            img = open_image(BytesIO(img))
            img_data = (image2np(img.data) * 255).astype('uint8')
            pil_img = PILImage.fromarray(img_data)
            pil_img.save(f'static/image_{i}.jpeg', format="JPEG")

            # Resize image if needed (e.g., for input size 28x28)
            img = cv2.imread(f'static/image_{i}.jpeg')
            img1 = cv2.resize(img, (28, 28))
            img2 = cv2.resize(img, (624, 624))

            # Save form data to a JSON file
            emp = []
            emp.append(name)
            emp.append(age)
            emp.append(phone)
            emp.append(email)
            emp.append(model)
            with open(f'static/text_{i}.csv', "w") as file:
                json.dump(emp, file)

            # Make predictions using your model_predict function
            # preds = model_predict(img1, img2, name, age, phone, email, model, i)
            if (model == 'Custom CNN'):
                image_features = encode(img1)
                pred1 = learn.predict(image_features)
                prob = pred1[0]
                pred = np.argmax(pred1, axis=1)
                cls = pred[0]
                cls = detect(cls)

            elif model == 'YOLOv8':
                pred1 = learn1(img2)
                x = pred1[0].probs.top5
                y = pred1[0].probs.top5conf.tolist()
                prob = []
                for i in range(8):
                    if i in x:
                        y_index = x.index(i)
                        y_value = y[y_index]
                        prob.append(y_value)
                    else:
                        prob.append(0)
                prob[4:6], prob[6:8] = prob[6:8], prob[4:6]
                max_position = prob.index(max(prob))
                cls = detect(max_position)


            else:
                image_features = encode(img1)
                pred1 = learn.predict(image_features)
                prob1 = pred1[0]
                pred1 = learn1(img2)
                x = pred1[0].probs.top5
                y = pred1[0].probs.top5conf.tolist()
                prob2 = []
                for i in range(8):
                    if i in x:
                        y_index = x.index(i)
                        y_value = y[y_index]
                        prob2.append(y_value)
                    else:
                        prob2.append(0)
                prob2[4:6], prob2[6:8] = prob2[6:8], prob2[4:6]
                prob = [(0.3*x + 0.7*y) for x, y in zip(prob1, prob2)]
                max_position = prob.index(max(prob))
                cls = detect(max_position)

            directory_path = 'runs'

            if os.path.exists(directory_path):
                try:
                    # Use shutil.rmtree() to remove the directory and its contents.
                    shutil.rmtree(directory_path)
                    print(f"Directory '{directory_path}' has been successfully removed.")
                except Exception as e:
                    print(f"Error while removing directory: {e}")
            else:
                print(f"Directory '{directory_path}' does not exist.")



            res = {"class": cls, "probs": prob, "image": img2, "name": name, "age": age, "phone": phone,
                      "email": email, "model": model}
            result.append(res)
        print(result)
        sum_probs = [0] * 8
        for i in result:
            res = i['probs']
            sum_probs = [x + y for x, y in zip(sum_probs, res)]
        average_probs = [x / len(result) for x in sum_probs]
        max_position = average_probs.index(max(average_probs))
        cls = detect(max_position)

        matching_element = None
        highest_probability = -1
        for i in result:
            print(i['class'])
            if i['class'] == cls:
                print(i)
                if i['probs'][max_position] > highest_probability:
                    highest_probability = i['probs'][max_position]
                    matching_element = i

        img_path = matching_element['image']

        cv2.imwrite("static/image.jpeg", img_path)
        result1 = {"class": cls, "probs": average_probs, "image": img_path, "name": name, "age": age, "phone": phone, "email": email, "model": model}
        em=[]
        em.append(result1['name'])
        em.append(result1['age'])
        em.append(result1['phone'])
        em.append(result1['email'])
        em.append(result1['model'])
        em.append(result1['class'])
        em.append(result1['probs'])
        with open(f'static/text.csv', "w") as file:
            json.dump(em, file)
        return render_template('result1.html', result=result1)
        # return preds
    return 'OK'

@app.route('/upload_beng', methods=["POST", "GET"])
def upload_beng():
    if request.method == 'POST':
        # Get the file from post request
        model = request.form['model']
        name = request.form['name']
        age = request.form['age']
        phone = request.form['phone']
        email = request.form['email']
        uploaded_files = request.files.getlist('files[]')

        result=[]
        for i, uploaded_file in enumerate(uploaded_files):
            # Read image file
            print(i)
            img = uploaded_file.read()
            img = open_image(BytesIO(img))
            img_data = (image2np(img.data) * 255).astype('uint8')
            pil_img = PILImage.fromarray(img_data)
            pil_img.save(f'static/image_{i}.jpeg', format="JPEG")

            # Resize image if needed (e.g., for input size 28x28)
            img = cv2.imread(f'static/image_{i}.jpeg')
            img1 = cv2.resize(img, (28, 28))
            img2 = cv2.resize(img, (624, 624))

            # Save form data to a JSON file
            emp = []
            emp.append(name)
            emp.append(age)
            emp.append(phone)
            emp.append(email)
            emp.append(model)
            with open(f'static/text_{i}.csv', "w") as file:
                json.dump(emp, file)

            # Make predictions using your model_predict function
            # preds = model_predict(img1, img2, name, age, phone, email, model, i)
            if (model == 'Custom CNN'):
                image_features = encode(img1)
                pred1 = learn.predict(image_features)
                prob = pred1[0]
                pred = np.argmax(pred1, axis=1)
                cls = pred[0]
                cls = detect(cls)

            elif model == 'YOLOv8':
                pred1 = learn1(img2)
                x = pred1[0].probs.top5
                y = pred1[0].probs.top5conf.tolist()
                prob = []
                for i in range(8):
                    if i in x:
                        y_index = x.index(i)
                        y_value = y[y_index]
                        prob.append(y_value)
                    else:
                        prob.append(0)
                prob[4:6], prob[6:8] = prob[6:8], prob[4:6]
                max_position = prob.index(max(prob))
                cls = detect(max_position)


            else:
                image_features = encode(img1)
                pred1 = learn.predict(image_features)
                prob1 = pred1[0]
                pred1 = learn1(img2)
                x = pred1[0].probs.top5
                y = pred1[0].probs.top5conf.tolist()
                prob2 = []
                for i in range(8):
                    if i in x:
                        y_index = x.index(i)
                        y_value = y[y_index]
                        prob2.append(y_value)
                    else:
                        prob2.append(0)
                prob2[4:6], prob2[6:8] = prob2[6:8], prob2[4:6]
                prob = [(0.3*x + 0.7*y) for x, y in zip(prob1, prob2)]
                max_position = prob.index(max(prob))
                cls = detect(max_position)

            directory_path = 'runs'

            if os.path.exists(directory_path):
                try:
                    # Use shutil.rmtree() to remove the directory and its contents.
                    shutil.rmtree(directory_path)
                    print(f"Directory '{directory_path}' has been successfully removed.")
                except Exception as e:
                    print(f"Error while removing directory: {e}")
            else:
                print(f"Directory '{directory_path}' does not exist.")


            res = {"class": cls, "probs": prob, "image": img2, "name": name, "age": age, "phone": phone,
                      "email": email, "model": model}
            result.append(res)
        print(result)
        sum_probs = [0] * 8
        for i in result:
            res = i['probs']
            sum_probs = [x + y for x, y in zip(sum_probs, res)]
        average_probs = [x / len(result) for x in sum_probs]
        max_position = average_probs.index(max(average_probs))
        cls = detect(max_position)

        matching_element = None
        highest_probability = -1
        for i in result:
            print(i['class'])
            if i['class'] == cls:
                print(i)
                if i['probs'][max_position] > highest_probability:
                    highest_probability = i['probs'][max_position]
                    matching_element = i

        img_path = matching_element['image']

        result1 = {"class": cls, "probs": average_probs, "image": img_path, "name": name, "age": age, "phone": phone, "email": email, "model": model}
        print(result1)
        return render_template('result1_bengali.html', result=result1)
        # return preds
    return 'OK'

@app.route('/upload_hindi', methods=["POST", "GET"])
def upload_hindi():
    if request.method == 'POST':
        # Get the file from post request
        model = request.form['model']
        name = request.form['name']
        age = request.form['age']
        phone = request.form['phone']
        email = request.form['email']
        uploaded_files = request.files.getlist('files[]')

        result=[]
        for i, uploaded_file in enumerate(uploaded_files):
            # Read image file
            print(i)
            img = uploaded_file.read()
            img = open_image(BytesIO(img))
            img_data = (image2np(img.data) * 255).astype('uint8')
            pil_img = PILImage.fromarray(img_data)
            pil_img.save(f'static/image_{i}.jpeg', format="JPEG")

            # Resize image if needed (e.g., for input size 28x28)
            img = cv2.imread(f'static/image_{i}.jpeg')
            img1 = cv2.resize(img, (28, 28))
            img2 = cv2.resize(img, (624, 624))

            # Save form data to a JSON file
            emp = []
            emp.append(name)
            emp.append(age)
            emp.append(phone)
            emp.append(email)
            emp.append(model)
            with open(f'static/text_{i}.csv', "w") as file:
                json.dump(emp, file)

            # Make predictions using your model_predict function
            # preds = model_predict(img1, img2, name, age, phone, email, model, i)
            if (model == 'Custom CNN'):
                image_features = encode(img1)
                pred1 = learn.predict(image_features)
                prob = pred1[0]
                pred = np.argmax(pred1, axis=1)
                cls = pred[0]
                cls = detect(cls)

            elif model == 'YOLOv8':
                pred1 = learn1(img2)
                x = pred1[0].probs.top5
                y = pred1[0].probs.top5conf.tolist()
                prob = []
                for i in range(8):
                    if i in x:
                        y_index = x.index(i)
                        y_value = y[y_index]
                        prob.append(y_value)
                    else:
                        prob.append(0)
                prob[4:6], prob[6:8] = prob[6:8], prob[4:6]
                max_position = prob.index(max(prob))
                cls = detect(max_position)


            else:
                image_features = encode(img1)
                pred1 = learn.predict(image_features)
                prob1 = pred1[0]
                pred1 = learn1(img2)
                x = pred1[0].probs.top5
                y = pred1[0].probs.top5conf.tolist()
                prob2 = []
                for i in range(8):
                    if i in x:
                        y_index = x.index(i)
                        y_value = y[y_index]
                        prob2.append(y_value)
                    else:
                        prob2.append(0)
                prob2[4:6], prob2[6:8] = prob2[6:8], prob2[4:6]
                prob = [(0.3*x + 0.7*y) for x, y in zip(prob1, prob2)]
                max_position = prob.index(max(prob))
                cls = detect(max_position)

            directory_path = 'runs'

            if os.path.exists(directory_path):
                try:
                    # Use shutil.rmtree() to remove the directory and its contents.
                    shutil.rmtree(directory_path)
                    print(f"Directory '{directory_path}' has been successfully removed.")
                except Exception as e:
                    print(f"Error while removing directory: {e}")
            else:
                print(f"Directory '{directory_path}' does not exist.")

            res = {"class": cls, "probs": prob, "image": img2, "name": name, "age": age, "phone": phone,
                      "email": email, "model": model}
            result.append(res)
        print(result)
        sum_probs = [0] * 8
        for i in result:
            res = i['probs']
            sum_probs = [x + y for x, y in zip(sum_probs, res)]
        average_probs = [x / len(result) for x in sum_probs]
        max_position = average_probs.index(max(average_probs))
        cls = detect(max_position)

        matching_element = None
        highest_probability = -1
        for i in result:
            print(i['class'])
            if i['class'] == cls:
                print(i)
                if i['probs'][max_position] > highest_probability:
                    highest_probability = i['probs'][max_position]
                    matching_element = i

        img_path = matching_element['image']

        result1 = {"class": cls, "probs": average_probs, "image": img_path, "name": name, "age": age, "phone": phone, "email": email, "model": model}
        print(result1)
        return render_template('result1_hindi.html', result=result1)
        # return preds
    return 'OK'


@app.route('/generate_pdf')
def generate_pdf():
    # Create a BytesIO buffer to hold the PDF data

    with open('static/text.csv', mode="r") as file:
        # Create a CSV reader object
        loaded_list = json.load(file)

    name = loaded_list[0]
    age = loaded_list[1]
    phone = loaded_list[2]
    email = loaded_list[3]
    model = loaded_list[4]
    disease_class = loaded_list[5]
    probability = loaded_list[6]
    current_datetime = datetime.datetime.now()

    table_data = [["Disease", "Probability"],
                      ["Actinic Keratoses and Intraepithelial Carcinoma", "{:.2f}".format(probability[0] * 100)],
                      ["Basal Cell Carcinoma", "{:.2f}".format(probability[1] * 100)],
                      ["Benign Keratosis-like Lesions", "{:.2f}".format(probability[2] * 100)],
                      ["Dermatofibroma", "{:.2f}".format(probability[3] * 100)],
                      ["Melanocytic Nevi", "{:.2f}".format(probability[4] * 100)],
                      ["Vascular Carcinoma", "{:.2f}".format(probability[5] * 100)],
                      ["Melanoma", "{:.2f}".format(probability[6] * 100)],
                    ["Normal Skin", "{:.2f}".format(probability[7] * 100)]]



    # Format the date and time as a string
    formatted_datetime = current_datetime.strftime("%d-%m-%Y")


    buffer = BytesIO()



    pdf = canvas.Canvas(buffer)

    pdf.rect(20, 20, 550, 780)


    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(240, 750, "Skin-O-Vis")

    pdf.setFont("Helvetica", 12)
    pdf.drawString(430, 700, "Date: " + formatted_datetime)
    pdf.drawString(100, 700, "Name: " + name)
    pdf.drawString(100, 685, "Age: " + age)
    pdf.drawString(100, 670, "Contact: " + phone)
    pdf.drawString(100, 655, "Email-ID: " + email)
    pdf.drawString(100, 640, "Detection Model: " + model)
    pdf.drawString(100, 625, "Disease detected: " + disease_class)

    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkgrey),  # Header row background color
        ('TEXTCOLOR', (0, 0), (-1, 0), (1, 1, 1)),  # Header text color
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Center alignment for all cells
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Header font
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),  # Header padding
        ('BACKGROUND', (0, 1), (-1, -1), (0.85, 0.85, 0.85)),  # Table data background color
        ('GRID', (0, 0), (-1, -1), 1, (0, 0, 0)),  # Table grid
    ]))
    table.wrapOn(pdf, 400, 280)  # Set table size
    table.drawOn(pdf, 160, 420)  # Set table position

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(100, 350, "Disclaimer: ")

    styles = getSampleStyleSheet()
    style = styles["Normal"]
    style.alignment = 4  # 0=Left, 1=Center, 2=Right, 4=Justify
    style.fontName='Helvetica'
    style.fontSize = 12

    paragraph = Paragraph("This AI-generated report is preliminary and not a substitute for professional medical advice. Consult a healthcare provider for a comprehensive evaluation. Decisions about your health should be made in consultation with a qualified physician.", style)
    paragraph.wrapOn(pdf, 400, 100)
    paragraph.drawOn(pdf, 100, 300)


    pdf.showPage()
    pdf.save()

    buffer.seek(0)


    message = MIMEMultipart()
    message["To"] = email
    message["From"] = 'Skin-O-Vis'
    message["Subject"] = 'AI generated medical report'

    title = '<b> Title line here. </b>'

    email_body = """
    Dear recipient,

    Please find attached the AI-generated medical report for your recent skin analysis.

    This AI-generated report is preliminary and not a substitute for professional medical advice. Consult a healthcare provider for a comprehensive evaluation. Decisions about your health should be made in consultation with a qualified physician.

    Sincerely,
    Skin-O-Vis
        """
    messageText = MIMEText(email_body, 'plain')
    message.attach(messageText)

    pdf_attachment = MIMEText(buffer.getvalue(), 'base64', 'pdf')
    pdf_attachment.add_header('Content-Disposition', f'attachment; filename=test_report.pdf')
    message.attach(pdf_attachment)


    email_id = 'skinovis.corp@gmail.com'
    password = 'gipelnfikrdqnzak'

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(email_id, password)


        server.sendmail(email_id, email, message.as_string())


        server.quit()

        print("PDF sent successfully to your email.")
    except Exception as e:
        print(str(e))


    return Response(buffer.getvalue(), mimetype='application/pdf', headers={'Content-Disposition': 'attachment; filename=test_report.pdf'})




@app.route("/get_response", methods=["POST"])
def get_response():
    pre_prompt = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'. Dont give direct conversation, just answer the question"
    prompt_input = request.form.get("prompt_input")

    allowed_words_file = "allowed_words.txt"
    allowed_words = []
    if os.path.isfile(allowed_words_file):
        with open(allowed_words_file, "r") as file:
            allowed_words = [line.strip() for line in file]


    prompt_words = set(prompt_input.lower().split())  # Split prompt into words

    intersection = any(word.lower() in prompt_words for word in allowed_words)

    if not intersection:
        return jsonify({"response": "Invalid prompt. Please use an allowed word."})

    # Generate LLM response
    output = replicate.run(
        'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5',
        input={
            "prompt": f"{pre_prompt} {prompt_input} Assistant: ",
            "temperature": 0.1,
            "top_p": 0.9,
            "max_length": 1000,
            "repetition_penalty": 1
        }
    )

    full_response = ""

    for item in output:
        full_response += item

    return jsonify({"response": full_response})



if __name__ == '__main__':
    port = os.environ.get('PORT', 8008)

    if "prepare" not in sys.argv:
        app.run(debug=False, host='0.0.0.0', port=port)