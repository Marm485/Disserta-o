from flask import Flask, render_template, url_for, redirect, request, session
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
import sqlite3
from datetime import date

app = Flask(__name__)

app.config["IMAGE_UPLOADS"] = 'static/imgs'
app.config["IMAGE_THUMBNAILS"] = "static/thumbnails"

ALLOWED_EXTENSIONS = {'jpeg', 'png', 'jpg'}
MODEL_FILES = ['models/iNaturalist.tflite',
               'models/iNat_and_floraOn.tflite', 'models/floraOn.tflite']
LABEL_FILES = ['dicts/iNaturalist.txt',
               'dicts/iNat_and_floraOn.txt', 'dicts/floraOn.txt']
MODEL_NAMES = ['iNaturalist','Flora_On_and_iNat','Flora_On']


class Model:
    def __init__(self, model_file, dict_file):
        with open(dict_file, 'r') as f:
            self.labels = [line.strip().replace('_', ' ')
                           for line in f.readlines()]
        self.interpreter = tf.lite.Interpreter(model_path=model_file)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.floating_model = self.input_details[0]['dtype'] == np.float32
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

    def classify(self, model_name, file, maxResults=5, min_confidence=0):
        with Image.open(file).resize((self.width, self.height)) as img:
            input_data = np.expand_dims(img, axis=0)
            if self.floating_model:
                input_data = (np.float32(input_data) - 127.5) / 127.5
            self.interpreter.set_tensor(
                self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(
                self.output_details[0]['index'])
            results = np.squeeze(output_data)
            top_categories = results.argsort()[::-1]
            if maxResults != None:
                top_categories = top_categories[:maxResults]
            #print("==> %s <==" % file)
            info = []
            info.append(model_name)
            info.append(str(file)[(str(file).rfind('/')+1):])
            for i in top_categories:
                if self.floating_model:
                    r = float(results[i])
                else:
                    r = float(results[i] / 255.0)
                if min_confidence != None and r < min_confidence:
                    break
                info.append(self.labels[i].replace(' ','_'))
                info.append(str(round((r*100), 2)))
            return info

create_tests_table = '''
CREATE TABLE IF NOT EXISTS tests
(
    id integer PRIMARY KEY,
    filename text NOT NULL,
    expert_id integer NOT NULL,
    data text NOT NULL,
    expert_label text NOT NULL,
    img blob NOT NULL,
    notes text
)
'''

create_classifications_table = '''
CREATE TABLE IF NOT EXISTS classifications
(
    id integer PRIMARY KEY,
    test_id integer,
    model text NOT NULL,
    label_1 text NOT NULL,
    confidence_1 real,
    label_2 text NOT NULL,
    confidence_2 real,
    label_3 text NOT NULL,
    confidence_3 real,
    label_4 text NOT NULL,
    confidence_4 real,
    label_5 text NOT NULL,
    confidence_5 real,
    FOREIGN KEY (test_id) REFERENCES tests(id)
)
'''
#connect to database
def connect_to_db():
    conn = sqlite3.connect('db/tests.db')
    c = conn.cursor()
    c.execute(create_tests_table)
    c.execute(create_classifications_table)
    conn.commit()
    return conn

#insert data in the database
def insert_in_db(conn,db,query):
    cursor = conn.cursor()
    if db == 'tests':
        try:
            cursor.execute('''INSERT INTO tests
            VALUES (?,?,?,?,?,?,?)''',query)
            conn.commit()
        except sqlite3.Error as error:
            print('Failed to insert data in table tests - ', error)
    elif db == 'classifications':
        try:
            cursor.execute('''INSERT INTO classifications
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''',query)
            conn.commit()
        except sqlite3.Error as error:
            print('Failed to insert data in table classifications - ', error)
    else:
        print('Error: table {} not found!'.format(db))

#get id of test(unique)
def get_id(cursor,db):
    cursor.execute('''SELECT * 
    FROM {}'''.format(db))
    results = cursor.fetchall()
    return len(results) + 1

#convert files to binary so they can be stored in the database
def converttoBinary(filename):
    with open(filename,'rb') as file:
        blobData = file.read()
    return blobData


def check_extension(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_thumbnail(inp,out):
    image = Image.open(inp)
    #debug('Image dimensions: %d x %d' % (image.width, image.height))
    max_pixels = 320
    aspect_ratio = float(image.width) / float(image.height)
    if aspect_ratio > 1.0:
        (new_width, new_height) = (max_pixels, int(max_pixels / aspect_ratio))
    else:
        (new_width, new_height) = (int(max_pixels * aspect_ratio), max_pixels)
    #debug('Thumbnail dimensions: %d x %d' % (new_width, new_height))
    thumb_image = image.resize((new_width, new_height))
    thumb_image.save(out)

@app.route("/")
def hello_world():
    return redirect(url_for('upload_image'))


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        #check image
        if request.files:
            images = request.files.getlist("images[]")
            #print(images)

        #check specialist data
        if request.values:
            f_name = request.form['fname']
            expert_label = request.form['species_in']
        else:
            print('No values in request!')
            redirect(request.url)

        extension = True
        # wrong file extension
        for image in images:
            if not check_extension(image.filename):
                print("Error: file extensions allowed %s!" % (ALLOWED_EXTENSIONS))
                extension = False
        
        if extension == False:
            redirect(request.url)
        else:
            names = ""
            for image in images:
                filename = secure_filename(image.filename)

                image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

                names = names + filename + ","

                print('Image has been saved!')

            #print(names)
            return redirect(url_for('loading_results', names=names,\
                f_name=f_name, expert_label=expert_label))
    else:
        print('No image selected!')
        redirect(request.url)

    return render_template("upload.html")


@app.route('/results/<names>/<f_name>/<expert_label>', methods=['GET','POST'])
def loading_results(names,f_name,expert_label):
    if request.method == 'GET':
        #connect to database
        conn = connect_to_db()
        cursor = conn.cursor()

        exp_label = {'label':expert_label}
        #date
        today = date.today()

        names = names.split(',')
        #eliminate last element
        del names[-1]

        rowspan = len(names)

        inat = []
        fl_inat = []
        flora = []
        
        #for each image
        for name in names:

            #test's id
            id = get_id(cursor,'tests')
            
            expert_id = 1050 #colocar em textbox na interface

            filename = app.config["IMAGE_UPLOADS"] + '/' + name
            thumbnail_name = app.config["IMAGE_THUMBNAILS"] + '/' + name

            generate_thumbnail(filename,thumbnail_name)

            #image to blob, to send to the db
            img_blob = converttoBinary(filename)
            
            #values to insert in table 'tests'
            data_tests = (id,name,expert_id,str(today),expert_label,img_blob,'notas')        
            insert_in_db(conn,'tests',data_tests)
            
            #test each model with the image
            for i in range(3):
                model = Model(MODEL_FILES[i], LABEL_FILES[i])
                output = model.classify(MODEL_NAMES[i],filename)

                #'iNaturalist','Flora_On_and_iNat','Flora_On'
                if MODEL_NAMES[i] == "iNaturalist":
                    inat.append(output)
                elif MODEL_NAMES[i] == "Flora_On_and_iNat":
                    fl_inat.append(output)
                else:
                    flora.append(output)

                #classification's id
                classif_id = get_id(cursor,'classifications')

                #insert in db
                data_classifications = (classif_id,id,MODEL_NAMES[i],output[2],\
                    output[3],output[4],output[5],output[6],output[7],\
                        output[8],output[9],output[10],output[11])
                insert_in_db(conn,'classifications',data_classifications)
        conn.commit()
        conn.close()
    
    #back to the main page
    if request.method == 'POST':
        return redirect(url_for('upload_image'))

    return render_template("results.html",row_span=rowspan,inat=inat,\
        fl_inat=fl_inat,flora=flora,names=names,\
            f_name=f_name,expert_label=exp_label)


if __name__ == '__main__':
    app.run(use_reloader = True)
