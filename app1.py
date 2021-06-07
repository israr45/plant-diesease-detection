from flask import Flask, redirect, url_for, request, render_template
import os
from fastai import *
from fastai.vision import *

app = Flask(__name__)
model_path =  "model"
model_name = "Plantvillage.pkl"

path = "dataset"
'''
data = ImageDataBunch.from_folder(path,
                                 ds_tfms=get_transforms(do_flip=True),
                                 bs=32,
                                 size=224,
                                 valid_pct= 0.22).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet34, metrics = [accuracy])
'''

learn = load_learner(model_path, model_name)

print('Model Loaded. Check http://127.0.0.1:5000/')


@app.route('/', methods=['GET', 'POST'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        pred_class,_,_=learn.predict(file_path)
        pred_class
        print (pred_class)
        # Make prediction
        #preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        return pred_class
    return None


if __name__ == '__main__':
    app.run(debug=True)
