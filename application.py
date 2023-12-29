from flask import Flask, render_template, request, jsonify, send_file
import io
from PIL import Image
from src.pipeline.predict_pipeline import Operations


app = Flask(__name__, template_folder='ui/templates', static_folder='ui/static')


op = Operations()

loaded_models = {}
plants = ["Potato", "Pepper", "Tomato"]
for plant in plants:
    model, class_names = op.select_model(plant)
    loaded_models[plant] = {"model": model, "class_names": class_names}

@app.route("/")
def render_home():
    return render_template('home.html')

@app.route("/classify/<plant>")
def render_classification(plant):
    return render_template('classifier.html')

@app.route("/classify/<plant>/predict", methods=['POST'])
def predict(plant):
    if request.method == 'POST':
        file = request.files['file']
        contents = file.read()
        image = Image.open(io.BytesIO(contents))

        result_dict = op.make_prediction(plant_name=plant, model=loaded_models[plant]["model"],
                                         class_names=loaded_models[plant]["class_names"], img=image)
        return jsonify(result_dict)

if __name__ == "__main__":
    app.run(host='localhost', port=8000)
