from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

import io
from PIL import Image 


from src.pipeline.predict_pipeline import Operations
op = Operations()

loaded_models = {}
plants = ["Potato", "Pepper", "Tomato"]  
for plant in plants:
        model, class_names = op.select_model(plant)
        loaded_models[plant] = {"model": model, "class_names": class_names}

app = FastAPI()
app.mount("/ui/static", StaticFiles(directory="ui/static"), name="static")


@app.get("/")
async def render_home():
    html_home_page = "ui/templates/home.html"
    return FileResponse(html_home_page)

@app.get("/classify/{plant}")
async def render_classification():
    html_classification_page = "ui/templates/classifier.html"
    return FileResponse(html_classification_page)

# UploadFile is a class from FastAPI's fastapi.UploadFile module. It represents a file uploaded via an HTTP request.
# File(...) is a parameter type annotation used in FastAPI to specify that the parameter should receive files uploaded in the request.
# The ellipsis (...) indicates that the file is required.
@app.post("/classify/{plant}/predict")
async def predict( plant: str,  file: UploadFile = File(...) ): 
    # Read the contents of the uploaded file
    contents = await file.read()

    # Convert the file contents to an image
    image = Image.open(io.BytesIO(contents))

    result_dict = op.make_prediction(plant_name=plant, model=loaded_models[plant]["model"], class_names=loaded_models[plant]["class_names"], img=image)
    return result_dict


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
