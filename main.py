import io
import multipart

print("multipart v", multipart.__version__)

from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
import numpy
import tensorflow as tf

app = FastAPI()

model = tf.keras.models.load_model('/app/data/npk.h5')
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])


@app.get("/")
async def root():
    content = """
    <body>
    <form action="/detect" enctype="multipart/form-data" method="post">
    <input name="files" type="file">
    <input type="submit">
    </form>
    </body>
        """
    return HTMLResponse(content=content)


@app.get("/detect")
async def detect(file: UploadFile):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    img = tf.keras.utils.img_to_array(img)
    full_batch = tf.data.Dataset.from_tensors([img])
    predictions = probability_model.predict(full_batch)
    return {"prediction": {
        "artificial": numpy.asscalar(predictions[0]),
        "human": numpy.asscalar(predictions[1])
    }}
