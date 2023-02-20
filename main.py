import io
import multipart

print("multipart v", multipart.__version__)

from fastapi import FastAPI, UploadFile
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config
from starlette.requests import Request
from starlette.middleware.sessions import SessionMiddleware
from PIL import Image
import numpy
import tensorflow as tf

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="R71bsYchVfwGRpuAlXedPIGfs8NNslh5vP0hyeNvOFw=")

config = Config('.env')  # read config from .env file
oauth = OAuth(config)
oauth.register(
    name='github',
    client_id='Iv1.bf157c32fec85d0c',
    client_secret='47a2429f4a39cef229c1f4096e8371ace450d652',
    access_token_url='https://github.com/login/oauth/access_token',
    access_token_params=None,
    authorize_url='https://github.com/login/oauth/authorize',
    authorize_params=None,
    api_base_url='https://api.github.com/',
    client_kwargs={'scope': 'user:email'},
)

model = tf.keras.models.load_model('data/npk.h5')
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

@app.get("/login")
async def login(request: Request):
    redirect_uri = 'https://aidetector-api.pkasila.net/auth'
    return await oauth.github.authorize_redirect(request, redirect_uri)

@app.get("/auth")
async def auth(request: Request):
    token = await oauth.github.authorize_access_token(request)
    user = token['userinfo']
    return dict(user)

@app.post("/detect")
async def detect(file: UploadFile):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    img = img.resize((512, 512))
    img = tf.keras.utils.img_to_array(img)
    full_batch = tf.data.Dataset.from_tensors([img])
    predictions = probability_model.predict(full_batch)[0]
    return {"prediction": {
        "artificial": numpy.asscalar(predictions[0]),
        "human": numpy.asscalar(predictions[1])
    }}
