import io
import multipart
import hashlib

print("multipart v", multipart.__version__)

from fastapi import FastAPI, UploadFile
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config
from starlette.requests import Request
from starlette.middleware.sessions import SessionMiddleware
from PIL import Image
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

model = tf.keras.models.load_model('data/npk.h5', compile=False)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    0.001,
    decay_steps=100,
    decay_rate=0.96,
    staircase=True)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

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
    src = Image.open(io.BytesIO(contents))
    img = src.resize((512, 512))
    img = tf.keras.utils.img_to_array(img)[:, :, :3]
    full_batch = tf.data.Dataset.from_tensors([img])
    predictions = probability_model.predict(full_batch)[0]

    exif_data = {}
    saved = False

    try:
        exif_data = src.getexif()
        if predictions.item(1) < 0.5 and len(exif_data.keys()) > 0:
            m = hashlib.sha256()
            m.update(contents)
            file_name = f"data/exif_checked/{m.hexdigest()}.webp"
            src.save(file_name)
            saved = True
    except Exception as e:
        print(e)

    return {
        "prediction": {
            "artificial": predictions.item(0),
            "human": predictions.item(1)
        },
        "dataset_enhance": {
            "has_exif": len(exif_data.keys()) > 0,
            "saved": saved
        }
    }
