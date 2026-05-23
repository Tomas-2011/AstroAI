from flask import Flask, render_template, request, send_from_directory
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

app = Flask(__name__)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename
    )

@app.route("/planets")
def planets():

    return render_template(
        "planets.html",
        planets=planet_info
    )

# Carpeta uploads
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Desactivar notación científica
np.set_printoptions(suppress=True)

# Cargar modelo
model = load_model(
    "keras_Model.h5",
    compile=False
)

# Cargar labels
class_names = open(
    "labels.txt",
    "r"
).readlines()

# Información de planetas
planet_info = {

    "Mercury": {
        "description":"Mercurio es el planeta más cercano al Sol.",
        "color":"Gris",
        "gravity":"3.7 m/s²",
        "temperature":"167°C",
        "distance":"57.9 millones km",
        "year":"88 días",
        "fun_fact":"Un día dura más que un año."
    },

    "Venus": {
        "description":"Venus es el planeta más caliente.",
        "color":"Amarillo pálido",
        "gravity":"8.87 m/s²",
        "temperature":"464°C",
        "distance":"108.2 millones km",
        "year":"225 días",
        "fun_fact":"Gira al revés."
    },

    "Earth": {
        "description":"La Tierra es el único planeta conocido con vida.",
        "color":"Azul y verde",
        "gravity":"9.8 m/s²",
        "temperature":"15°C",
        "distance":"149.6 millones km",
        "year":"365 días",
        "fun_fact":"El 71% está cubierto por agua."
    },

    "Mars": {
        "description":"Marte es conocido como el planeta rojo.",
        "color":"Rojo",
        "gravity":"3.7 m/s²",
        "temperature":"-63°C",
        "distance":"227.9 millones km",
        "year":"687 días",
        "fun_fact":"Tiene el volcán más grande del sistema solar."
    },

    "Jupiter": {
        "description":"Júpiter es el planeta más grande.",
        "color":"Marrón y blanco",
        "gravity":"24.8 m/s²",
        "temperature":"-145°C",
        "distance":"778 millones km",
        "year":"11.8 años",
        "fun_fact":"Podrían caber más de 1300 Tierras."
    },

    "Saturn": {
        "description":"Saturno es famoso por sus anillos.",
        "color":"Dorado",
        "gravity":"10.4 m/s²",
        "temperature":"-178°C",
        "distance":"1.4 mil millones km",
        "year":"29 años",
        "fun_fact":"Sus anillos contienen hielo."
    },

    "Uranus": {
        "description":"Urano gira casi de lado.",
        "color":"Azul claro",
        "gravity":"8.69 m/s²",
        "temperature":"-224°C",
        "distance":"2.9 mil millones km",
        "year":"84 años",
        "fun_fact":"Es el planeta más frío."
    },

    "Neptune": {
        "description":"Neptuno tiene vientos extremos.",
        "color":"Azul intenso",
        "gravity":"11.15 m/s²",
        "temperature":"-214°C",
        "distance":"4.5 mil millones km",
        "year":"165 años",
        "fun_fact":"Sus vientos superan los 2000 km/h."
    },

    "Pluto": {
        "description":"Plutón es un planeta enano.",
        "color":"Marrón y blanco",
        "gravity":"0.62 m/s²",
        "temperature":"-229°C",
        "distance":"5.9 mil millones km",
        "year":"248 años",
        "fun_fact":"Fue considerado el noveno planeta."
    },

    "Moon": {
        "description":"La Luna es el satélite natural de la Tierra.",
        "color":"Gris",
        "gravity":"1.62 m/s²",
        "temperature":"-20°C",
        "distance":"384400 km desde la Tierra",
        "year":"27.3 días",
        "fun_fact":"Es el único cuerpo celeste visitado por humanos."
    }
}


@app.route("/", methods=["GET", "POST"])
def home():

    image_name = None
    planet_name = None
    confidence_score = None
    info = None

    if request.method == "POST":

        file = request.files["image"]

        if file:

            # Guardar imagen
            image_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                file.filename
            )

            file.save(image_path)

            image_name = file.filename

            # Crear array
            data = np.ndarray(
                shape=(1, 224, 224, 3),
                dtype=np.float32
            )

            # Abrir imagen
            image = Image.open(
                image_path
            ).convert("RGB")

            # Resize
            size = (224, 224)

            image = ImageOps.fit(
                image,
                size,
                Image.Resampling.LANCZOS
            )

            # Convertir a array
            image_array = np.asarray(image)

            # Normalizar
            normalized_image_array = (
                image_array.astype(np.float32) / 127.5
            ) - 1

            # Cargar al array
            data[0] = normalized_image_array

            # Predicción
            prediction = model.predict(data)

            index = np.argmax(prediction)

            class_name = class_names[index]

            confidence_score = float(
                prediction[0][index]
            )

            planet_name = class_name[2:].strip()

            # Info del planeta
            info = planet_info.get(
                planet_name,
                "Información no disponible."
            )

    return render_template(
        "index.html",
        image_name=image_name,
        planet_name=planet_name,
        confidence_score=confidence_score,
        info=info
    )


if __name__ == "__main__":
    app.run(debug=True)