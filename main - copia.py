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

    "Earth": {
        "description":
        "La Tierra es el único planeta conocido con vida.",

        "color":
        "Azul y verde"
    },

    "Mars": {
        "description":
        "Marte es conocido como el planeta rojo.",

        "color":
        "Rojo"
    },

    "Jupiter": {
        "description":
        "Júpiter es el planeta más grande del sistema solar.",

        "color":
        "Marrón y blanco"
    },

    "Saturn": {
        "description":
        "Saturno es famoso por sus anillos.",

        "color":
        "Dorado"
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