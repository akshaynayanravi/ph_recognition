from flask import Blueprint, Flask, render_template, request
from ph_recognition import get_ph_type
from flask import Blueprint, Flask, render_template, request

from ph_recognition import get_ph_type

FLASK_HOST = "127.0.0.1"
FLASK_PORT = "5000"
DEBUG = True

ph_classifier_app = Blueprint("ph_classifier", __name__)


@ph_classifier_app.route("/")
def home():
    return render_template("index.html")


@ph_classifier_app.route("/", methods=["POST"])
def classify():
    # print(request.form)
    # int_features = [int(x) for x in request.form.values()]
    red = int(request.form.get("red"))
    green = int(request.form.get("green"))
    blue = int(request.form.get("blue"))

    ph_type = get_ph_type(red=red, green=green, blue=blue)

    return render_template("index.html", ph_type=f"PH Type is {ph_type}")


if __name__ == "__main__":
    app = Flask(__name__)
    app.register_blueprint(ph_classifier_app)
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=DEBUG)
    logger.info(
        f"Flask server started: Running on - http://{FLASK_HOST}:{FLASK_PORT}"
        + f" | Debug - {DEBUG} | CONFIG - {CONFIG}"
        + f" | app - {app.name}"
    )
