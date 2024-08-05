from flask import Blueprint

main_bp = Blueprint('main', __name__)
upload_bp = Blueprint('upload', __name__)

from . import main, upload
