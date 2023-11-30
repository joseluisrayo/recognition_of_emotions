from flask import Blueprint, request, jsonify

import traceback

# Logger
from src.utils.Logger import Logger
# Models
from src.models.UserModel import User
# Security
from src.utils.Security import Security
# Services
from src.services.AuthService import AuthService

main = Blueprint('auth_blueprint', __name__)


@main.route('/', methods=['POST'])
def login():
    try:
        username = request.json['username']
        password = request.json['password']

        if (str(username) == 'admin' and str(password) == "123456"):
            return jsonify({'success': True, 'url': 'dashboard'})
        else:
            response = jsonify({'success': False})
            return response
    except Exception as ex:
        Logger.add_to_log("error", str(ex))
        Logger.add_to_log("error", traceback.format_exc())

        return jsonify({'message': "ERROR", 'success': False})