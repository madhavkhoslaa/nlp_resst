from flask import Flask
from flask_restful import Api ,Resource

app = Flask(__name__)
api = Api(app)

#api.add_resource(#classname
#endpoint)
api.add_resource(UserRegister,/register)

if __name__ == "__main__":
    app.run(debug=True)
