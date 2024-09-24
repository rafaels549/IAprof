from flask import Flask, request, jsonify
from flask_cors import CORS 
from use_model import gerar_texto 

app = Flask(__name__)


CORS(app)

@app.route('/profissao', methods=['POST'])
def profissao():
    data = request.get_json()  
    profissao = data.get('profissao')  

    if not profissao:
        return jsonify({'error': 'Profissão não fornecida!'}), 400  
    
    resultado = gerar_texto(profissao)
    return jsonify({'message': resultado}), 200 

if __name__ == '__main__':
    app.run(debug=True)
