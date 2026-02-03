from flask import Flask, render_template, jsonify
import pandas as pd
import os

app = Flask(__name__)

# Ruta principal: Carga el HTML
@app.route('/')
def index():
    return render_template('index.html')

# API: Envía los datos del CSV al mapa
@app.route('/get_data')
def get_data():
    path = os.path.join(app.root_path, 'predictions.csv')
    df = pd.read_csv(path)
    # Convertimos a JSON para que el JS del mapa lo procese
    return jsonify(df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=False) # Recordar: False para producción