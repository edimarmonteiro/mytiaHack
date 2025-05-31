from flask import Flask, request, jsonify
import cv2
import os
from datetime import datetime

app = Flask(__name__)

def detectar_circulos(imagem_path):
    image = cv2.imread(imagem_path)
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=10,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=20
    )

    total = 0
    if circles is not None:
        circles = circles[0, :].astype("int")
        total = len(circles)
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)

    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_saida = f"outputs/circulos_{timestamp}.jpg"
    cv2.imwrite(path_saida, output)

    return total, path_saida

@app.route("/detectar", methods=["POST"])
def detectar():
    if 'imagem' not in request.files:
        return jsonify({"erro": "Arquivo 'imagem' não enviado"}), 400

    imagem = request.files['imagem']
    os.makedirs("uploads", exist_ok=True)
    caminho_imagem = os.path.join("uploads", imagem.filename)
    imagem.save(caminho_imagem)

    total, saida = detectar_circulos(caminho_imagem)

    return jsonify({
        "barras_detectadas": total,
        "imagem_processada": saida
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)