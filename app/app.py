from flask import Flask, request, jsonify
import cv2
import os
import numpy as np
from datetime import datetime
import base64

app = Flask(__name__)

def image_to_base64(image):
    """Convert an image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def detectar_circulos(imagem_path):
    # Read and create a copy of the image
    image = cv2.imread(imagem_path)
    output = image.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge enhancement
    gray = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
    gray = cv2.convertScaleAbs(gray)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,           # Reduced from 1.2 to 1 for better accuracy
        minDist=20,     # Increased minimum distance between circles
        param1=30,      # Reduced from 50 to be more sensitive
        param2=25,      # Reduced from 30 to detect more circles
        minRadius=8,    # Adjusted radius range
        maxRadius=25
    )

    barras_info = []
    if circles is not None:
        # Convert coordinates to integers
        circles = np.round(circles[0, :]).astype("int")
        
        # Sort circles by x and y position for more organized numbering
        circles = sorted(circles, key=lambda x: (x[1] // 30, x[0]))
        
        for idx, (x, y, r) in enumerate(circles, 1):
            # Draw circle outline
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            
            # Draw filled background for text
            text = f"#{idx}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(output, (x-10, y-r-25), (x+text_width, y-r-5), (0, 0, 0), -1)
            
            # Draw ID number with better visibility
            cv2.putText(output, text, (x-10, y-r-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Store information about each bar
            barra_info = {
                "id": idx,
                "posicao_x": int(x),
                "posicao_y": int(y),
                "raio": int(r),
                "diametro_mm": round(r * 2 * 0.264583, 2)  # Assuming pixels to mm conversion
            }
            barras_info.append(barra_info)

    # Convert images to base64
    original_base64 = image_to_base64(image)
    preprocessed_base64 = image_to_base64(gray)
    output_base64 = image_to_base64(output)

    return {
        "total": len(barras_info),
        "imagens": {
            "original": f"data:image/jpeg;base64,{original_base64}",
            "preprocessada": f"data:image/jpeg;base64,{preprocessed_base64}",
            "final": f"data:image/jpeg;base64,{output_base64}"
        }
    }

@app.route("/detectar", methods=["POST"])
def detectar():
    if 'imagem' not in request.files:
        return jsonify({"erro": "Arquivo 'imagem' não enviado"}), 400

    imagem = request.files['imagem']
    os.makedirs("uploads", exist_ok=True)
    caminho_imagem = os.path.join("uploads", imagem.filename)
    imagem.save(caminho_imagem)

    resultado = detectar_circulos(caminho_imagem)
    
    # Clean up the uploaded file
    os.remove(caminho_imagem)

    return jsonify(resultado)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)