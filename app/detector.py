import cv2
import sys
import os
from datetime import datetime

def detectar_circulos(imagem_path):
    image = cv2.imread(imagem_path)
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # Detecta círculos
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

    print(f"✅ {total} círculos (barras) detectados")
    print(f"🖼️ Imagem salva em: {path_saida}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Informe o caminho da imagem como argumento.")
        sys.exit(1)
    detectar_circulos(sys.argv[1])