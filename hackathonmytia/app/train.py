from ultralytics import YOLO

def treinar_modelo():
    model = YOLO("yolo11n.pt")  # modelo base leve

    model.train(
        data="barras_dataset/data.yaml",
        epochs=20,
        imgsz=640,
        batch=8,
        name="barra_circular"
    )

    print("\n✅ Treinamento finalizado! Pesos salvos em: runs/detect/barra_circular/weights/best.pt")

if __name__ == "__main__":
    treinar_modelo()
