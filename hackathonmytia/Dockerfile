FROM python:3.10-slim

# Instala dependências básicas e bibliotecas do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Cria diretório de trabalho
WORKDIR /app

# Copia os arquivos do projeto
COPY ./app /app

# Instala bibliotecas Python
RUN pip install --upgrade pip && \
    pip install opencv-python ultralytics matplotlib

# Comando padrão
CMD ["python", "detector.py"]