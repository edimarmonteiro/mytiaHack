FROM python:3.10-slim

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Define diretório de trabalho
WORKDIR /app

# Copia os arquivos do projeto para dentro do container
COPY ./app /app

# Instala bibliotecas Python necessárias
RUN pip install --upgrade pip && \
    pip install flask opencv-python-headless ultralytics matplotlib

# Expõe a porta usada pelo Flask
EXPOSE 5001

# Comando para iniciar a API Flask
CMD ["python", "app.py"]