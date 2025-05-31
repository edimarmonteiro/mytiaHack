# mytiaHack

## Instalação do projeto

docker composer up --build

## Executando o projeto

### Caminho 1 - via terminal

curl -X POST http://localhost:5001/detectar \ -F "imagem=@caminho/para/sua/imagem.jpg"

### Caminho 2 - via Postman (ou semelhantes)

- Rota: http://localhost:5001/detectar
- Método: Post
- Parâmetros
    - imagem - file