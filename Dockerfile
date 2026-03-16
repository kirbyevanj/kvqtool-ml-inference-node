FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir grpcio-tools

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY proto/ proto/
RUN python -m grpc_tools.protoc \
    -Iproto \
    --python_out=proto \
    --grpc_python_out=proto \
    proto/inference.proto

COPY app/ app/

EXPOSE 50051

CMD ["python", "-m", "app.main"]
