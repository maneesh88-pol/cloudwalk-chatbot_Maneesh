FROM python:3.11-slim
WORKDIR /app
LABEL maintainer="Maneesh Polavarapu <polavarapumaneesh.info@gmail.com>"
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
