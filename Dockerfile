FROM python:3.10
LABEL prod "prediction"
EXPOSE 8000
ENV PROJECT_DIR /usr/local/src/app
ENV SOURCE_DIR /usr/local/src/src
COPY app ${PROJECT_DIR}  
COPY src ${SOURCE_DIR}  
COPY Pipfile ${PROJECT_DIR}/Pipfile
COPY Pipfile.lock ${PROJECT_DIR}/Pipfile.lock
WORKDIR ${PROJECT_DIR}
RUN ["pip", "install", "pipenv"]
RUN ["python", "-m", "pipenv", "install", "--deploy"]
ENTRYPOINT  ["python", "-m", "pipenv", "run", "uvicorn", "main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]  
