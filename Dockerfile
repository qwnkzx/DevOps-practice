FROM python:3.12-slim AS builder

WORKDIR /ml_api

RUN python -m venv /ml/project

ENV PATH="/ml/project/bin:$PATH"

COPY requirements.txt . 

RUN pip install --no-cache-dir -r requirements.txt 

#-------------------------------------------------------------------------------

FROM python:3.12-slim AS second

WORKDIR /ml_api 

COPY --from=builder /ml/project /ml/project
ENV PATH="/ml/project/bin:$PATH"

COPY /src/app.py src/predict.py ./  

RUN useradd --create-home --shell /bin/bash appuser && \ 
    chown -R appuser:appuser /ml_api

USER appuser 

EXPOSE 8000

ENTRYPOINT [ "python", "app.py" ]