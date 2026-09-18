# ML API + Docker

Простой ML API на FastAPI с контейнеризацией.

## Стек
- FastAPI
- scikit-learn / XGBoost
- Docker (multi-stage build)
- Docker Compose

## Что сделано
- Multi-stage Dockerfile
- Виртуальное окружение в образе
- Запуск от non-root пользователя
- .dockerignore

## Запуск

```bash
docker compose up --build