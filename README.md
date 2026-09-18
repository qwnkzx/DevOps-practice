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

<img src="images/Снимок экрана — 2026-09-18 в 12.14.35.png" width="700">

Видно, что образ весит почти 4гб (да, это ML но всё равно не очень приятно)

## Обратим внимание на 11 и 14 строку
