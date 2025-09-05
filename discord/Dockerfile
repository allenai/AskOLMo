FROM python:3.11-slim

WORKDIR /discord

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "discord_chat.py"]