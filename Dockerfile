FROM python:3.11-slim

WORKDIR /discord

COPY pyproject.toml .

# Download project dependencies
RUN pip install . --no-cache-dir && \
    pip uninstall -y askolmo

# Install project
COPY . .
RUN pip install --no-cache-dir --no-deps .

CMD ["askolmo"]
