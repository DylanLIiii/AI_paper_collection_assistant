# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    cron \
    tzdata \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set timezone to EST
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"

# Set the working directory in the container
WORKDIR /app

# Copy project files
COPY . .

# Create required directories
RUN mkdir -p /app/out/cache

# Set environment variables
ENV GEMINI_API_KEY=""
ENV SLACK_KEY=""
ENV SLACK_CHANNEL_ID=""
ENV FLASK_APP="paper_assistant.api.app:create_app()"
ENV FLASK_ENV=production

# Expose the port the app runs on
EXPOSE 8000

# Install dependencies using pixi
RUN pixi install

# Create start script
RUN echo "#!/bin/bash\n\
if [ ! -f /app/out/output.json ]; then\n\
    echo 'Running initial paper generation...'\n\
    pixi run python -m paper_assistant.cli.commands generate\n\
fi\n\
cron\n\
pixi run gunicorn --bind 0.0.0.0:8000 --workers 2 --timeout 120 'paper_assistant.api.app:create_app()'" > /app/start.sh

RUN chmod +x /app/start.sh

# Command to run the application
CMD ["/app/start.sh"]
