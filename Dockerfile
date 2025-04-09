FROM python:3.11-slim

WORKDIR /app

# Copy dependency files
COPY pyproject.toml .
COPY uv.lock .
COPY .env.example .

# Install dependencies directly with pip instead of using uv
RUN pip install --no-cache-dir tweepy python-dotenv langchain langchain-openai thirdweb-ai pydantic

# Copy application code
COPY *.py .
COPY .env.example .env

# Run the Twitter bot
CMD ["python", "twiiter_bot.py"] 