# Docker Deployment for MVP AskNebula

This guide explains how to deploy the MVP AskNebula service using Docker and Docker Compose.

## Prerequisites

- Docker installed on your server
- Docker Compose installed on your server
- Your API keys and credentials ready for the `.env` file

## Setup

1. Clone the repository to your server:
   ```
   git clone https://github.com/yourusername/mvp-ask-nebula.git
   cd mvp-ask-nebula
   ```

2. Create your `.env` file from the template:
   ```
   cp .env.example .env
   ```

3. Edit the `.env` file with your actual API keys and credentials:
   ```
   nano .env
   ```

## Deployment

1. Build and start the container in detached mode:
   ```
   docker-compose up -d
   ```

2. Check the container status:
   ```
   docker-compose ps
   ```

3. View the logs:
   ```
   docker-compose logs
   ```

   To follow the logs in real-time:
   ```
   docker-compose logs -f
   ```

## Managing the Container

### Stop the service
```
docker-compose stop
```

### Restart the service
```
docker-compose restart
```

### Rebuild and restart (after code changes)
```
docker-compose up -d --build
```

### Completely remove the container
```
docker-compose down
```

## Data Persistence

The Docker Compose configuration is set up to preserve:

1. The `last_mention.json` file, which tracks which Twitter mentions have been processed
2. The `twitter_bot.log` file for logging

These files are mounted as volumes from your host system to the container.

## Troubleshooting

1. If the container fails to start, check the logs:
   ```
   docker-compose logs
   ```

2. If you update your code, remember to rebuild the container:
   ```
   docker-compose up -d --build
   ```

3. If you're having authentication issues, verify that your `.env` file is correctly mounted by checking:
   ```
   docker-compose exec asknebula cat .env
   ```

4. To enter the container for debugging:
   ```
   docker-compose exec asknebula bash
   ``` 