# MVP AskNebula - AI-Powered Blockchain Data Service

MVP AskNebula is an AI-powered blockchain data service that provides detailed information about blockchain transactions, addresses, ENS names, and more. The project consists of two main components:

1. **AskNebula Core Service** - A backend service that uses LangChain and ThirdWeb to query blockchain data
2. **Twitter Bot** - A Twitter (X) bot that answers blockchain-related questions asked by users through mentions

## Features

- **Blockchain Data Retrieval**: Get detailed information about:
  - ENS name resolution
  - Wallet balances
  - Transaction details
  - Contract interactions
  - Suspicious activity detection
  - Cross-chain data
  
- **Intelligent Response Formatting**: 
  - Complex answers are automatically formatted as Twitter threads
  - Simple answers are provided as single tweets
  - Off-topic questions receive clever responses

- **Continuous Monitoring**:
  - The Twitter bot continuously monitors for mentions
  - Processes new questions as they arrive
  - Respects Twitter API rate limits

## Architecture

The project is organized into three main Python modules:

1. **askNebula.py** - Core blockchain data service using ThirdWeb and LangChain
2. **thread_creator.py** - Content formatting service for Twitter
3. **twiiter_bot.py** - Twitter API integration for monitoring and responding to mentions

## Setup

### Prerequisites

- Python 3.7+
- Twitter API credentials (API Key/Secret, Access Token/Secret, and Bearer Token)
- ThirdWeb API key for AskNebula service
- OpenAI API key

### Installation

1. Clone the repository
2. Install dependencies using `uv`:
   ```
   uv pip install tweepy python-dotenv langchain langchain-openai thirdweb-ai pydantic
   ```

3. Create a `.env` file with your API credentials (see `.env.example` for required variables)

## Usage

### Running the AskNebula Service Standalone

To use the AskNebula service directly:

```python
from askNebula import NebulaAgent

# Initialize the agent
agent = NebulaAgent()

# Run a query
response = agent.run("What's the address for vitalik.eth?")
print(response)
```

### Running the Twitter Bot

To start the bot, simply run:
```
python twiiter_bot.py
```

The bot will:
1. Authenticate with Twitter
2. Find the most recent mention and store its ID (without responding to it or any older mentions)
3. Check for new mentions approximately every 90 seconds (respecting Twitter API Basic tier rate limits)
4. Process any new mentions (created after the bot started) and reply with blockchain data
5. Track the last processed mention to avoid duplicate responses

## How It Works

### Core AskNebula Service

The `NebulaAgent` class in `askNebula.py` provides blockchain data by:

1. Using ThirdWeb's Nebula service to access on-chain data
2. Leveraging LangChain for agent-based interaction
3. Using specialized tools for different types of blockchain queries:
   - Balance queries
   - ENS resolution
   - Transaction details
   - Off-topic handling

### Twitter Content Formatting

The `ContentCreator` class in `thread_creator.py`:

1. Processes the user's blockchain query
2. Gets data from the AskNebula service
3. Determines the best format for the response:
   - Thread (3 tweets) for complex data
   - Single post for simple responses
   - Clever responses for off-topic queries
4. Formats the content according to Twitter's character limits and best practices

### Twitter Bot

The Twitter bot in `twiiter_bot.py`:

1. Monitors for new mentions
2. Extracts the blockchain query from the tweet text
3. Sends the query to the thread_creator module
4. Posts the formatted response as a reply to the original mention
5. Tracks processed mentions to avoid duplicate responses

## Rate Limits and Performance

The bot is designed to work within Twitter's Basic API tier limitations:
- The mentions endpoint allows 10 requests per 15 minutes
- The tweet creation endpoint allows 100 requests per 24 hours
- The bot sleeps for 90 seconds between mention checks to stay under rate limits

## Troubleshooting

### Logs

Check `twitter_bot.log` for detailed information about the bot's operation and any errors encountered.

### Common Issues

- **Rate limits**: If you encounter rate limit errors, the bot will automatically wait before trying again
- **Authentication issues**: Ensure your API keys and tokens are correct in the `.env` file
- **Missing dependencies**: Make sure all required packages are installed with `uv`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
