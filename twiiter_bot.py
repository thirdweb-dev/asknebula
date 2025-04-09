import tweepy
from dotenv import load_dotenv
import os
import json
import time
from datetime import datetime
import logging
import traceback

# Import thread_creator functionality
from thread_creator import process_blockchain_query

# Configure logging with a less verbose format
logging.basicConfig(
    level=logging.INFO,  # Changed to INFO to reduce verbosity
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("twitter_bot.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("TwitterBot")

load_dotenv()

# Keys and tokens
API_KEY = os.getenv("TWITTER_API_KEY")
API_SECRET = os.getenv("TWITTER_API_SECRET")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

# Log API credentials status (without revealing the actual keys)
logger.debug(f"API_KEY present: {bool(API_KEY)}")
logger.debug(f"API_SECRET present: {bool(API_SECRET)}")
logger.debug(f"ACCESS_TOKEN present: {bool(ACCESS_TOKEN)}")
logger.debug(f"ACCESS_TOKEN_SECRET present: {bool(ACCESS_TOKEN_SECRET)}")
logger.debug(f"BEARER_TOKEN present: {bool(BEARER_TOKEN)}")

# File to store the latest mention ID
LAST_MENTION_FILE = "last_mention.json"

# Bot's username (without the @ symbol)
BOT_USERNAME = "AskNebula"  # Updated with correct capitalization

# Authenticate to Twitter using v2 API
try:
    logger.info("Authenticating with Twitter API...")
    client = tweepy.Client(
        consumer_key=API_KEY,
        consumer_secret=API_SECRET,
        access_token=ACCESS_TOKEN,
        access_token_secret=ACCESS_TOKEN_SECRET,
        bearer_token=BEARER_TOKEN,
    )
    logger.info("Twitter API authentication successful")
except Exception as e:
    logger.error(f"Twitter API authentication failed: {e}")
    logger.error(traceback.format_exc())
    raise


def create_tweet(tweet_text: str):
    """Create a single tweet."""
    logger.debug(f"Creating tweet: {tweet_text[:30]}...")
    try:
        response = client.create_tweet(text=tweet_text)
        tweet_id = response.data["id"]
        logger.info(f"Created tweet with ID {tweet_id}")
        return response
    except Exception as e:
        logger.error(f"Failed to create tweet: {e}")
        logger.error(f"Full error: {traceback.format_exc()}")
        raise


def post_thread(thread_content, reply_to_id=None):
    """
    Post a thread of tweets, either as standalone or as a reply.

    Args:
        thread_content: Dictionary with tweet1, tweet2, tweet3 keys or an object with those attributes
        reply_to_id: Optional ID of tweet to reply to, or None for standalone thread

    Returns:
        List of response objects from tweet creation
    """
    logger.info("Creating Twitter thread...")

    # Handle both dictionary and object formats
    if isinstance(thread_content, dict):
        tweet1 = thread_content.get("tweet1", "")
        tweet2 = thread_content.get("tweet2", "")
        tweet3 = thread_content.get("tweet3", "")
    else:
        # Assume it's an object with attributes
        tweet1 = getattr(thread_content, "tweet1", "")
        tweet2 = getattr(thread_content, "tweet2", "")
        tweet3 = getattr(thread_content, "tweet3", "")

    try:
        # Post the first tweet (either standalone or as a reply)
        post_params = {"text": tweet1}
        if reply_to_id:
            post_params["in_reply_to_tweet_id"] = reply_to_id

        response1 = client.create_tweet(**post_params)
        tweet_id1 = response1.data["id"]
        logger.info(f"Created thread tweet 1 with ID {tweet_id1}")

        # Post the second tweet as a reply to the first
        response2 = client.create_tweet(text=tweet2, in_reply_to_tweet_id=tweet_id1)
        tweet_id2 = response2.data["id"]
        logger.info(f"Created thread tweet 2 with ID {tweet_id2}")

        # Post the third tweet as a reply to the second
        response3 = client.create_tweet(text=tweet3, in_reply_to_tweet_id=tweet_id2)
        tweet_id3 = response3.data["id"]
        logger.info(f"Created thread tweet 3 with ID {tweet_id3}")

        return [response1, response2, response3]
    except Exception as e:
        logger.error(f"Failed to create thread: {e}")
        logger.error(f"Full error: {traceback.format_exc()}")
        raise


def get_last_mention_id():
    """Get the ID of the last processed mention from the JSON file.

    Returns:
        The ID of the last mention or None if no mentions have been processed
    """
    logger.debug(f"Reading last mention ID from {LAST_MENTION_FILE}")
    try:
        with open(LAST_MENTION_FILE, "r") as f:
            data = json.load(f)
            last_id = data.get("last_mention_id")
            logger.debug(f"Read last mention ID: {last_id}")
            return last_id
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.debug(f"Could not read last mention ID: {e}")
        # If the file doesn't exist or is invalid, return None
        return None


def save_last_mention_id(mention_id):
    """Save the ID of the last processed mention to the JSON file.

    Args:
        mention_id: The ID of the last mention
    """
    logger.debug(f"Saving last mention ID: {mention_id}")
    data = {"last_mention_id": mention_id, "timestamp": datetime.now().isoformat()}
    try:
        with open(LAST_MENTION_FILE, "w") as f:
            json.dump(data, f)
        logger.debug(f"Successfully saved last mention ID")
    except Exception as e:
        logger.error(f"Failed to save last mention ID: {e}")
        logger.error(traceback.format_exc())


def get_most_recent_mention_id():
    """Get the ID of the most recent mention to use as a starting point.

    Returns:
        The ID of the most recent mention or None if no mentions exist
    """
    logger.debug("Fetching most recent mention ID")
    try:
        # Query parameters for mentions timeline - simplified to remove fields causing warnings
        params = {
            "expansions": "author_id",
            "max_results": 5,  # We only need the most recent one
        }

        # Get mentions
        mentions = client.get_users_mentions(id=client.get_me()[0].id, **params)

        # If no mentions, return None
        if not mentions.data:
            logger.debug("No mentions found")
            return None

        # Get the most recent mention (highest ID)
        latest_mention = max(mentions.data, key=lambda x: x.id)
        logger.debug(f"Found most recent mention ID: {latest_mention.id}")
        return latest_mention.id
    except Exception as e:
        logger.error(f"Error getting most recent mention: {e}")
        logger.error(traceback.format_exc())
        return None


def get_recent_mentions():
    """Get recent mentions of the authenticated user.

    Returns:
        A list of mentions, sorted from oldest to newest
    """
    last_mention_id = get_last_mention_id()
    logger.debug(f"Checking for mentions since ID: {last_mention_id}")

    # Query parameters for mentions timeline - simplified to remove fields causing warnings
    params = {
        "expansions": "author_id",
    }

    # Add since_id parameter if we have a last mention ID
    if last_mention_id:
        params["since_id"] = last_mention_id

    # Get mentions
    try:
        mentions = client.get_users_mentions(id=client.get_me()[0].id, **params)

        # If no new mentions, return empty list
        if not mentions.data:
            return []

        # Return mentions sorted by ID (oldest first)
        sorted_mentions = sorted(mentions.data, key=lambda x: x.id)
        if sorted_mentions:
            logger.info(f"Found {len(sorted_mentions)} new mentions")
        return sorted_mentions
    except Exception as e:
        logger.error(f"Error getting mentions: {e}")
        logger.error(traceback.format_exc())
        return []


def is_direct_mention(tweet_text):
    """Check if the tweet directly mentions the bot's username.

    This ensures we only respond to tweets that explicitly tag the bot,
    not replies to the bot's tweets where the bot is auto-tagged.
    Case-insensitive to handle different capitalizations of the username.

    Args:
        tweet_text: The text of the tweet

    Returns:
        Boolean indicating if the bot is directly mentioned
    """
    # Convert both the tweet text and the username to lowercase for case-insensitive comparison
    tweet_text_lower = tweet_text.lower()
    bot_username_lower = BOT_USERNAME.lower()

    # Check if @BOT_USERNAME is in the tweet text (case-insensitive)
    is_mention = f"@{bot_username_lower}" in tweet_text_lower
    return is_mention


def process_mentions():
    """Process new mentions and respond to them."""
    mentions = get_recent_mentions()

    if not mentions:
        logger.info("No new mentions found")
        return

    # Process each mention
    for mention in mentions:
        logger.info(f"Processing mention ID {mention.id}")
        try:
            # Extract the tweet text
            tweet_text = mention.text

            # Check if this is a direct mention of the bot (not a reply)
            if not is_direct_mention(tweet_text):
                logger.info(f"Skipping mention {mention.id} - not a direct mention")
                save_last_mention_id(mention.id)  # Save ID to avoid reprocessing
                continue

            # Remove the @username part from the tweet text
            # This helps process the actual query without the mention
            clean_text = " ".join(
                [
                    word
                    for word in tweet_text.split()
                    if not word.lower().startswith("@" + BOT_USERNAME.lower())
                ]
            )

            logger.info(f"Processing query: {clean_text}")

            # Process the tweet content using thread_creator
            result = process_blockchain_query(clean_text, verbose=False)
            logger.info(f"Result type: {result['type']}")

            # Reply based on content type
            reply_id = mention.id
            if result["type"] == "thread":
                # Post a thread in reply to the mention
                logger.info(f"Creating thread in response to mention {mention.id}")
                responses = post_thread(result["content"], reply_id)
                logger.info(f"Created thread in response to mention {mention.id}")
            else:
                # Post a single tweet in reply to the mention
                logger.info(f"Creating reply to mention {mention.id}")
                response = client.create_tweet(
                    text=result["content"]["post"], in_reply_to_tweet_id=reply_id
                )
                logger.info(f"Created reply to mention {mention.id}")

            # Save the ID of the processed mention
            save_last_mention_id(mention.id)

            # Sleep to avoid rate limiting
            time.sleep(2)

        except Exception as e:
            logger.error(f"Error processing mention {mention.id}: {e}")
            logger.error(traceback.format_exc())

            # Save the ID even if processing failed to avoid getting stuck
            save_last_mention_id(mention.id)
            continue


def initialize_bot():
    """Initialize the bot by setting the last mention ID to the most recent mention.
    This ensures we only respond to new mentions after the bot starts.
    """
    # Check if we already have a last mention ID file
    if get_last_mention_id() is None:
        logger.info(
            "No last mention ID found. Getting most recent mention as starting point..."
        )

        # Get the most recent mention ID
        most_recent_id = get_most_recent_mention_id()

        if most_recent_id:
            # Save the most recent mention ID
            logger.info(f"Found most recent mention ID: {most_recent_id}")
            save_last_mention_id(most_recent_id)
            logger.info("Bot will only respond to new mentions from now on.")
        else:
            logger.info(
                "No existing mentions found. Bot will respond to all new mentions."
            )
    else:
        logger.info(
            f"Bot already initialized with last mention ID: {get_last_mention_id()}"
        )


def run_bot():
    """Main function to run the Twitter bot continuously."""
    logger.info("Starting Twitter bot - monitoring for mentions")

    # Get the authenticated user info
    try:
        user = client.get_me()[0]
        logger.info(f"Authenticated as @{user.username} (ID: {user.id})")

        # Set the bot username from the authenticated user if not hardcoded
        global BOT_USERNAME
        if (
            user.username != BOT_USERNAME
            and user.username.lower() != BOT_USERNAME.lower()
        ):
            logger.info(f"Updating bot username from {BOT_USERNAME} to {user.username}")
            BOT_USERNAME = user.username
        logger.info(f"Bot will respond to @{BOT_USERNAME} mentions")
    except Exception as e:
        logger.error(f"Failed to get authenticated user: {e}")
        logger.error(traceback.format_exc())
        raise

    # Initialize the bot to only listen for new mentions
    initialize_bot()

    # Continuous loop to check for mentions
    mention_check_count = 0
    while True:
        try:
            mention_check_count += 1
            logger.info(f"Check #{mention_check_count}: Checking for new mentions...")
            process_mentions()

            # Sleep for 90 seconds to respect rate limits
            # Basic tier allows 10 requests per 15 minutes for mentions endpoint
            # This gives us a safe ~10 requests per 15 minutes (900 seconds)
            logger.info(f"Waiting for next check... (90 seconds)")
            time.sleep(90)

        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            logger.error(traceback.format_exc())
            # If we hit an unexpected error, wait a bit longer before trying again
            logger.info("Waiting 300 seconds before retrying...")
            time.sleep(300)


# Function to test thread creation
def test_thread_creation(thread_content=None):
    """Test function to verify thread creation works correctly"""
    logger.info("Running thread creation test...")

    if not thread_content:
        # Sample thread content
        thread_content = {
            "tweet1": "This is the first tweet in our test thread. #blockchain",
            "tweet2": "This is the second tweet continuing our discussion.",
            "tweet3": "This is the third and final tweet in our thread.",
        }

    try:
        # Create the thread
        logger.info("Attempting to create a test thread...")
        responses = post_thread(thread_content)
        logger.info("Test thread created successfully!")
        return responses
    except Exception as e:
        logger.error(f"Test thread creation failed: {e}")
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    try:
        logger.info("=== Twitter Bot Starting ===")
        run_bot()
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        logger.critical(traceback.format_exc())
        raise
