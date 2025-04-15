import json
import os
from typing import Dict, List, Any, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import NebulaAgent from askNebula.py
from askNebula import NebulaAgent

# Load ENV
load_dotenv()


class TwitterContent(BaseModel):
    """Response content for Twitter, either a thread or a single post."""

    type: str = Field(..., description="Type of content: 'thread' or 'post'")
    content: Dict[str, str] = Field(..., description="Content of the post or thread")


class TwitterThread(BaseModel):
    """Model for Twitter thread with exactly three tweets."""

    tweet1: str = Field(..., description="Content of the first tweet in the thread")
    tweet2: str = Field(..., description="Content of the second tweet in the thread")
    tweet3: str = Field(..., description="Content of the third tweet in the thread")


class ContentCreator:
    """A class that creates Twitter content using LangChain agents."""

    def __init__(self, api_key=None, model="gpt-4o-mini", verbose=False):
        """Initialize the ContentCreator.

        Args:
            api_key: The API key for OpenAI. If None, it will be read from the OPENAI_API_KEY environment variable.
            model: The model to use for the LLM.
            verbose: Whether to print verbose output.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.verbose = verbose

        # Initialize LLM
        self.llm = ChatOpenAI(model=model)

        # Initialize NebulaAgent for blockchain data
        self.nebula_agent = NebulaAgent(verbose=False)

        # Create a prompt template with system instructions
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are an AI assistant that creates Twitter content for blockchain-related information.
                    
                    Given a user's blockchain query:
                    1. First query the AskNebula agent to get information about the query
                    2. Analyze the response and decide what type of content is appropriate:
                       - For ALL transaction data (including simple transactions): Always create a 3-tweet thread
                       - For complex blockchain data (like details about projects, tokens, etc.): Create a 3-tweet thread
                       - For very simple responses (like ENS resolution): Create a single post
                       - For off-topic matters: Create a clever, funny response
                    
                    Use the most appropriate tool for each scenario:
                    - create_twitter_thread: For ANY transaction data, no matter how simple, and for complex blockchain data
                    - create_twitter_post: Only for very simple blockchain data that has no transaction details
                    - create_clever_response: For off-topic or humorous responses
                    
                    Never make up blockchain data. Always get the data from the AskNebula agent first.
                    """,
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        # Initialize tools
        self._initialize_tools()

        # Create the agent
        self._create_agent()

    def _initialize_tools(self):
        """Initialize the tools for the agent."""

        @tool
        def query_nebula(query: str) -> str:
            """Query the AskNebula agent to get blockchain-related information.

            Args:
                query: A question about blockchain, like asking about a transaction, wallet, ENS name, etc.

            Returns:
                The response from the AskNebula agent
            """
            if self.verbose:
                print(f"Querying AskNebula with: {query}")
            response = self.nebula_agent.run(query)
            if self.verbose:
                print(f"AskNebula response received (length: {len(response)})")
            return response

        @tool
        def create_twitter_thread(blockchain_data: str) -> Dict[str, str]:
            """Create a Twitter thread with exactly 3 tweets about complex blockchain information.

            Args:
                blockchain_data: The blockchain data to create a thread about (like transaction details)

            Returns:
                A dictionary with three tweets (tweet1, tweet2, tweet3)
            """
            if self.verbose:
                print("Creating Twitter thread...")

            # Define thread creation prompt
            thread_prompt = """
            Write a Twitter thread about the following blockchain information in exactly 3 tweets:
            
            {blockchain_data}
            
            Requirements:
            - The first tweet MUST include a simple plain English explanation of what this blockchain data shows
            - For transactions, the first tweet should clearly explain what the transaction is doing in everyday language
            - NO hashtags whatsoever
            - Use bullet points (•) for better readability
            - Each bullet point should be on its own line
            - NO markdown formatting (no backticks, no asterisks for bold)
            - For technical data like addresses, use clear labels: "From: 0x123..." not "**From**: `0x123...`"
            - Easy-to-read content with clear keywords and concise language
            - Natural flow between tweets, the transition between tweets should be seamless
            - Each Tweet should be about the same length and try to be as concise as possible
            - Use emojis to make the thread more engaging
            - Twitter has character limits, so keep each tweet under 280 characters
            - Use proper spacing between sentences
            - Format numbers with commas for better readability (e.g., "1,234,567" not "1234567")
            - Use proper units (e.g., "ETH" for Ether values)
            - Keep technical details clear but concise
            - If the information is about a transaction, you MUST include a link to the block explorer with the COMPLETE transaction hash in the last tweet
            - DO NOT truncate the transaction hash in the block explorer URL - include the FULL hash
            - For transactions, use the appropriate block explorer: 
              - Ethereum: https://etherscan.io/tx/0x[full transaction hash]
              - Base: https://basescan.org/tx/0x[full transaction hash]
              - Polygon: https://polygonscan.com/tx/0x[full transaction hash]
              - Arbitrum: https://arbiscan.io/tx/0x[full transaction hash]
              - Optimism: https://optimistic.etherscan.io/tx/0x[full transaction hash]
              
            Format:
            Return the thread as plain text with each tweet on a new line, separated by line breaks:

            Tweet 1: [First tweet content with plain English explanation of what the data shows]

            Tweet 2: [Second tweet content]

            Tweet 3: [Third tweet content with block explorer link using the complete transaction hash]
            """

            # Get the model response
            prompt = thread_prompt.format(blockchain_data=blockchain_data)
            response = self.llm.invoke(prompt)
            thread_text = response.content

            # Parse and clean the response
            thread_data = self._parse_thread_from_text(thread_text)
            thread_data["tweet1"] = self._clean_tweet(thread_data["tweet1"])
            thread_data["tweet2"] = self._clean_tweet(thread_data["tweet2"])
            thread_data["tweet3"] = self._clean_tweet(thread_data["tweet3"])

            # Extract transaction hash if present in the data
            tx_hash = self._extract_transaction_hash(blockchain_data)

            # If this is a transaction and the third tweet doesn't include a link,
            # try to add one to the appropriate block explorer
            if tx_hash and "https://" not in thread_data["tweet3"]:
                # Try to determine the blockchain
                blockchain = self._determine_blockchain(blockchain_data)
                explorer_url = self._get_explorer_url(blockchain, tx_hash)

                # Add the explorer URL to tweet3 if it's not too long
                current_length = len(thread_data["tweet3"])
                url_text = f"\nView on {blockchain}: {explorer_url}"

                if current_length + len(url_text) <= 280:
                    thread_data["tweet3"] = thread_data["tweet3"] + url_text
                elif current_length <= 240:
                    # Truncate if needed to fit url
                    thread_data["tweet3"] = (
                        thread_data["tweet3"][:240] + "...\n" + explorer_url
                    )
            # Check if the tweet already has a URL but it might be truncated
            elif tx_hash and "https://" in thread_data["tweet3"]:
                # Check if the full transaction hash is in the tweet
                if tx_hash not in thread_data["tweet3"]:
                    # Determine the blockchain
                    blockchain = self._determine_blockchain(blockchain_data)
                    explorer_url = self._get_explorer_url(blockchain, tx_hash)

                    # Replace any truncated or incorrect block explorer URLs with the correct one
                    import re

                    explorer_patterns = [
                        r"https://etherscan\.io/tx/0x[a-fA-F0-9]+",
                        r"https://basescan\.org/tx/0x[a-fA-F0-9]+",
                        r"https://polygonscan\.com/tx/0x[a-fA-F0-9]+",
                        r"https://arbiscan\.io/tx/0x[a-fA-F0-9]+",
                        r"https://optimistic\.etherscan\.io/tx/0x[a-fA-F0-9]+",
                    ]

                    for pattern in explorer_patterns:
                        if re.search(pattern, thread_data["tweet3"]):
                            thread_data["tweet3"] = re.sub(
                                pattern, explorer_url, thread_data["tweet3"]
                            )
                            break

            if self.verbose:
                print("Twitter thread created successfully")

            return thread_data

        @tool
        def create_twitter_post(blockchain_data: str) -> str:
            """Create a single Twitter post for simple blockchain information that fits in one tweet.

            Args:
                blockchain_data: The simple blockchain data to create a post about (like an ENS resolution)

            Returns:
                A string with the tweet content
            """
            if self.verbose:
                print("Creating single Twitter post...")

            # Define post creation prompt
            post_prompt = """
            Write a single Twitter post about the following blockchain information:
            
            {blockchain_data}
            
            Requirements:
            - MUST include a plain English explanation of what this blockchain data shows
            - For transactions, clearly explain what the transaction is doing in everyday language
            - NO hashtags or any trending topics with # - they are absolutely not allowed
            - Keep it under 280 characters
            - NO markdown formatting (no backticks, no asterisks for bold)
            - For technical data like addresses, use clear labels: "Address: 0x123..."
            - Format numbers with commas for better readability (e.g., "1,234,567" not "1234567")
            - Use proper units (e.g., "ETH" for Ether values)
            - Be concise and informative
            - Do not include any #hashtags at all - they are forbidden
            - If it's a transaction, include a brief explanation of what's happening
            - Use emojis to make the post more engaging
            
            Format:
            Return just the tweet content without any prefixes like "Tweet:" or "Post:"
            """

            # Get the model response
            prompt = post_prompt.format(blockchain_data=blockchain_data)
            response = self.llm.invoke(prompt)
            post_text = response.content.strip()

            # Clean the post
            post_text = self._clean_tweet(post_text)

            # Remove any hashtags that might have slipped through
            post_text = self._remove_hashtags(post_text)

            if self.verbose:
                print("Twitter post created successfully")

            return post_text

        @tool
        def create_clever_response(topic: str) -> str:
            """Create a clever, possibly humorous response for off-topic or non-blockchain matters.

            Args:
                topic: The off-topic query or subject

            Returns:
                A string with a clever response
            """
            if self.verbose:
                print("Creating clever response...")

            # Define clever response prompt
            clever_prompt = """
            Create a clever, possibly humorous Twitter response about the following topic:
            
            {topic}
            
            Requirements:
            - Keep it under 280 characters
            - Be friendly and engaging
            - If at all possible, make a subtle reference to blockchain or crypto
            - Don't be condescending even if the topic is unrelated to blockchain
            - Use humor when appropriate
            - NO hashtags or any words with # in front of them - hashtags are not allowed
            
            Format:
            Return just the tweet content without any prefixes or hashtags
            """

            # Get the model response
            prompt = clever_prompt.format(topic=topic)
            response = self.llm.invoke(prompt)
            clever_text = response.content.strip()

            # Clean the text
            clever_text = self._clean_tweet(clever_text)

            # Remove any hashtags that might have slipped through
            clever_text = self._remove_hashtags(clever_text)

            if self.verbose:
                print("Clever response created successfully")

            return clever_text

        @tool
        def analyze_nebula_response(response: str) -> Dict[str, Any]:
            """Analyze the response from Nebula to determine what kind of Twitter content to create.

            Args:
                response: The response from the AskNebula agent

            Returns:
                A dictionary with the analysis results
            """
            if self.verbose:
                print("Analyzing response to determine content type...")

            # Define analysis prompt
            analysis_prompt = """
            Analyze the following response from a blockchain information service and determine what kind of Twitter content should be created:
            
            {response}
            
            Your task is to determine:
            1. The type of content that should be created: 'thread', 'post', or 'clever'
            2. A brief explanation of why you chose this content type
            
            Use these guidelines:
            - 'thread': For ANY transaction data (even simple single transactions), and for complex blockchain data with multiple data points
            - 'post': For only very simple blockchain data that fits in one tweet (like ENS resolution, simple balance checks) and has NO transaction details
            - 'clever': For off-topic matters or when the response seems to be addressing a non-technical question
            
            IMPORTANT: If the response mentions ANY transaction hash (e.g., 0x followed by hexadecimal characters), ALWAYS choose 'thread'.
            If words like "transaction", "tx", "transfer", or "sent" appear in the response, ALWAYS choose 'thread'.
            
            Respond in JSON format with the following structure:
            {{
                "content_type": [thread, post, or clever],
                "explanation": [brief explanation],
                "is_transaction": [true or false],
                "has_technical_details": [true or false]
            }}
            """

            # Get the model response
            prompt = analysis_prompt.format(response=response)
            analysis_response = self.llm.invoke(prompt)

            # Parse the JSON response
            try:
                # Extract JSON from the response - find anything that looks like JSON
                json_text = ""
                in_json = False
                for line in analysis_response.content.split("\n"):
                    line = line.strip()
                    if line.startswith("{") or in_json:
                        in_json = True
                        json_text += line
                        if line.endswith("}"):
                            break

                if not json_text:
                    json_text = analysis_response.content

                analysis_result = json.loads(json_text)

                if self.verbose:
                    print(
                        f"Analysis result: content_type={analysis_result.get('content_type', 'unknown')}"
                    )

                return analysis_result
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                if self.verbose:
                    print("Failed to parse analysis response, using fallback")

                return {
                    "content_type": "post",
                    "explanation": "Fallback to post due to parsing error",
                    "is_transaction": False,
                    "has_technical_details": False,
                }

        # Store the tools
        self.tools = [
            query_nebula,
            create_twitter_thread,
            create_twitter_post,
            create_clever_response,
            analyze_nebula_response,
        ]

    def _create_agent(self):
        """Create the agent and agent executor."""
        # Create the agent
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            handle_parsing_errors=True,
            verbose=self.verbose,
            return_intermediate_steps=True,
        )

    def process_query(self, query: str) -> TwitterContent:
        """Process a blockchain query and create appropriate Twitter content.

        Args:
            query: The blockchain query to process

        Returns:
            A TwitterContent object with the appropriate type and content
        """
        if self.verbose:
            print(f"\n\nQuery: {query}")
            print("-" * 50)

        # Run the agent
        response = self.agent_executor.invoke({"input": query})

        if self.verbose:
            print("\nFinal Response:")
            print(response["output"])

            # Print intermediate steps
            print("\nIntermediate Steps:")
            for i, step in enumerate(response["intermediate_steps"]):
                print(f"\nStep {i+1}:")
                tool_name = step[0].tool
                tool_input = step[0].tool_input
                tool_output = step[1]

                print(f"Tool: {tool_name}")
                if isinstance(tool_input, str) and len(tool_input) > 100:
                    print(f"Input: {tool_input[:100]}...")
                else:
                    print(f"Input: {tool_input}")

                if isinstance(tool_output, str) and len(tool_output) > 100:
                    print(f"Output: {tool_output[:100]}...")
                elif isinstance(tool_output, dict):
                    print(f"Output: {json.dumps(tool_output, indent=2)}")
                else:
                    print(f"Output: {tool_output}")

        # Extract content from the agent's response
        content_type = "post"  # Default
        content = {}

        # Look through the intermediate steps to find the content
        for step in response["intermediate_steps"]:
            tool_name = step[0].tool
            tool_output = step[1]

            if tool_name == "create_twitter_thread" and isinstance(tool_output, dict):
                content_type = "thread"

                # Extract the transaction hash if present
                tx_hash = None
                if (
                    isinstance(step[0].tool_input, dict)
                    and "blockchain_data" in step[0].tool_input
                ):
                    tx_hash = self._extract_transaction_hash(
                        step[0].tool_input["blockchain_data"]
                    )

                # Determine the chain if it's a transaction
                blockchain = None
                if (
                    tx_hash
                    and isinstance(step[0].tool_input, dict)
                    and "blockchain_data" in step[0].tool_input
                ):
                    blockchain = self._determine_blockchain(
                        step[0].tool_input["blockchain_data"]
                    )

                # Fix the URL in tweet3 if needed
                if tx_hash and blockchain and "tweet3" in tool_output:
                    # Get correct explorer URL
                    explorer_url = self._get_explorer_url(blockchain, tx_hash)

                    # Check if the correct URL is already there
                    if explorer_url not in tool_output["tweet3"]:
                        # URL pattern to find and replace
                        import re

                        url_pattern = r"https?://[^\s]+"
                        if re.search(url_pattern, tool_output["tweet3"]):
                            # Replace incorrect URL with correct one
                            tool_output["tweet3"] = re.sub(
                                url_pattern, explorer_url, tool_output["tweet3"]
                            )
                        else:
                            # Add the URL if none exists
                            tool_output["tweet3"] = (
                                tool_output["tweet3"].rstrip()
                                + f"\n\nView on {blockchain}: {explorer_url}"
                            )

                content = tool_output
                break
            elif tool_name == "create_twitter_post" and isinstance(tool_output, str):
                content_type = "post"
                content = {"post": tool_output}
                break
            elif tool_name == "create_clever_response" and isinstance(tool_output, str):
                content_type = "post"
                content = {"post": tool_output}
                break

        # If we didn't find content in the steps, try to parse it from the output
        if not content:
            if "tweet1" in response["output"] or "Tweet 1:" in response["output"]:
                content_type = "thread"
                content = self._parse_thread_from_text(response["output"])

                # Try to extract transaction hash from the query
                tx_hash = self._extract_transaction_hash(query)
                if tx_hash and "tweet3" in content:
                    # Try to determine the blockchain from the query
                    blockchain = self._determine_blockchain(query)
                    explorer_url = self._get_explorer_url(blockchain, tx_hash)

                    # Check if the correct URL is already there
                    if explorer_url not in content["tweet3"]:
                        # Replace or add URL
                        import re

                        url_pattern = r"https?://[^\s]+"
                        if re.search(url_pattern, content["tweet3"]):
                            content["tweet3"] = re.sub(
                                url_pattern, explorer_url, content["tweet3"]
                            )
                        else:
                            content["tweet3"] = (
                                content["tweet3"].rstrip()
                                + f"\n\nView on {blockchain}: {explorer_url}"
                            )
            else:
                content_type = "post"
                content = {"post": self._clean_tweet(response["output"])}

        return TwitterContent(type=content_type, content=content)

    def _clean_tweet(self, tweet: str) -> str:
        """Clean a tweet by removing markdown and fixing formatting but preserving emojis.

        Args:
            tweet: The tweet to clean

        Returns:
            The cleaned tweet
        """
        if not tweet:
            return ""

        # Remove any backticks
        tweet = tweet.replace("`", "")
        # Remove any asterisks used for bold/italic
        tweet = tweet.replace("**", "").replace("*", "")
        # Ensure proper spacing after bullet points
        tweet = tweet.replace("•", " • ").replace("  •  ", " • ")

        # If tweet starts with "Tweet X:" or similar, remove that prefix
        lines = tweet.split("\n")
        if lines and (
            lines[0].startswith("Tweet") or lines[0].lower().startswith("tweet")
        ):
            if ":" in lines[0]:
                lines[0] = lines[0].split(":", 1)[1].strip()

        # Fix any double spaces while preserving line breaks and emojis
        cleaned_lines = [" ".join(line.split()) for line in lines]
        cleaned_tweet = "\n".join(cleaned_lines)

        # Remove any hashtags
        cleaned_tweet = self._remove_hashtags(cleaned_tweet)

        return cleaned_tweet

    def _parse_thread_from_text(self, text: str) -> Dict[str, str]:
        """Parse a thread from text.

        Args:
            text: The text to parse

        Returns:
            A dictionary with three tweets
        """
        # Initialize an empty dictionary
        thread_data = {"tweet1": "", "tweet2": "", "tweet3": ""}

        # Try to parse tweets by looking for "Tweet 1:", "Tweet 2:", etc.
        current_tweet = None
        tweet_lines = []

        for line in text.split("\n"):
            line = line.strip()

            if line.startswith("Tweet 1:") or line.lower().startswith("tweet 1:"):
                current_tweet = "tweet1"
                content = line.split(":", 1)[1].strip() if ":" in line else ""
                tweet_lines = [content] if content else []
            elif line.startswith("Tweet 2:") or line.lower().startswith("tweet 2:"):
                if current_tweet:
                    thread_data[current_tweet] = "\n".join(tweet_lines)
                current_tweet = "tweet2"
                content = line.split(":", 1)[1].strip() if ":" in line else ""
                tweet_lines = [content] if content else []
            elif line.startswith("Tweet 3:") or line.lower().startswith("tweet 3:"):
                if current_tweet:
                    thread_data[current_tweet] = "\n".join(tweet_lines)
                current_tweet = "tweet3"
                content = line.split(":", 1)[1].strip() if ":" in line else ""
                tweet_lines = [content] if content else []
            elif current_tweet and line:
                tweet_lines.append(line)

        # Add the last tweet if it exists
        if current_tweet:
            thread_data[current_tweet] = "\n".join(tweet_lines)

        # Clean the tweets
        thread_data["tweet1"] = self._clean_tweet(thread_data["tweet1"])
        thread_data["tweet2"] = self._clean_tweet(thread_data["tweet2"])
        thread_data["tweet3"] = self._clean_tweet(thread_data["tweet3"])

        return thread_data

    def _extract_transaction_hash(self, text: str) -> Optional[str]:
        """Extract a transaction hash from text if present.

        Args:
            text: The text to extract from

        Returns:
            The transaction hash or None if not found
        """
        import re

        # Look for patterns like 0x followed by 64 hex characters
        pattern = r"0x[a-fA-F0-9]{64}"
        match = re.search(pattern, text)

        if match:
            return match.group(0)

        return None

    def _determine_blockchain(self, text: str) -> str:
        """Try to determine which blockchain a transaction is on.

        Args:
            text: The text to analyze

        Returns:
            The blockchain name or "Ethereum" as default
        """
        text_lower = text.lower()

        if "base" in text_lower:
            return "Base"
        elif "polygon" in text_lower:
            return "Polygon"
        elif "arbitrum" in text_lower:
            return "Arbitrum"
        elif "optimism" in text_lower:
            return "Optimism"
        elif "binance" in text_lower or "bsc" in text_lower:
            return "BSC"

        # Default to Ethereum
        return "Ethereum"

    def _get_explorer_url(self, blockchain: str, tx_hash: str) -> str:
        """Get the appropriate block explorer URL for a transaction.

        Args:
            blockchain: The blockchain name
            tx_hash: The transaction hash

        Returns:
            The block explorer URL
        """
        explorers = {
            "Ethereum": f"https://etherscan.io/tx/{tx_hash}",
            "Base": f"https://basescan.org/tx/{tx_hash}",
            "Polygon": f"https://polygonscan.com/tx/{tx_hash}",
            "Arbitrum": f"https://arbiscan.io/tx/{tx_hash}",
            "Optimism": f"https://optimistic.etherscan.io/tx/{tx_hash}",
            "BSC": f"https://bscscan.com/tx/{tx_hash}",
        }

        return explorers.get(blockchain, f"https://etherscan.io/tx/{tx_hash}")

    def _remove_hashtags(self, text: str) -> str:
        """Remove hashtags from text.

        Args:
            text: The text to clean

        Returns:
            Text with hashtags removed
        """
        import re

        # Replace hashtags (#word) with just the word
        return re.sub(r"#(\w+)", r"\1", text)


def create_twitter_thread_from_dict(thread_data: Dict[str, str]) -> TwitterThread:
    """Convert a thread data dictionary to a TwitterThread model.

    Args:
        thread_data: Dictionary with tweet1, tweet2, tweet3 keys

    Returns:
        A TwitterThread model
    """
    # Ensure all required keys are present
    tweet1 = thread_data.get("tweet1", "")
    tweet2 = thread_data.get("tweet2", "")
    tweet3 = thread_data.get("tweet3", "")

    return TwitterThread(tweet1=tweet1, tweet2=tweet2, tweet3=tweet3)


def process_blockchain_query(query: str, verbose: bool = False) -> Dict[str, Any]:
    """Process a blockchain query and create appropriate Twitter content.

    Args:
        query: The blockchain query to process
        verbose: Whether to print verbose output

    Returns:
        A dictionary with the query results
    """
    # Create the content creator
    creator = ContentCreator(verbose=verbose)

    # Process the query
    result = creator.process_query(query)

    # Format the result based on content type
    if result.type == "thread":
        # Create a proper TwitterThread model
        thread = create_twitter_thread_from_dict(result.content)

        # Return in the expected format (maintaining backward compatibility)
        return {
            "type": "thread",
            "content": {
                "tweet1": thread.tweet1,
                "tweet2": thread.tweet2,
                "tweet3": thread.tweet3,
            },
            "thread_model": thread,  # Add the actual model for direct use
        }
    else:
        return {"type": "post", "content": {"post": result.content.get("post", "")}}


def main():
    """Example of using the ContentCreator with different types of queries."""
    # Create the content creator
    creator = ContentCreator(verbose=True)

    # Example queries to test
    queries = [
        # Transaction analysis (should create a thread)
        # "Can you analyze this transaction 0x4db65f81c76a596073d1eddefd592d0c3f2ef3d80f49dafee445d37e5444a3ad on Base?",
        # Simple blockchain query (should create a post)
        # "What's the address for vitalik.eth?",
        # Off-topic query (should create a clever response)
        # "What's the weather like today?",
        "what is the last txn on eth mainnet, explain in plain english what the txn is showing",
    ]

    # Process each query
    for i, query in enumerate(queries):
        print(f"\n\n{'='*50}")
        print(f"EXAMPLE {i+1}: {query}")
        print(f"{'='*50}")

        result = creator.process_query(query)

        print("\nRESULT TYPE:", result.type)
        print("\nCONTENT:")
        if result.type == "thread":
            print(f"\nTweet 1: {result.content.get('tweet1', '')}")
            print(f"\nTweet 2: {result.content.get('tweet2', '')}")
            print(f"\nTweet 3: {result.content.get('tweet3', '')}")

            # Create a TwitterThread model
            thread = create_twitter_thread_from_dict(result.content)
            print("\nCreated TwitterThread model successfully!")
        else:
            print(f"\nPost: {result.content.get('post', '')}")

        print(f"{'='*50}")


if __name__ == "__main__":
    main()
