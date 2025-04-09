import json
import os

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from thirdweb_ai import Nebula
from thirdweb_ai.adapters.langchain import get_langchain_tools
from langchain.tools import tool
from thirdweb_ai.services.service import Service


class NebulaAgent:
    """A class that encapsulates the Nebula agent functionality for blockchain operations."""

    def __init__(self, secret_key=None, model="gpt-4o-mini", verbose=True):
        """Initialize the NebulaAgent.

        Args:
            secret_key: The secret key for Nebula. If None, it will be read from the THIRDWEB_SECRET_KEY environment variable.
            model: The model to use for the LLM.
            verbose: Whether to print verbose output.
        """
        self.secret_key = secret_key or os.getenv("THIRDWEB_SECRET_KEY")
        self.verbose = verbose

        # Initialize Nebula
        self.nebula = Nebula(secret_key=self.secret_key)

        # Initialize LLM
        self.llm = ChatOpenAI(model=model)

        # Create a prompt template with better system instructions
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are an AI assistant specialized in blockchain operations. For balance queries on specific contracts, use the get_balance_on_contract tool directly instead of the generic chat tool.
                    - Always use the most specific tool available for the task.
                    - If a chain is not specified, you will assume the user is asking for the balance of an ENS name or an address on the mainnet.
                    - You must not make up any information, like contract addresses, chain ids, etc. You will only work with the information provided by the user.
                    - By default you will avoid using the get_balance_on_contract tool and use the generic chat tool instead unless the user explicitly asks for the balance of a specific contract. If no contract address is provided, you will assume the user is requesting the balance of an ENS name so you will default to the native currency of the chain.
                    - Get transaction details is only for transaction hashes, not for ENS names or addresses. if the tools in incorrectly used you need to pivot and find the transactions for the wallet address and ignore this tool call.
                    - If the user asks for information about an ENS name, you will use the resolve_ens_name tool to get the address of the ENS name and then try to get the balances of the address for mainnet unless another chain is specified, and also try to get any other information you can find about the address.
                    - If the request is about finding suspicious activity of a wallet address, you will get the last 10 transactions of the address and analyze them looking for patterns that are suspicious, like wash trading, or other patterns.
                    - When a chain is provided but not it's chain id, you need to automatically resolve the chain id from the chain name.
                    - For off-topic questions, personal messages, greetings, or any requests unrelated to blockchain, use the respond_to_offtopic tool, but if the user asks about blockchain or crypto, you must always first try to get the information from the blockchain and then pivot to the respond_to_offtopic tool if you can't find the information in the blockchain.
                    - respond_to_offtopic tool must always be used as the last resort, only use it if you can't find the information in the blockchain.
                    - For general blockchain queries that don't refer to a specific address (like "latest transactions" or "last tx on eth mainnet"), use the chat tool with the exact query and don't ask for an address unless absolutely necessary.
                    - When users ask about the "last tx" or "latest transaction" on a blockchain without specifying an address, provide information about the most recent transaction on that blockchain without requesting an address.
                    - When using the chat tool for chain-specific queries, always include the chain context in your message by extracting it from the user's query.
                    - NEVER use the chat tool with 'context: None' when the query mentions a specific blockchain - always extract and include the chain in your message.

                    - the user not always use the correct language, so you need to be able to understand the user's intent and respond to the user in a way that is most likely to get the information the user is looking for, for example they might not use the word "transaction" but they might use "txn" or "tx".

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
        # Get thirdweb tools for LangChain
        self.thirdweb_tools = get_langchain_tools(self.nebula.get_tools())

        # Define custom tools
        @tool
        def get_balance_on_contract(
            ens_name: str,
            contract_address: str,
            chain: str = None,
            chain_id: int = None,
        ) -> str:
            """Get the balance of an ENS name on a contract, use this only when the contract address is explicitly provided.

            If no chain_id is provided it must utomatically try to resolve the chain_id from the chain name
            For Sepolia the chain_id is 11155111
            For Mainnet the chain_id is 1
            For Optimism the chain_id is 10
            For Base the chain_id is 8453
            For Arbitrum the chain_id is 42161
            For Polygon the chain_id is 137
            And so on for other chains

            Args:
                ens_name: The ENS name to get the balance of
                contract_address: The address of the contract to get the balance of
                chain: The chain to get the balance of
                chain_id: The chain id to get the balance of

            Returns:
                The balance of the ENS name on the contract
            """
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "example_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "ens_name": {
                                "type": "string",
                                "description": "The ENS name being queried",
                            },
                            "balance": {
                                "type": "integer",
                                "description": "The balance of the address on the specified contract",
                            },
                        },
                        "required": ["ens_name", "balance"],
                    },
                },
            }

            self.nebula.response_format = response_format
            if self.verbose:
                print(f"calling nebula with response format: {response_format}")

            # Get the response from Nebula
            return self.nebula._post(
                "/chat",
                {
                    "message": f"What is the balance of {ens_name} on contract {contract_address} chain_id: {chain_id} chain: {chain}"
                },
            )

        @tool
        def resolve_ens_name(ens_name: str) -> str:
            """Resolve an ENS name to an address

            Args:
                ens_name: The ENS name to resolve

            Returns:
                The address of the ENS name

            """

            structured_response = {
                "type": "json_schema",
                "json_schema": {
                    "name": "example_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "ens_name": {"type": "string"},
                            "address": {"type": "string"},
                        },
                        "required": ["ens_name", "address"],
                    },
                },
            }

            self.nebula.response_format = structured_response
            if self.verbose:
                print(f"calling nebula with response format: {structured_response}")

            # Use nebula's _post method instead of calling _post directly
            return self.nebula._post(
                "/chat", {"message": f"What is the address of {ens_name} chain_id: 1"}
            )

        @tool
        def get_transaction_details(
            transaction_hash: str, chain: str = None, chain_id: int = None
        ) -> str:
            """Get the details of a transaction from any EVM chain, if the chain_id is not provided it must utomatically try to resolve the chain_id from the chain name.



            Args:
                transaction_hash: The hash of the transaction to get the details of
                chain: The chain to get the transaction details of
                chain_id: The chain id to get the transaction details of

            Returns:
                The details of the transaction
            """

            structured_response = {
                "type": "json_schema",
                "json_schema": {
                    "name": "transaction_details_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "transaction_overview": {
                                "type": "object",
                                "properties": {
                                    "block": {"type": "integer"},
                                    "timestamp": {
                                        "type": "string",
                                        "format": "date-time",
                                    },
                                    "status": {
                                        "type": "string",
                                        "enum": ["success", "failed", "pending"],
                                    },
                                    "sender": {"type": "string"},
                                    "recipient": {"type": "string"},
                                    "contract": {"type": "string"},
                                    "value": {
                                        "type": "string",
                                        "description": "Transaction value in native currency",
                                    },
                                    "gas_used": {"type": "integer"},
                                    "gas_limit": {"type": "integer"},
                                    "gas_price": {"type": "string"},
                                    "transaction_fee": {"type": "string"},
                                    "nonce": {"type": "integer"},
                                    "position_in_block": {"type": "integer"},
                                },
                                "required": ["status", "sender", "block", "timestamp"],
                            },
                            "operation_details": {
                                "type": "object",
                                "properties": {
                                    "function_name": {"type": "string"},
                                    "function_signature": {"type": "string"},
                                    "input_parameters": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "type": {"type": "string"},
                                                "value": {"type": "string"},
                                            },
                                        },
                                    },
                                    "decoded_input": {"type": "string"},
                                    "operation_type": {
                                        "type": "string",
                                        "enum": [
                                            "transfer",
                                            "swap",
                                            "mint",
                                            "burn",
                                            "approve",
                                            "other",
                                        ],
                                    },
                                    "tokens_involved": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "token_address": {"type": "string"},
                                                "token_name": {"type": "string"},
                                                "token_symbol": {"type": "string"},
                                                "token_amount": {"type": "string"},
                                                "token_type": {
                                                    "type": "string",
                                                    "enum": [
                                                        "ERC20",
                                                        "ERC721",
                                                        "ERC1155",
                                                        "Other",
                                                    ],
                                                },
                                            },
                                        },
                                    },
                                },
                                "description": "Details about the specific operation performed in the transaction",
                                "required": ["function_name", "operation_type"],
                            },
                            "events_emitted": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "description": {"type": "string"},
                                        "contract_address": {"type": "string"},
                                        "parameters": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "name": {"type": "string"},
                                                    "type": {"type": "string"},
                                                    "value": {"type": "string"},
                                                    "indexed": {"type": "boolean"},
                                                },
                                            },
                                        },
                                    },
                                    "required": ["name", "contract_address"],
                                },
                            },
                            "blockchain_context": {
                                "type": "object",
                                "properties": {
                                    "chain_id": {"type": "integer"},
                                    "chain_name": {"type": "string"},
                                    "block_explorer_url": {"type": "string"},
                                    "network_congestion": {
                                        "type": "string",
                                        "enum": ["low", "medium", "high"],
                                    },
                                    "current_gas_price": {"type": "string"},
                                },
                                "required": ["chain_id", "chain_name"],
                            },
                            "security_analysis": {
                                "type": "object",
                                "properties": {
                                    "risk_level": {
                                        "type": "string",
                                        "enum": ["low", "medium", "high", "critical"],
                                    },
                                    "concerns": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "unusual_patterns": {"type": "boolean"},
                                    "known_exploits": {"type": "boolean"},
                                },
                            },
                            "observations": {
                                "type": "string",
                                "description": "Additional insights or observations about the transaction",
                            },
                            "impact_summary": {
                                "type": "object",
                                "properties": {
                                    "financial_impact": {"type": "string"},
                                    "protocol_impact": {"type": "string"},
                                    "user_impact": {"type": "string"},
                                },
                            },
                        },
                        "required": [
                            "transaction_overview",
                            "operation_details",
                            "events_emitted",
                            "blockchain_context",
                            "observations",
                            "security_analysis",
                            "impact_summary",
                        ],
                    },
                },
            }

            self.nebula.response_format = structured_response
            if self.verbose:
                print(f"calling nebula with response format: {structured_response}")

            # Use nebula's _post method instead of calling _post directly
            return self.nebula._post(
                "/chat",
                {
                    "message": f"What is the details of transaction {transaction_hash} chain_id: {chain_id} chain: {chain}"
                },
            )

        @tool
        def respond_to_offtopic(query: str) -> str:
            """Handle off-topic questions, personal messages, greetings, jokes, or any requests unrelated to blockchain data.
            Use this tool when the user's query doesn't require blockchain data or other specific tools.

            Args:
                query: The off-topic question or message from the user

            Returns:
                A friendly, appropriate response to the off-topic query
            """
            # Create a system prompt that ensures appropriate responses
            system_prompt = """You are AskNebula, a friendly AI assistant specialized in blockchain.
            When responding to off-topic questions or personal messages:
            - Be friendly, helpful and concise
            - Use humor when appropriate
            - Never provide harmful, disrespectful, offensive or inappropriate content
            - If the message mentions blockchain or crypto in any way, acknowledge it but explain you need specific details to provide blockchain data
            - For birthday wishes or greetings to others, respond in a friendly, celebratory way
            - For completely unrelated topics, gently remind the user about your blockchain expertise while still being helpful
            - Never give financial advice, only provide information but if asked don't be rude, just play along and say you are not a financial advisor, reply with the same tone and language as the user (as long as the language is not offensive or disrespectful).
            """

            # Use the existing LLM instance
            response = self.llm.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ]
            )

            return response.content

        # Store the custom tools
        self.custom_tools = [
            get_balance_on_contract,
            resolve_ens_name,
            get_transaction_details,
            respond_to_offtopic,
        ]

        # Combine thirdweb tools with our custom tools
        self.tools = self.thirdweb_tools + self.custom_tools

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

    def get_available_tools(self):
        """Get a list of available tools.

        Returns:
            A list of dictionaries with tool names and descriptions.
        """
        return [
            {"name": tool.name, "description": tool.description} for tool in self.tools
        ]

    def run(self, query, return_intermediate_steps=False):
        """Run the agent with the given query.

        Args:
            query: The query to run the agent with.
            return_intermediate_steps: Whether to return the intermediate steps.

        Returns:
            The response from the agent.
        """
        if self.verbose:
            print(f"\n\nQuery: {query}")
            print("-" * 50)

        response = self.agent_executor.invoke({"input": query})

        if self.verbose:
            print("\nFinal Response:")
            print(response["output"])

            # Try to extract structured response if possible
            print("\nIntermediate Steps:")
            for step in response["intermediate_steps"]:
                # If the output is a string, try to parse it as JSON
                if isinstance(step[1], str):
                    try:
                        parsed_json = json.loads(step[1])
                        print("\nStructured Response (JSON):")
                        print(parsed_json)
                    except json.JSONDecodeError:
                        # Not JSON, just continue
                        pass

        if return_intermediate_steps:
            return response
        else:
            return response["output"]


def main():
    """Example of using the NebulaAgent."""
    # Create the agent
    agent = NebulaAgent(verbose=True)

    # Display available tools
    print("Available tools:")
    for tool in agent.get_available_tools():
        print(f"- {tool['name']}: {tool['description']}")
    print("\n")

    # Example query
    query = "what is the last txn on eth mainnet, explain in plain english what the txn is showing"
    print(f"Running query: {query}")

    # Run the agent
    response = agent.run(query)

    # Print the response
    print("\nFinal Response:")
    print(response)


if __name__ == "__main__":
    main()
