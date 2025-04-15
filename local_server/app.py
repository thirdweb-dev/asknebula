from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import sys
import os
import json

# Add the parent directory to the Python path to import thread_creator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from thread_creator import process_blockchain_query


# Custom JSON Encoder to handle objects that aren't JSON serializable
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            # Check if object has __dict__ attribute (like most Python objects)
            if hasattr(obj, "__dict__"):
                return obj.__dict__
            # Check if object is iterable (like lists)
            iterable = iter(obj)
            return list(iterable)
        except TypeError:
            # As a last resort, convert to string
            return str(obj)


app = Flask(__name__)
app.json_encoder = CustomJSONEncoder  # Set custom JSON encoder for Flask

# Configure CORS to allow requests from the frontend
CORS(
    app,
    origins="*",
    allow_headers=["Content-Type", "X-Simplified-Response"],
    methods=["GET", "POST", "OPTIONS"],
)


@app.route("/api/create-thread", methods=["POST", "OPTIONS"])
def create_thread():
    if request.method == "OPTIONS":
        return Response(status=200)

    try:
        data = request.get_json()
        query = data.get("query")

        if not query:
            return jsonify({"error": "Query is required"}), 400

        # Process the query and get the thread
        result = process_blockchain_query(query)

        # Check if we need to manually serialize TwitterThread
        if "thread_model" in result and result["thread_model"] is not None:
            try:
                # Extract data from thread_model into a separate field
                thread = result["thread_model"]
                result["thread_data"] = {
                    "tweet1": thread.tweet1,
                    "tweet2": thread.tweet2,
                    "tweet3": thread.tweet3,
                }
                # Remove the non-serializable thread_model
                del result["thread_model"]
            except Exception as thread_err:
                print(f"Error processing thread model: {thread_err}")

        # Try to use jsonify
        try:
            return jsonify(result)
        except TypeError as json_err:
            print(f"JSON serialization error: {json_err}")
            # If jsonify fails, use manual serialization
            safe_result = {"type": result.get("type", "unknown"), "content": {}}

            # Extract content fields safely
            content = result.get("content", {})
            if isinstance(content, dict):
                for key, value in content.items():
                    safe_result["content"][key] = str(value)

            # Use standard json module with custom encoder
            return Response(
                response=json.dumps(safe_result, cls=CustomJSONEncoder),
                status=200,
                mimetype="application/json",
            )

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
