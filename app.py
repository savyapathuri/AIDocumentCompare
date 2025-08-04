import os
import traceback
import json
from flask import Flask, render_template, request
from openai import AzureOpenAI
from dotenv import load_dotenv

# --- Step 1: Load Environment Variables ---
load_dotenv()

# --- Step 2: Initialize the Flask App ---
app = Flask(__name__)

# --- Step 3: Configure the Azure OpenAI Client ---
try:
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2024-05-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialize AzureOpenAI client.")
    print(f"Please ensure your .env file is correct. Error: {e}")
    client = None

def get_file_content(file):
    """Reads the content of a file and returns it as a string."""
    try:
        return file.read().decode('utf-8', errors='ignore')
    except Exception:
        # Fallback for other potential errors
        return "Could not read file content."

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    error = None

    if request.method == 'POST':
        if not client:
            error = "The Azure OpenAI client is not configured. Please check the server logs for details."
            return render_template('index.html', results=results, error=error)

        file1 = request.files.get('file1')
        file2 = request.files.get('file2')

        if not file1 or not file2:
            error = "Please upload both files to compare."
            return render_template('index.html', results=results, error=error)

        try:
            content1 = get_file_content(file1)
            content2 = get_file_content(file2)

            prompt = (
                "You are an AI assistant that compares two documents and highlights their differences. "
                "Your task is to return the full text of both documents, keeping the original text and structure intact as much as possible. "
                "In the returned documents, highlight the specific word-level differences. "
                "Wrap deleted text in Document 1 with `<del>` tags. "
                "Wrap added text in Document 2 with `<ins>` tags. "
                "For lines that are completely new or deleted, wrap the entire line. "
                "For lines that have only minor changes, only wrap the changed words.\n\n"
                "Also, provide a summary of the changes. "
                "The entire output MUST be a single, valid JSON object with three keys: 'doc1_highlighted' (string), "
                "'doc2_highlighted' (string), and 'summary' (JSON array of strings). "
                "Each string in the 'summary' array should describe a single difference, including the line number where it occurred. "
                "For example: [\"Difference at line 20: 'brown' in Document 1 was replaced with 'red' in Document 2.\"]. "
                "Do not include any conversational text or formatting outside of the main JSON object.\n\n"
                "--- Document 1 ---\n"
                f"{content1}\n"
                "--- Document 2 ---\n"
                f"{content2}\n"
                "--- End of Documents ---"
            )

            response = client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                messages=[
                    {"role": "system", "content": "You are a helpful AI document comparison assistant that returns JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            response_content = response.choices[0].message.content
            # Clean up the response in case the AI wraps it in markdown
            if response_content.strip().startswith("```json"):
                response_content = response_content.strip()[7:-3].strip()

            try:
                results = json.loads(response_content)
            except json.JSONDecodeError:
                error = "The AI returned an invalid format. Please try again."
                print("--- INVALID JSON RESPONSE ---")
                print(response_content)
                print("-----------------------------")


        except Exception as e:
            error_traceback = traceback.format_exc()
            print("--- AN ERROR OCCURRED DURING API CALL ---")
            print(error_traceback)
            print("-----------------------------------------")
            error = f"An unexpected error occurred. Please check the terminal for the full traceback. Error: {e}"

    return render_template('index.html', results=results, error=error)

if __name__ == '__main__':
    app.run(debug=True)
