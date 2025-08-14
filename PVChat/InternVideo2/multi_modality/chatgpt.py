from openai import OpenAI
import os
import json
import os
import base64
import openai
import os
from tqdm import tqdm
api_key = os.getenv("OPENAI_API_KEY")
# Initialize OpenAI client
client = OpenAI(
    api_key=api_key,
    base_url="https://api.openai-sb.com/v1"
)

# Configure OpenAI API
openai.api_key = api_key
openai.api_base = "https://api.openai-sb.com/v1"


def get_caption_from_api(answer):
    original = "Help me replace all the descriptions of people in the following paragraph, such as human, he, she, etc., with <person>.Just return the modified one:"
    prompt = original + answer

    for attempt in range(3):  # Retry up to 3 times
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                stream=True,
            )

            reply = ""
            for res in response:
                content = res.choices[0].delta.content
                if content:
                    reply += content

            return reply
        except Exception as e:
            print(f"Error processing answer on attempt {attempt + 1}: {e}")
    return answer  # Fallback to the original answer if API fails


def process_qa_dataset(input_file, output_file):
    with open(input_file, "r") as f:
        data = json.load(f)

    for video in tqdm(data["videos"], desc="Processing videos"):
        for qa in video["qa_pairs"]:
            if not qa["is_special"]:  # Skip if is_special is True
                qa["answer"] = get_caption_from_api(qa["answer"])

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)


# Input and output file paths
input_file = "qa_dataset.json"
output_file = "qa_dataset_withperson.json"

# Process the dataset
process_qa_dataset(input_file, output_file)