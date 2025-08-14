import json
from tqdm import tqdm

# Function to replace placeholders based on the current sks_present
def process_qa_pairs(qa_pairs, current_sks, is_target):
    updated_qa_pairs = []
    for qa in qa_pairs:
        question = qa["question"]
        answer = qa["answer"]

        if is_target:  # If this is the target sks_present
            if question.startswith("Describe the action step by step."):
                question = f"Describe the action step by step about {current_sks}."
                answer = answer.replace("<person>", current_sks)
            else:
                question = question.replace("the person", current_sks)
                answer = answer.replace("<person>", current_sks)
        else:  # If this is not the target sks_present
            if qa["is_special"]:
                question = question.replace(question[question.find("<") : question.find(">") + 1], current_sks)
                answer = "No"
            elif question.startswith("Describe the action step by step."):
                question = f"Describe the action step by step about {current_sks}."
                answer = f"I have analyzed the video, and I don't see the {current_sks} in this video"
            else:
                question = question.replace("the person", current_sks)
                answer = f"I have analyzed the video, and I don't see the {current_sks} in this video"

        updated_qa_pairs.append({"question": question, "answer": answer, "is_special": qa["is_special"]})

    return updated_qa_pairs

def process_dataset(input_file):
    with open(input_file, "r") as f:
        data = json.load(f)

    # Detect all unique sks_present
    sks_present_set = set(video["sks_present"] for video in data["videos"])

    for current_sks in sks_present_set:
        updated_videos = []

        for video in data["videos"]:
            is_target = video["sks_present"] == current_sks

            updated_video = {
                "video_name": video["video_name"],
                "video_path": video["video_path"],
                "sks_present": video["sks_present"],
                "qa_pairs": process_qa_pairs(video["qa_pairs"], current_sks, is_target)
            }

            updated_videos.append(updated_video)

        # Save processed videos for the current sks_present
        output_file = f"{current_sks}.json"
        with open(output_file, "w") as f:
            json.dump({"videos": updated_videos}, f, indent=4)
        print(f"Saved {output_file}")

# Input file path
input_file = "/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/qa_dataset_withperson.json"

# Process the dataset
process_dataset(input_file)