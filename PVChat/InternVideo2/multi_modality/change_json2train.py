import json
import random

# Load original dataset
with open('<yufei>test.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# Load negative responses
with open('/root/autodl-tmp/yufei/ConsisID/negative_sample_3question.json', 'r', encoding='utf-8') as f:
    negative_responses = json.load(f)

# Load confirmation responses
with open('/root/autodl-tmp/yufei/ConsisID/confirm_question.json', 'r', encoding='utf-8') as f:
    confirm_responses = json.load(f)

def is_action_question(question,this_sks):
    return any(q.replace("<sks>", this_sks) == question for q in negative_responses["questions"]["action_questions"])

def is_clothing_question(question,this_sks):
    return any(q.replace("<sks>", this_sks) == question for q in negative_responses["questions"]["clothing_questions"])

def is_location_question(question,this_sks):
    return any(q.replace("<sks>", this_sks) == question for q in negative_responses["questions"]["location_questions"])

# Get sks_name of positive samples
yufei_video = next(video for video in dataset["videos"] if video["video_name"].startswith("yufei"))
target_sks = yufei_video["sks_present"]

# Process dataset
for video in dataset["videos"]:
    # Skip positive samples and test set
    if not video["video_name"].startswith("cong"):
        continue

    # Modify negative samples
    print(f"Processing negative sample: {video['video_name']}")

    # Change sks_present to target person
    #video["sks_present"] = target_sks
    this_sks="<"+video["sks_present"]+">"
    this_sks="<yufei>"
    # Process each QA pair
    for qa in video["qa_pairs"]:
        if qa["is_special"]:
            # Handle special question responses
            new_answer = random.choice(confirm_responses["no_answers"])
            qa["answer"] = new_answer.replace("<sks>", f"<{target_sks}>")
            qa["question"] = qa["question"].replace(video["sks_present"], target_sks)
        else:
            # Handle other question responses
            question = qa["question"]
            if is_action_question(question,this_sks):
                new_answer = random.choice(negative_responses["negative_responses"]["action_responses"])
                qa["answer"] = new_answer.replace("<sks>", f"<{target_sks}>")
            elif is_clothing_question(question,this_sks):
                new_answer = random.choice(negative_responses["negative_responses"]["clothing_responses"])
                qa["answer"] = new_answer.replace("<sks>", f"<{target_sks}>")
            elif is_location_question(question,this_sks):
                new_answer = random.choice(negative_responses["negative_responses"]["location_responses"])
                qa["answer"] = new_answer.replace("<sks>", f"<{target_sks}>")
            qa["question"] = qa["question"].replace(video["sks_present"], target_sks)
            # Keep other questions unchanged

# Save modified dataset
output_file = '<yufei>test2.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"Processing completed, results saved in {output_file}")