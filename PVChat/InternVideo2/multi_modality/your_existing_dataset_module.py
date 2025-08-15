# your_existing_dataset_module.py

class PersonalizedVideoDataset(Dataset):
    def __init__(self, json_path: str, tokenizer, split='train'):
        # load json file
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.videos = data['video']  # Make sure the top-level key is 'video'
        self.sks_name = Path(json_path).stem
        self.split = split
        self.tokenizer = tokenizer

        self.all_qa_pairs = self.flatten_qa_pairs(self.videos)

    def __len__(self):
        return len(self.all_qa_pairs)

    def flatten_qa_pairs(self, videos):
        all_qa = []
        for video_idx, video in enumerate(videos):
            for qa_pair in video['qa_pairs']:
                all_qa.append((video_idx, qa_pair))
        return all_qa

    def __getitem__(self, idx):
        video_idx, qa_pair = self.all_qa_pairs[idx]
        video_data = self.videos[video_idx]

        # load video
        video_tensor = load_video(
            video_data['video_path'],
            num_segments=8,
            return_msg=False,
            resolution=224,
            hd_num=6
        )

        question = qa_pair['question']
        answer = qa_pair['answer']

        if self.split == 'train':
            conversation = "[INST] "
            if video_tensor.shape[1] == 1:
                ilen = video_tensor.shape[0]
                conversation += ("<Image>" + IMG_TOKEN + "</Image>") * ilen
            else:
                ilen = video_tensor.shape[1]
                conversation += ("<Video>" + VID_TOKEN + "</Video>") * ilen
            conversation += "[/INST] "
            sks_and_tokens = f"[{self.sks_name}]" + 'is' + ''.join([f"<token{i}>" for i in range(16)]) + " "
            conversation += "[INST]" + sks_and_tokens + question + "[/INST]"
            conversation += answer + "</s>"

            tokenized = build_input_ids(
                self.tokenizer,
                conversation,
                max_length=512,
                add_special_tokens=True,
                truncation=True,
                padding='longest',
                return_tensors='pt',
                video_placeholder="[<VID_PLH>]"
            )

            labels = tokenized['input_ids'].clone()
            inst_tokens = self.tokenizer.encode("[/INST]", add_special_tokens=False)
            inst_end_indices = torch.where(labels == inst_tokens[-1])[0]
            second_inst_end = inst_end_indices[1] if len(inst_end_indices) > 1 else inst_end_indices[0]
            labels[:second_inst_end + 1] = -100
            labels[tokenized['index']] = -100
        else:
            conversation = "[INST] "
            if video_tensor.shape[1] == 1:
                ilen = video_tensor.shape[0]
                conversation += ("<Image>" + IMG_TOKEN + "</Image>") * ilen
            else:
                ilen = video_tensor.shape[1]
                conversation += ("<Video>" + VID_TOKEN + "</Video>") * ilen
            conversation += "[/INST] "
            sks_and_tokens = f"[{self.sks_name}]" + 'is' + ''.join([f"<token{i}>" for i in range(16)]) + " "
            conversation += "[INST]" + sks_and_tokens + question + "[/INST]"

            tokenized = build_input_ids(
                self.tokenizer,
                conversation,
                max_length=512,
                add_special_tokens=True,
                truncation=True,
                padding='longest',
                return_tensors='pt',
                video_placeholder="[<VID_PLH>]"
            )
            labels = None

        return_dict = {
            'video': video_tensor,
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'video_idx': tokenized['index'],
            'question': question
        }

        if labels is not None:
            return_dict['labels'] = labels
            return_dict['is_special'] = qa_pair.get('is_special', False)
            return_dict['sks_present'] = video_data['sks_present'] == f'<{self.sks_name}>'

        return return_dict
