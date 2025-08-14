# utils.py

import torch

def build_input_ids(tokenizer, conversation, max_length, add_special_tokens,
                   truncation, video_placeholder="[<VID_PLH>]", padding=False,
                   return_tensors="pt"):
    input_ids = []
    indexs = []
    attention_mask = []
    start, total_len = 0, 0

    while True:
        # 寻找视频占位符
        index = conversation.find(video_placeholder, start)

        if index == -1:  # 处理最后的文本部分
            inputs = tokenizer(
                conversation[start:],
                max_length=max_length - total_len,
                truncation=truncation,
                padding=padding,
                return_tensors=return_tensors
            )
            # 添加文本部分
            input_ids.append(inputs.input_ids[0])
            attention_mask.append(inputs.attention_mask[0])
            indexs.append(torch.zeros_like(inputs.input_ids[0]))
            break

        else:  # 处理占位符之前的文本
            inputs = tokenizer(
                conversation[start:index],
                max_length=max_length,
                truncation=truncation,
                padding=padding,
                return_tensors=return_tensors
            )

            # 添加文本部分
            input_ids.append(inputs.input_ids[0])
            attention_mask.append(inputs.attention_mask[0])
            # 假设视频 token 的长度为 96+16，这里需要根据实际情况调整
            input_ids.append(torch.zeros(96 + 16, dtype=torch.long))
            attention_mask.append(torch.ones(96 + 16, dtype=torch.long))
            indexs.append(torch.ones(96 + 16, dtype=torch.bool))

            start = index + len(video_placeholder)

    # 拼接所有部分
    input_ids = torch.cat(input_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    indexs = torch.cat(indexs, dim=0).to(torch.bool)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'index': indexs
    }
