import json
import torch
from torch.utils.data import TensorDataset, RandomSampler, DistributedSampler, DataLoader
from processer.classes import InputFeatures
from hyperparameter import args
from transformers import BertTokenizer


def read_json(json_dir=args.train_file_path, do_lower_case=True):

    with open(json_dir, encoding='utf-8') as f:

        text_lists = []
        tag_lists = []
        for line in f:
            line = json.loads(line)
            
            token_list = []
            for token in line['text']:
                if token == '”' or token == '“' or token == '‘' or token == '’':
                    token = '"'
                if do_lower_case:
                    token_list.append(token.lower())
                else:
                    token_list.append(token)
                    print(token)

            text_lists.append(token_list)

            tags = ['O' for _ in range(len(line['text']))]
            labels = line['label']
            for tag, value in labels.items():

                for entity, indexs in value.items():
                    for index in indexs:
                        tags[index[0]] = 'B-' + tag
                        for i in range(index[0]+1, index[1]+1):
                            tags[i] = 'I-' + tag

            tag_lists.append(tags)

    return text_lists, tag_lists


def build_corpus(tag_lists):

    tag2id = {tag: i for i, tag in enumerate(args.id2tag)}
    tag_ids = []

    for tags in tag_lists:
        tag_id = []
        for tag in tags:
            if tag not in tag2id.keys():
                tag2id[tag] = len(tag2id)
            tag_id.append(tag2id[tag])
        tag_ids.append(tag_id)

    return tag_ids


def convert_examples_to_features(token_list, tag_list, id2tag, max_seq_length, tokenizer,
                                 cls_token_at_end=False, cls_token="[CLS]", cls_token_segment_id=1,
                                 sep_token="[SEP]", pad_on_left=False, pad_token=0, pad_token_segment_id=0,
                                 sequence_a_segment_id=0, mask_padding_with_zero=True,):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)

        >>  token_list = [['关', '于', '存', '量', '客', '户', '的', '房', '贷', '利', '率', '是', '否', '调', '整', '，', '交', '行', '正', '在', '研', '究'],
                          ['约', '维', '蒂', '奇', '有', '望', '与', '吉', '拉', '蒂', '诺', '搭', '档', '锋', '线', '。', '2', '0']]
            tag_list = [[1, 2, 2, 2, 4, 4, 4, 4, 4, 5, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                        [4, 4, 4, 4, 8, 9, 9, 9, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]]

           convert_examples_to_features(token_list, tag_list, args.tag2id, 128, tokenizer)
    """
    tag2id = {tag: i for i, tag in enumerate(id2tag)}
    features = []
    for tokens, tags in zip(token_list, tag_list):

        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            tags = tags[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        tags += [tag2id['[SEP]']]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            tags += [tag2id['[CLS]']]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            tags = [tag2id['[CLS]']] + tags
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(tags)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            tags = ([pad_token] * padding_length) + tags
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            tags += [pad_token] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(tags) == max_seq_length

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, input_len=input_len,
                                      segment_ids=segment_ids, tag_ids=tags))
    return features


def build_dataset(tokenizer, mode=True):

    if mode:
        # on train set
        text_lists, tag_lists = read_json(args.train_file_path, do_lower_case=True)
        tag_ids = build_corpus(tag_lists)
        features = convert_examples_to_features(text_lists, tag_ids, args.id2tag, args.train_max_seq_length, tokenizer)
    else:
        # on dev set
        text_lists, tag_lists = read_json(args.dev_file_path, do_lower_case=True)
        tag_ids = build_corpus(tag_lists)
        features = convert_examples_to_features(text_lists, tag_ids, args.id2tag, args.eval_max_seq_length, tokenizer)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.tag_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)

    return dataset

def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_labels = all_labels[:,:max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels,all_lens


if __name__ == '__main__':
    # text_lists, tag_lists = read_json()
    # tag_ids = build_corpus(tag_lists)
    # print(text_lists[100:110])
    # print(tag_ids[:10])

    tokenizer = BertTokenizer.from_pretrained('../bert-base-chinese')
    # token_list = [['关', '于', '存', '量', '客', '户', '的', '房', '贷', '利', '率', '是', '否', '调', '整', '，', '交', '行', '正', '在', '研', '究'],
    #               ['约', '维', '蒂', '奇', '有', '望', '与', '吉', '拉', '蒂', '诺', '搭', '档', '锋', '线', '。', '2', '0']]
    # tag_list = [[1, 2, 2, 2, 4, 4, 4, 4, 4, 5, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    #             [4, 4, 4, 4, 8, 9, 9, 9, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]]

    train_dataset = build_dataset(tokenizer, True)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    batch = next(iter(train_dataloader))
    print(batch)