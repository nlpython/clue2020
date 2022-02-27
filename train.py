import torch
import os
from models.model import BertCrfForNer
from transformers import BertTokenizer, get_linear_schedule_with_warmup, AdamW, BertConfig
from torch.utils.data import DataLoader
from hyperparameter import args
from processer.utils import build_dataset, collate_fn
from processer.common import seed_everything
from processer.progressbar import ProgressBar
from processer.classes import SeqEntityScore
from torch.utils.data import RandomSampler, DistributedSampler, SequentialSampler
from loguru import logger


def train(epochs):

    logger.add(args.log_output_dir + '/{time}.log')
    logger.info('======== model config ========')
    for key, value in vars(args).items():
        logger.info(f'{key}: {value}')

    num_labels = len(args.id2tag)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    config = BertConfig.from_pretrained(args.bert_path, num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    model = BertCrfForNer.from_pretrained(args.bert_path, config=config).to(device)

    # load train data
    train_dataset = build_dataset(tokenizer, args.do_train)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    # load dev data
    dev_dataset = build_dataset(tokenizer, args.do_eval)
    dev_sampler = SequentialSampler(dev_dataset) if args.local_rank == -1 else DistributedSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.eval_batch_size,
                                collate_fn=collate_fn)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.bert_learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.bert_learning_rate},

        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.crf_learning_rate},

        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.other_learning_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.other_learning_rate}
    ]
    t_total = len(train_dataloader) * epochs
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.other_learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    logger.info('=========begin trainning=========')
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    global_step = 0
    best_f1 = 0.0
    for epoch in range(epochs):
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3]}
            # tokens = tokenizer.convert_ids_to_tokens(batch[0][0][1:-1])
            # print(tokens)
            outputs = model(**inputs)       # ? type id
            loss = outputs[0]  # models outputs are always tuple in pytorch-transformers (see doc)
            loss.backward()
            pbar(epoch, step, {'loss': loss.item()})
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    print(" ")
                    _, f1 = evaluate(dev_dataloader, model, tokenizer, device)
                    if f1 > best_f1:
                        best_f1 = f1
                        # Save models checkpoint
                        output_dir = os.path.join(args.checkpoint_dir, "checkpoint-{}".format(f1*100))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving models checkpoint to %s", output_dir)
                        tokenizer.save_vocabulary(output_dir)
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)
        logger.info("\n")
        # if 'cuda' in str(args.device):
        #     torch.cuda.empty_cache()
    return global_step, tr_loss / global_step

def evaluate(dev_dataloader, model, tokenizer, device):
    metric = SeqEntityScore(args.id2tag, markup=args.markup)
    # eval_output_dir = args.output_dir
    # if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
    #     os.makedirs(eval_output_dir)
    # Eval!
    logger.info('====stop trainning, eval=====')
    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(dev_dataloader), desc="Evaluating")
    # if isinstance(models, nn.DataParallel):
    #     models = models.module
    for step, batch in enumerate(dev_dataloader):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            tags = model.crf.decode(logits, inputs['attention_mask'])
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        out_label_ids = inputs['labels'].cpu().numpy().tolist()
        input_lens = batch[4].cpu().numpy().tolist()
        tags = tags.squeeze(0).cpu().numpy().tolist()
        for i, label in enumerate(out_label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif j == input_lens[i] - 1:
                    metric.update(pred_paths=temp_2, label_paths=temp_1)
                    break
                else:
                    temp_1.append(args.id2tag[out_label_ids[i][j]])
                    temp_2.append(args.id2tag[tags[i][j]])
        pbar(None, step)
    logger.info("\n")
    eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    logger.info("***** Eval results *****")
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    logger.info("***** Entity results *****")
    for key in sorted(entity_info.keys()):
        logger.info("******* %s results ********" % key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        logger.info(info)
    return results, eval_info['f1']


if __name__ == '__main__':
    train(args.num_train_epochs)