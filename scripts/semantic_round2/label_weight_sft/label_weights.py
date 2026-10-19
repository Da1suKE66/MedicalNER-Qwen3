"""Exact completion-token weights for literal semantic relation-label spans.

Original prompts, completion token IDs, target records and EOS are unchanged.
A token receives label weight only when its non-separator span intersects the
third TAB field. Cross-field semantic tokens are rejected, never guessed.
"""
from __future__ import annotations
from collections import Counter

FIELD_NAMES=('source_type','source_name','relation_label','target_type','target_name','evidence')


def relation_field_spans(completion):
    spans=[];pos=0
    for line in completion.splitlines(keepends=True):
        if not line.endswith('\n'):
            raise ValueError('Completion must preserve final record newline')
        fields=line[:-1].split('\t')
        if len(fields)!=6 or any(not field for field in fields):
            raise ValueError('Expected all six nonempty semantic fields')
        start=pos
        for index,field in enumerate(fields):
            spans.append((start,start+len(field),FIELD_NAMES[index]))
            start+=len(field)+1
        pos+=len(line)
    if not spans or pos!=len(completion):
        raise ValueError('Invalid completion boundaries')
    return spans


def completion_token_categories(completion, tokenizer):
    spans=relation_field_spans(completion)
    encoded=tokenizer(completion,add_special_tokens=False,return_offsets_mapping=True)
    ids=list(encoded['input_ids']);offsets=encoded['offset_mapping']
    if ids!=list(tokenizer.encode(completion,add_special_tokens=False)) or len(ids)!=len(offsets):
        raise ValueError('Offset tokenizer changed token IDs')
    categories=[]
    for a,b in offsets:
        if not (0<=a<b<=len(completion)):
            raise ValueError('Invalid completion token offset')
        overlap={name for left,right,name in spans if max(a,left)<min(b,right)}
        if len(overlap)>1:
            raise ValueError('Token crosses multiple semantic fields')
        categories.append(next(iter(overlap)) if overlap else 'separators')
    return ids,categories


def weighted_completion(completion,tokenizer,label_weight):
    if not 1<=label_weight<=8:
        raise ValueError('Label weight must be moderate, between1 and8')
    ids,categories=completion_token_categories(completion,tokenizer)
    if tokenizer.eos_token_id in ids:
        raise ValueError('Embedded EOS in completion')
    return ids+[tokenizer.eos_token_id], [float(label_weight) if c=='relation_label' else 1. for c in categories]+[1.],categories+['eos']


def weighted_mean(losses,weights):
    """Pure scalar reference for CPU objective tests."""
    if len(losses)!=len(weights) or not losses or any(w<0 for w in weights) or sum(weights)<=0:
        raise ValueError('Invalid weighted mean')
    return sum(l*w for l,w in zip(losses,weights))/sum(weights)


def attach_weights(encoded, completion, tokenizer, label_weight):
    """Attach weights without changing the original full example or its mask."""
    ids,weights,categories=weighted_completion(completion,tokenizer,label_weight)
    prompt_length=len(encoded['input_ids'])-len(ids)
    if prompt_length<1 or encoded['input_ids'][prompt_length:]!=ids:
        raise ValueError('Completion encoding differs from original SFT')
    if encoded['labels']!=[-100]*prompt_length+ids:
        raise ValueError('Original prompt/EOS masking changed')
    result=dict(encoded)
    result['loss_weights']=[0.]*prompt_length+weights
    result['_label_tokens']=categories.count('relation_label')
    return result


def pad_weights(items, max_length):
    result=[]
    for item in items:
        w=item['loss_weights'];labels=item['labels']
        if len(w)!=len(labels) or len(w)>max_length:
            raise ValueError('Weight alignment/length failure')
        if any((weight==0)!=(label==-100) for weight,label in zip(w,labels)):
            raise ValueError('Prompt/padding mask must match weights exactly')
        result.append(w+[0.]*(max_length-len(w)))
    return result


def causal_weighted_loss(logits, labels, weights, denominator=None):
    """Exact token-weighted causal CE; denominator may span accumulation group."""
    import torch
    import torch.nn.functional as F
    if logits.shape[:2]!=labels.shape or labels.shape!=weights.shape:
        raise ValueError('Logit/label/weight shape mismatch')
    shift_labels=labels[:,1:].contiguous()
    shift_weights=weights[:,1:].contiguous()
    if torch.any(shift_weights[shift_labels==-100]!=0):
        raise ValueError('Prompt or padding carries loss weight')
    losses=F.cross_entropy(logits[:,:-1,:].float().reshape(-1,logits.shape[-1]),
        shift_labels.reshape(-1),ignore_index=-100,reduction='none').view_as(shift_labels)
    mass=shift_weights.sum() if denominator is None else denominator
    if float(mass)<=0:
        raise ValueError('No weighted target tokens')
    return (losses*shift_weights).sum()/mass


class WeightedLossMixin:
    """Single-process HF4.56.2 loss normalized over the entire accumulation group.

    Reusing ordinary per-row mean losses would silently change the relative
    weighting of long/short examples. Weight1 must retain ordinary token CE.
    """
    def get_batch_samples(self, epoch_iterator, num_batches, device):
        batches,_=super().get_batch_samples(epoch_iterator,num_batches,device)
        if self.accelerator.num_processes!=1 or self.args.n_gpu>1:
            raise ValueError('Candidate supports exactly one process/device')
        denominator=sum(batch['loss_weights'][:,1:].sum() for batch in batches)
        return batches,denominator.to(device) if batches else None

    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        labels=inputs.pop('labels')
        weights=inputs.pop('loss_weights')
        inputs.pop('_row_indices',None)
        outputs=model(**inputs)
        loss=causal_weighted_loss(outputs.logits,labels,weights,num_items_in_batch)
        return (loss,outputs) if return_outputs else loss
