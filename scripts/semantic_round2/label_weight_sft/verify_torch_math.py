"""Run on the server before any real training; tiny tensors, no model/data.

This has not been executed locally because the audit venv has no Torch.
The default device is CPU; --device cuda also checks the actual CUDA operator.
"""
import argparse,json,tempfile
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from label_weights import causal_weighted_loss,WeightedLossMixin


def check_actual_trainer_accumulation():
    """Actual HF Trainer/Accelerate SGD updates versus explicit grouped CE."""
    from transformers import Trainer,TrainingArguments
    torch.manual_seed(28)
    initial=torch.randn(3,6,7)
    examples=[
        {'input_ids':[0]*6,'labels':[-100,-100,3,4,5,6],'loss_weights':[0.,0.,1.,4.,1.,1.]},
        {'input_ids':[1]*4,'labels':[-100,2,3,-100],'loss_weights':[0.,1.,4.,0.]},
        {'input_ids':[2]*5,'labels':[-100,3,4,5,6],'loss_weights':[0.,1.,1.,4.,1.]},
    ]

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.table=torch.nn.Parameter(initial.clone())
        def forward(self,input_ids,**kwargs):
            return SimpleNamespace(logits=self.table[input_ids[:,0],:input_ids.shape[1],:])

    def collate(items):
        n=max(len(item['input_ids']) for item in items)
        return {key:torch.tensor([item[key]+[pad]*(n-len(item[key])) for item in items],dtype=dtype)
                for key,pad,dtype in [('input_ids',0,torch.long),('labels',-100,torch.long),('loss_weights',0.,torch.float)]}

    denominators=[]
    class TestedTrainer(WeightedLossMixin,Trainer):
        def _get_train_sampler(self,train_dataset=None):
            return torch.utils.data.SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)
        def training_step(self,model,inputs,num_items_in_batch=None):
            denominators.append(float(num_items_in_batch))
            return super().training_step(model,inputs,num_items_in_batch)

    expected=Toy();optimizer=torch.optim.SGD(expected.parameters(),lr=.03)
    for group in ([0,1],[2]):
        optimizer.zero_grad()
        denominator=sum(sum(examples[i]['loss_weights'][1:]) for i in group)
        loss=0.
        for i in group:
            batch=collate([examples[i]])
            logits=expected(batch['input_ids']).logits
            per_token=F.cross_entropy(logits[:,:-1,:].reshape(-1,7),batch['labels'][:,1:].reshape(-1),ignore_index=-100,reduction='none')
            loss=loss+(per_token*batch['loss_weights'][:,1:].reshape(-1)).sum()/denominator
        loss.backward();optimizer.step()
    actual=Toy()
    with tempfile.TemporaryDirectory(prefix='weighted-sft-toy-') as tmp:
        args=TrainingArguments(output_dir=tmp,use_cpu=True,per_device_train_batch_size=1,
            gradient_accumulation_steps=2,num_train_epochs=1,max_steps=-1,learning_rate=.03,
            max_grad_norm=0.,weight_decay=0.,warmup_ratio=0.,save_strategy='no',report_to=[],
            logging_strategy='no',remove_unused_columns=False,dataloader_pin_memory=False,
            disable_tqdm=True,seed=42)
        opt=torch.optim.SGD(actual.parameters(),lr=.03)
        trainer=TestedTrainer(model=actual,args=args,train_dataset=examples,data_collator=collate,
                             optimizers=(opt,torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)))
        trainer.model_accepts_loss_kwargs=True
        trainer.train()
        assert trainer.state.global_step==2
    torch.testing.assert_close(actual.table,expected.table,rtol=1e-6,atol=1e-7)
    assert denominators==[12.,12.,7.],denominators
    assert not torch.equal(actual.table,initial)
    return {'actual_HF_Trainer_CPU':True,'microbatch_lengths':[6,4,5],
            'gradient_accumulation':2,'optimizer_updates':2,'weighted_denominators':denominators,
            'last_partial_group_verified':True,'SGD_tensors_match_explicit_grouped_CE':True,
            'max_abs_parameter_delta_vs_reference':float((actual.table-expected.table).abs().max())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');args=p.parse_args()
    torch.manual_seed(42)
    logits=torch.randn(2,6,7,device=args.device,requires_grad=True)
    labels=torch.tensor([[-100,-100,3,4,5,6],[-100,2,3,4,6,-100]],device=args.device)
    unit=(labels!=-100).float()
    actual=causal_weighted_loss(logits,labels,unit)
    expected=F.cross_entropy(logits[:,:-1,:].reshape(-1,7),labels[:,1:].reshape(-1),ignore_index=-100)
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
    weighted=unit.clone();weighted[0,3]=4.;weighted[1,2]=4.
    combined=causal_weighted_loss(logits,labels,weighted)
    denominator=weighted[:,1:].sum()
    accumulated=sum(causal_weighted_loss(logits[i:i+1],labels[i:i+1],weighted[i:i+1],denominator) for i in range(2))
    torch.testing.assert_close(combined,accumulated,rtol=1e-6,atol=1e-7)
    a=torch.autograd.grad(combined,logits,retain_graph=True)[0]
    b=torch.autograd.grad(accumulated,logits)[0]
    torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
    assert torch.all(a[0,0]==0) and torch.all(a[1,4]==0) and torch.all(a[:,-1]==0)
    trainer_check=check_actual_trainer_accumulation()
    print(json.dumps({'passed':True,'device':args.device,'torch':torch.__version__,
                     'unit_weight_matches_CE':True,'unequal_microbatch_accumulation_value_and_gradient_match':True,
                     'prompt_padding_last_logit_zero_gradient':True,'trainer_accumulation_regression':trainer_check}))


if __name__=='__main__':main()
