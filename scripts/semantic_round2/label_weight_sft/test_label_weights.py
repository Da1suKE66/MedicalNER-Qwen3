"""CPU tests of token spans, masks, full EOS retention and objective invariants."""
import sys
from pathlib import Path
import pytest
from transformers import AutoTokenizer

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from label_weights import attach_weights,weighted_completion,pad_weights,weighted_mean,relation_field_spans


@pytest.fixture(scope='module')
def tokenizer():
    return AutoTokenizer.from_pretrained(HERE.parents[2]/'medicalner-migration-20261012-v1/tokenizer/qwen3-8b',local_files_only=True)


TEXT='Symptom\tcauses copy\tis_core_symptom_of\tDiagnosis\tTest condition\tEvidence says causes and is_core_symptom_of.\n'


def test_only_third_field_weighted_even_with_same_words_elsewhere(tokenizer):
    ids,w,c=weighted_completion(TEXT,tokenizer,4)
    assert len(ids)==len(w)==len(c)
    assert c.count('relation_label')>0
    assert all(weight==(4 if cat=='relation_label' else 1) for weight,cat in zip(w,c))
    assert c[-1]=='eos' and ids[-1]==tokenizer.eos_token_id and w[-1]==1
    assert tokenizer.eos_token_id not in ids[:-1]


def test_input_ids_full_target_and_prompt_mask_preserved(tokenizer):
    ids,w,_=weighted_completion(TEXT,tokenizer,4)
    original={'input_ids':[1,2,3]+ids,'labels':[-100]*3+ids,'unchanged':'sentinel'}
    result=attach_weights(original,TEXT,tokenizer,4)
    assert result['input_ids']==original['input_ids'] and result['labels']==original['labels']
    assert result['loss_weights']==[0,0,0]+w and result['unchanged']=='sentinel'
    assert 'loss_weights' not in original


def test_padding_zero_and_supervised_tokens_never_zero(tokenizer):
    ids,_,_=weighted_completion(TEXT,tokenizer,4)
    a=attach_weights({'input_ids':[1]+ids,'labels':[-100]+ids},TEXT,tokenizer,4)
    b=attach_weights({'input_ids':[1,2]+ids,'labels':[-100,-100]+ids},TEXT,tokenizer,4)
    padded=pad_weights([a,b],len(b['labels']))
    assert padded[0][-1]==0 and padded[1][-1]==1
    assert padded[0][0]==0 and padded[1][:2]==[0,0]
    assert sum(padded[0])==sum(a['loss_weights'])


def test_changed_target_or_mask_rejected(tokenizer):
    ids,_,_=weighted_completion(TEXT,tokenizer,4)
    with pytest.raises(ValueError,match='Completion encoding'):
        attach_weights({'input_ids':[1]+ids[:-1],'labels':[-100]+ids[:-1]},TEXT,tokenizer,4)
    with pytest.raises(ValueError,match='masking'):
        attach_weights({'input_ids':[1]+ids,'labels':[1]+ids},TEXT,tokenizer,4)


def test_weight_one_is_exact_unweighted_ce():
    losses=[.3,1.7,.2,3.1,.04]
    assert weighted_mean(losses,[1]*5)==sum(losses)/5


def test_accumulation_must_use_global_denominator_not_mean_of_means():
    losses=[.1,4.,.3,1.2,2.1]; weights=[1.,4.,1.,1.,4.]
    total=weighted_mean(losses,weights)
    accumulated=sum(sum(l*w for l,w in zip(ls,ws))/sum(weights)
                    for ls,ws in [(losses[:1],weights[:1]),(losses[1:],weights[1:])])
    assert accumulated==pytest.approx(total)
    assert total!=pytest.approx((weighted_mean(losses[:1],weights[:1])+weighted_mean(losses[1:],weights[1:]))/2)


def test_multiple_records_unicode_and_literal_escaped_tab(tokenizer):
    text=TEXT+'Symptom\tcafé中文\\tphrase\tis_associated_symptom_of\tDiagnosis\tExample\tQuoted “evidence”.\n'
    ids,w,c=weighted_completion(text,tokenizer,4)
    assert len(relation_field_spans(text))==12
    assert ids[:-1]==tokenizer.encode(text,add_special_tokens=False)
    assert all(weight==1 for weight,cat in zip(w,c) if cat!='relation_label')


@pytest.mark.parametrize('text',[TEXT.rstrip('\n'),'one\ttwo\n',TEXT.replace('Test condition','')])
def test_ambiguous_or_incomplete_records_rejected(text):
    with pytest.raises(ValueError): relation_field_spans(text)
