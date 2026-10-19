"""CPU-only coverage/identity/finite-schedule checks; no model execution."""
import sys,json
from pathlib import Path
from copy import deepcopy
import pytest
HERE=Path(__file__).resolve().parent;W=HERE.parents[2]
sys.path[:0]=[str(HERE),str(W/'MedicalNER-Qwen3/agent')]
import semantic_dpo as dpo
import run_smoke_then_train as wrapper


def data(longest_side='rejected'):
    kinds=[('outside_entity_table',None),
        ('reviewed_semantic_contradiction','explicit_negation_label_flip'),
        ('reviewed_semantic_contradiction','subtype_direction_reversal'),
        ('reviewed_semantic_contradiction','explicit_positive_label_flip'),
        ('outside_entity_table',None),
        ('reviewed_semantic_contradiction','explicit_positive_label_flip')]
    pairs=[{'id':str(i),'mutation':{'kind':k,**({'semantic_change_kind':s} if s else {})}} for i,(k,s) in enumerate(kinds)]
    encoded=[{'id':str(i),'prompt_ids':[1]*(i+1),'chosen_ids':[1,2],'rejected_ids':[1,2]} for i in range(len(pairs))]
    encoded[4][longest_side+'_ids']=[1]*100
    return encoded,pairs


@pytest.mark.parametrize('side',['chosen','rejected','prompt'])
def test_all_three_semantic_subclasses_outside_and_global_max_fit_four(side):
    e,p=data(side);indices=dpo.smoke_indices(e,p)
    assert len(indices)==len(set(indices))==4 and 4 in indices
    assert {p[i]['mutation'].get('semantic_change_kind',p[i]['mutation']['kind']) for i in indices}=={
        'outside_entity_table','explicit_negation_label_flip','subtype_direction_reversal','explicit_positive_label_flip'}
    length=lambda i:len(e[i]['prompt_ids'])+max(len(e[i]['chosen_ids']),len(e[i]['rejected_ids']))
    assert max(map(length,indices))==max(map(length,range(len(e))))
    assert indices==dpo.smoke_indices(e,p)


def test_only_present_categories_and_deterministic_longest_fill():
    e,p=data();e=e[:3]+e[4:5];p=p[:3]+p[4:5]
    assert len(dpo.smoke_indices(e,p))==4


def test_semantic_only_release_covers_three_subclasses_without_inserting_mechanical_pair():
    e,p=data();keep=[i for i,x in enumerate(p) if x['mutation']['kind']=='reviewed_semantic_contradiction']
    e,p=[e[i] for i in keep],[p[i] for i in keep]
    e[-1]['chosen_ids']=[1]*100
    chosen=dpo.smoke_indices(e,p)
    assert len(chosen)==4 and len(e)-1 in chosen
    assert {p[i]['mutation']['semantic_change_kind'] for i in chosen}=={
        'explicit_positive_label_flip','explicit_negation_label_flip','subtype_direction_reversal'}
    assert all(p[i]['mutation']['kind']=='reviewed_semantic_contradiction' for i in chosen)


@pytest.mark.parametrize('defect',['too_few','duplicate','unaligned','unknown_semantic'])
def test_invalid_smoke_input_rejected(defect):
    e,p=data()
    if defect=='too_few':e,p=e[:3],p[:3]
    if defect=='duplicate':e[4]['id']=e[0]['id']
    if defect=='unaligned':p[0]['id']='different'
    if defect=='unknown_semantic':p[1]['mutation']['semantic_change_kind']='unsupported'
    with pytest.raises(ValueError):dpo.smoke_indices(e,p)


def original_config():
    return json.loads((HERE/'dpo24_config.json').read_text())


@pytest.mark.parametrize('count',[24])
def test_any_multiple_of_four_has_one_visit_and_expected_updates(count):
    c=original_config();c['expected_pairs']=count
    _,_,n=wrapper.validate_config(c);assert n==count
    schedule=dpo.epoch_schedule(count,1,42,4)
    assert len(schedule)==count and sorted(schedule)==list(range(count))


@pytest.mark.parametrize('count',[0,1,3,4,8,52,76,128,256,True,24.0])
def test_invalid_pair_count_rejected(count):
    c=original_config();c['expected_pairs']=count
    with pytest.raises(ValueError):wrapper.validate_config(c)


def test_fresh_parent_epoch_and_output_separation():
    for flag,value in [('--epochs','2'),('--gradient-accumulation','8'),('--parent-adapter','smoke_adapter')]:
        c=original_config();c['common_argv'][c['common_argv'].index(flag)+1]=value
        with pytest.raises(ValueError):wrapper.validate_config(c)
    c=original_config();c['formal_output']=c['smoke_output']
    with pytest.raises(ValueError):wrapper.validate_config(c)


def test_only_smoke_whitelist_changed_and_builder_identical():
    old=W/'audit_work/round2_20261019/dpo52_v2'
    expected=(old/'semantic_dpo.py').read_text().replace(
        '("explicit_negation_label_flip", "subtype_direction_reversal")',
        '("explicit_negation_label_flip", "subtype_direction_reversal", "explicit_positive_label_flip")')
    current=(HERE/'semantic_dpo.py').read_text()
    # The only other approved difference is descriptive VERSION/LIMITATIONS.
    def without_metadata(text):
        a=text.index('VERSION = ');b=text.index('LOCKS = ',a)
        return text[:a]+text[b:]
    assert without_metadata(current)==without_metadata(expected)
    assert (HERE/'build_pairs.py').read_bytes()==(old/'build_pairs.py').read_bytes()


def test_root_plan_budget_latest_start_and_cannot_extend():
    end=wrapper.PLAN_DEADLINE
    deadline,stop=wrapper.bounded_training_deadline(end-3600,end+3600)
    assert deadline==end and stop==end-3000
    deadline,stop=wrapper.bounded_training_deadline(wrapper.LATEST_START)
    assert stop==end-1800
    with pytest.raises(TimeoutError):wrapper.bounded_training_deadline(wrapper.LATEST_START+0.01)
    with pytest.raises(TimeoutError):wrapper.bounded_training_deadline(end-2400,end-1)


def test_frozen_data_and_optimizer_reject_drift():
    c=original_config();c['expected_train_sha256']='0'*64
    with pytest.raises(ValueError):wrapper.validate_config(c)
    for flag,value in [('--learning-rate','1e-5'),('--beta','0.2'),('--max-length','8192'),('--max-seconds','1800')]:
        c=original_config();c['common_argv'][c['common_argv'].index(flag)+1]=value
        with pytest.raises(ValueError):wrapper.validate_config(c)


def test_previous_label4_full130_and_metrics_identity_gate(tmp_path):
    import hashlib
    metrics=tmp_path/'metrics.json';metrics.write_text('{}')
    receipt={'completed':True,'failed':False,'samples':130,'metrics_sha256':hashlib.sha256(metrics.read_bytes()).hexdigest(),
      'loaded_adapter_verification':{'relations':{k:wrapper.LABEL4_PARAMETERS for k in ('expected_sha256','loaded_sha256','cuda_sha256')}}}
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(receipt))
    assert len(wrapper.check_previous_result(manifest))==64
    metrics.write_text('{"changed":true}')
    with pytest.raises(ValueError):wrapper.check_previous_result(manifest)
    for key,value in [('completed',False),('failed',True),('samples',129)]:
        wrong=deepcopy(receipt);wrong[key]=value;manifest.write_text(json.dumps(wrong))
        with pytest.raises(ValueError):wrapper.check_previous_result(manifest)


def test_exact_original_round3_algorithms_preserved():
    old=W/'audit_work/round3_20261019/trainer'
    for name in ('semantic_dpo.py','build_pairs.py'):
        assert (HERE/name).read_bytes()==(old/name).read_bytes()
