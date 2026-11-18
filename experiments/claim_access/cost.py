"""Report observed component work and charge the fixed evidence pipeline to the ordinary readers."""
import json
from pathlib import Path
import statistics
HERE=Path(__file__).resolve().parent
def rows(n):return list(map(json.loads,(HERE/(n+'.jsonl')).read_text().splitlines()))
def main():
    forms=json.loads((HERE/'formulations.json').read_text())
    retrieval={v['id']:v for v in rows('retrieval')};mask={v['id']:v for v in rows('supports')};generated=rows('answers')
    assert set(forms)==set(retrieval)==set(mask)=={v['id'] for v in generated}
    methods=sorted(generated[0]['measurements']);out={}
    for m in methods:
        qwen_pos=sum(v['measurements'][m]['positions'] for v in generated)
        answer_seconds=sum(v['measurements'][m]['seconds'] for v in generated)
        metrics=dict(generation_positions=qwen_pos,generation_seconds=answer_seconds,median_generated_request_seconds=statistics.median(v['measurements'][m]['seconds'] for v in generated))
        if m in ('span','span_quarters','wide','all_scored'):
            metrics['sum_observed_component_seconds']=sum(v['seconds'] for v in forms.values())+sum(v['seconds'] for v in retrieval.values())+answer_seconds
            metrics['sum_observed_4b_positions']=sum(v['positions'] for v in forms.values())+qwen_pos
        out[m]=metrics
    records=dict(scope='Development. Same-prompt generations are reused across arms; per-arm generation durations reflect actually evaluated distinct prompts. Ordinary readers shown with component sums execute formulation, lexical and dense retrieval, and the extractive shortlist once per request. Retrieval seconds include index construction and cross-encoder work. Model loading, benchmark adjudication and filesystem transfers are excluded. Mask-arm timings include shared control paths and are not separately attributable. No end-to-end savings claim.',n=len(generated),methods=out,
        measured_component_seconds=dict(formulations=sum(v['seconds'] for v in forms.values()),retrieval=sum(v['seconds'] for v in retrieval.values()),all_masking_controls=sum(v['seconds'] for v in mask.values()),unique_free_generation=sum(v['seconds'] for v in generated)),
        actual_4b_positions=dict(formulations=sum(v['positions'] for v in forms.values()),all_masking_controls=sum(v['positions'] for v in mask.values())))
    (HERE/'cost.json').write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__':main()
