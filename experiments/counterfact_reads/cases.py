"""Select cases using corpus structure and tokenization before looking at model outcomes."""
import hashlib
import json
from pathlib import Path

SOURCE = "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"

def select(path, tokenizer):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SOURCE
    records = json.loads(path.read_text())
    def order(record):
        return hashlib.sha256(f"counterfact-native-probe-20261118/{record['case_id']}".encode()).hexdigest()
    selected = []
    subjects, relations = set(), {}
    for record in sorted(records, key=order):
        fact = record["requested_rewrite"]
        relation, subject = fact["relation_id"], fact["subject"]
        words = [fact[k]["str"] for k in ("target_true", "target_new")]
        if subject in subjects or relations.get(relation, 0) >= 2 or words[0] == words[1]:
            continue
        if not all(word.isalpha() and len(tokenizer.encode(word, add_special_tokens=False)) == 1 for word in words):
            continue
        if not record["paraphrase_prompts"] or not record["neighborhood_prompts"]:
            continue
        selected.append(record)
        subjects.add(subject)
        relations[relation] = relations.get(relation, 0) + 1
        if len(selected) == 24:
            return selected
    raise ValueError("Insufficient eligible cases")

def manifest(records):
    return [{"id":r["case_id"], "sha256":hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()} for r in records]

if __name__ == "__main__":
    import sys
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(sys.argv[2], local_files_only=True)
    print(json.dumps(manifest(select(Path(sys.argv[1]),tok)),indent=2))
