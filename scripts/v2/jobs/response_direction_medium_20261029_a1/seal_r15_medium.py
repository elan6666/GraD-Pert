"""Server metadata only; no checkpoint/NPZ/H5AD numerical reads or GPU."""
import argparse, copy, hashlib, importlib.util, json, math, os, subprocess
from pathlib import Path
COUNTS = {'conditions': 1335, 'rows': 5317, 'deep_conditions': 128, 'deep_rows': 506, 'gradient_conditions': 32, 'gradient_rows': 125}
POPULATION_SHA = 'afa9c95fa26dc19167e9919bc09c86a8d99f9446466de95a1efd613b1f93bd86'

def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()

def identity_diff(before, after, prefix=""):
    changes = []
    for key in sorted(before.keys() | after.keys()):
        path = prefix + "/" + key
        a, b = key in before, key in after
        if a and b and isinstance(before[key], dict) and isinstance(after[key], dict):
            changes.extend(identity_diff(before[key], after[key], path))
        elif a != b or before.get(key) != after.get(key):
            changes.append(
                dict(
                    path=path,
                    before=dict(present=a, **({"value": before[key]} if a else {})),
                    after=dict(present=b, **({"value": after[key]} if b else {})),
                )
            )
    return changes

def validate_population(parent, panel, views, medium):
    counts = dict(
        conditions=len(parent["schedule"]),
        rows=sum(len(g["rows"]) for g in parent["schedule"]),
        deep_conditions=sum(bool(g["deep_rows"]) for g in parent["schedule"]),
        deep_rows=sum(len(g["deep_rows"]) for g in parent["schedule"]),
        gradient_conditions=sum(bool(g["gradient_rows"]) for g in parent["schedule"]),
        gradient_rows=sum(len(g["gradient_rows"]) for g in parent["schedule"]),
    )
    fields = ("index", "condition_id", "rows", "deep_rows", "gradient_rows", "queries", "view_seed")
    population_sha = digest([{k: g[k] for k in fields} for g in parent["schedule"]])
    if (
        counts != COUNTS
        or any(parent.get(k) != v for k, v in COUNTS.items())
        or parent.get("analysis_scope") != "medium_cap40_stratified"
        or population_sha != POPULATION_SHA
        or medium["population_sha256"] != POPULATION_SHA
        or medium["counts"] != COUNTS
    ):
        raise ValueError("frozen medium population/counts changed")
    if (
        panel.get("conditions"),
        panel.get("rows"),
        panel.get("controls_per_context"),
        panel.get("condition_context_groups"),
        len(views),
    ) != (32, 128, 4, 32, 32) or panel["panel_sha256"] != digest(panel["schedule"]):
        raise ValueError("fixed crossed32x4 panel changed")
    contexts = {}
    train = {g["condition_id"] for g in parent["schedule"]}
    for group, view in zip(panel["schedule"], views, strict=True):
        context = tuple(group["context"])
        if (
            len(context) != 4
            or group["condition_id"] not in train
            or len(group["rows"]) != 4
            or group["queries"] != parent["schedule"][0]["queries"]
            or view["condition_id"] != group["condition_id"]
            or tuple(view["context"]) != context
        ):
            raise ValueError("frozen condition/context/view/query order changed")
        contexts.setdefault(context, []).append(group)
    if len(contexts) != 16:
        raise ValueError("exact16 crossed contexts required")
    for groups in contexts.values():
        if len(groups) != 2 or len({g["condition_id"] for g in groups}) != 2:
            raise ValueError("two distinct conditions per context required")
        controls = [[r["control_row_id"] for r in g["rows"]] for g in groups]
        if controls[0] != controls[1] or len(set(controls[0])) != 4:
            raise ValueError("crossed shared four ordered controls changed")

def validate_performance(performance):
    parity = performance.get("full_parity", {})
    fast, reference = (
        performance.get(k, float("nan")) for k in ("full_fast_seconds", "full_reference_seconds")
    )
    if (
        performance.get("status") != "completed"
        or performance.get("same_sample_scope") is not True
        or performance.get("deterministic_algorithms") is not True
        or performance.get("model_updates") != 0
        or parity.get("passed") is not True
        or parity.get("atol") != 1e-8
        or not 0 <= parity.get("max_abs", float("inf")) <= 1e-8
        or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (fast, reference))
        or not 0 < fast < reference
    ):
        raise ValueError("frozen full-medium performance/correctness receipt differs")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def build(registry, owner=None):
    spec = copy.deepcopy(registry['spec'])
    values = {}
    for path, expected in registry['metadata'].items():
        p = Path(path)
        if not p.resolve().is_relative_to('/data/yilangliu') or p.suffix not in ('.json', '.yaml') or p.stat().st_size > 32*1024*1024 or sha(p) != expected:
            raise ValueError('exact bounded metadata changed: ' + path)
        if p.suffix == '.json':
            values[path] = json.loads(p.read_text())
    for path, expected in registry['source_artifacts'].items():
        if sha(path) != expected:
            raise ValueError('published helper changed')
    old = values[registry['old_postfit']['path']]
    prior = old['medium']
    if prior['parent'] != spec['medium']['parent'] or prior['performance'] != spec['medium']['performance'] or prior['panel'] != dict(spec['panel'], panel_sha256='b0b4d6e09ca40ab40cb51d881cfaaeb00d23d71d832bf45b0d1f82de45a1913b', conditions=32, rows=128) or prior['original_queue'] != registry['queue'] or old['shared_policy'] != spec['resource_policy']:
        raise ValueError('historical population/policy pointers changed')
    parent = values[spec['medium']['parent']['path']]
    validate_population(parent, values[spec['panel']['population']['path']], values[spec['panel']['matched_views']['path']], prior)
    validate_performance(values[spec['medium']['performance']['path']])
    identity = values[spec['run_manifest']['path']]
    journal = values[spec['journal']['path']]
    if journal['identity'] != identity or journal['last'] != dict(epoch=2, file='epoch-0002.pt', sha256=spec['checkpoint']['sha256']) or digest(values[spec['resolved_config']['path']]) != registry['resolved_semantic_sha256'] or identity['resolved_config_sha256'] != registry['resolved_semantic_sha256'] or identity_diff(parent['data'], identity['data']) != spec['medium']['identity_changes']:
        raise ValueError('run/resolved/CP/protocol identity differs')
    allowed = old['validation']['allowed_gene_indices']
    expression = identity['data']['training_expression_policy']
    if len(allowed) != 4775 or digest(allowed) != spec['allowed_axis']['sha256'] or parent['allowed_expression_ids'] != allowed or expression['exclude_test_target_expression'] is not True or len(expression['excluded_expression_gene_ids']) != 225 or identity['data']['effective_training_expression_ids_sha256'] != old['validation']['allowed_gene_ids_sha256']:
        raise ValueError('allowed4775/excluded225 boundary differs')
    for source in (spec['native_source'], spec['analysis_source']):
        env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
        head = subprocess.check_output(['git','-C',source['root'],'rev-parse','HEAD'],text=True,env=env).strip()
        dirty = subprocess.check_output(['git','-C',source['root'],'status','--porcelain'],text=True,env=env).strip()
        if head != source['commit'] or dirty or values[source['publication']]['published_commit'] != head:
            raise ValueError('clean original source changed')
    path = Path(spec['analysis_source']['root'])/'scripts/v2/response_anchor_postfit.py'
    loader = importlib.util.spec_from_file_location('_r15_identity_metadata',path)
    module = importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    module.validate_r15_identity(dict(source=spec['native_source'],config=spec['config'],run_manifest=spec['run_manifest']))
    output = Path(spec['output']).resolve()
    for protected in (spec['run_root'],spec['data_root'],spec['native_source']['root'],spec['analysis_source']['root'],old['output']):
        p=Path(protected).resolve()
        if output.is_relative_to(p) or p.is_relative_to(output):
            raise ValueError('output overlaps protected data/source/history')
    if output.exists() or not output.is_relative_to('/data/yilangliu'):
        raise ValueError('fresh server endpoint output required')
    if owner is not None:
        receipt=json.loads(Path(owner['path']).read_text())
        if sha(owner['path']) != owner['sha256'] or receipt.get('schema') != 'response-anchor-analysis-owner-v1' or set(receipt.get('process',{})) != {'pid','uid','start_ticks','command_sha256','script_basename'} or receipt['process']['uid'] != os.getuid() or Path(owner['path']).resolve() != Path(registry['job'])/'execution/DISPATCH_OWNER.json':
            raise ValueError('exact dynamic owner required')
    spec['resource_owner']=owner
    return spec


def commands(registry,spec,binding):
    native=spec['native_source']['root']
    env=dict(CUDA_VISIBLE_DEVICES='0',PYTORCH_ALLOC_CONF='expandable_segments:True',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH=native+'/src:'+native,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    return [dict(endpoint='R15-DIR',env=env,modes=[dict(mode=mode,command=[registry['python'],spec['analysis_source']['root']+'/scripts/v2/response_anchor_medium.py',mode,'--spec',binding['path'],'--spec-sha256',binding['sha256']]) for mode in ('seal','medium','panel','validate')])]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry',type=Path,required=True);p.add_argument('--registry-sha256',required=True)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--check-only',action='store_true');g.add_argument('--seal-dir',type=Path)
    p.add_argument('--owner-path',type=Path);p.add_argument('--owner-sha256')
    args=p.parse_args()
    if not Path.cwd().resolve().is_relative_to('/data/yilangliu') or sha(args.registry) != args.registry_sha256:
        raise ValueError('server-only frozen registry required')
    registry=json.loads(args.registry.read_text())
    owner=None if args.owner_path is None else dict(path=str(args.owner_path),sha256=args.owner_sha256)
    spec=build(registry,owner)
    if args.check_only:
        print(json.dumps(dict(status='metadata_checked_only',counts=COUNTS,owner_bound=owner is not None,checkpoint_opened=False,arrays_opened=False,scientific_complete=False)));return
    if owner is None or args.seal_dir.resolve() != Path(registry['job'])/'metadata-a1' or args.seal_dir.exists():
        raise ValueError('fresh metadata-a1 and dynamic owner required')
    args.seal_dir.mkdir(parents=True,exist_ok=False)
    path=args.seal_dir/'R15-DIR.SPEC.json';write(path,spec)
    binding=dict(path=str(path),sha256=sha(path))
    write(args.seal_dir/'COMMANDS.json',commands(registry,spec,binding))
    write(args.seal_dir/'METADATA_SEAL.json',dict(spec=binding,registry=dict(path=str(args.registry),sha256=args.registry_sha256),counts=COUNTS,scientific_complete=False))
    print(json.dumps(binding))


if __name__=='__main__':
    main()
