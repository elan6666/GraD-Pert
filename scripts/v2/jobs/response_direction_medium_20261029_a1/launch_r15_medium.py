"""One persistent script owner; only published D3 native CLI performs science."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry',type=Path,required=True);p.add_argument('--registry-sha256',required=True)
    p.add_argument('--sealer',type=Path,required=True);p.add_argument('--sealer-sha256',required=True)
    p.add_argument('--self-sha256',required=True);p.add_argument('--wrapper-publication',type=Path,required=True);p.add_argument('--wrapper-publication-sha256',required=True)
    args=p.parse_args()
    if not Path.cwd().resolve().is_relative_to('/data/yilangliu'):
        raise ValueError('server-only launcher')
    frozen={str(args.registry):args.registry_sha256,str(args.sealer):args.sealer_sha256,str(Path(__file__).resolve()):args.self_sha256,str(args.wrapper_publication):args.wrapper_publication_sha256}
    registry=json.loads(args.registry.read_text());spec=registry['spec'];analysis=Path(spec['analysis_source']['root'])
    frozen.update(registry['source_artifacts']);frozen.update(registry['metadata'])
    def verify():
        for path,digest in frozen.items():
            if sha(path)!=digest:raise ValueError('immutable job input changed: '+path)
    verify()
    publication=json.loads(args.wrapper_publication.read_text())
    if publication.get('status')!='published_job_artifacts' or publication.get('analysis_source')!=spec['analysis_source']['commit'] or publication.get('files')!={str(Path(__file__).resolve()):args.self_sha256,str(args.sealer):args.sealer_sha256,str(args.registry):args.registry_sha256}:
        raise ValueError('exact job-only wrapper publication required')
    helper=load(analysis/'scripts/v2/jobs/response_anchor_medium_20261029_a2/controller.py','_published_schedule_only')
    shared=load(analysis/'scripts/v2/shared_gpu.py','_published_shared_gpu')
    job=Path(registry['job'])
    if job.exists():raise ValueError('fresh job required; never overwrite/retry')
    job.mkdir(parents=True,exist_ok=False);control=job/'controller';control.mkdir();execution=job/'execution';execution.mkdir()
    owner_raw=shared.process_info(os.getpid())
    if owner_raw is None:raise ValueError('persistent actual owner unavailable')
    owner={key:owner_raw[key] for key in helper.OWNER_FIELDS}
    owner_path=execution/'DISPATCH_OWNER.json';helper.write_new(owner_path,dict(schema='response-anchor-analysis-owner-v1',process=owner))
    owner_binding=dict(path=str(owner_path),sha256=sha(owner_path));frozen[str(owner_path)]=owner_binding['sha256']
    native=spec['native_source']['root'];env=dict(PYTHONPATH=native+'/src:'+native,PYTORCH_ALLOC_CONF='expandable_segments:True',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    command=[registry['python'],str(args.sealer),'--registry',str(args.registry),'--registry-sha256',args.registry_sha256,'--seal-dir',str(job/'metadata-a1'),'--owner-path',str(owner_path),'--owner-sha256',owner_binding['sha256']]
    try:
        process,receipt=helper.spawn_phase(control,'metadata-seal',command,env,shared.process_info,owner,cwd=analysis)
        while process.poll() is None:time.sleep(0.02)
        result=helper.finish_phase(control,process,receipt)
        if result['status']!='completed':raise ValueError('metadata seal/identity failed')
        seal=load(args.sealer,'_frozen_r15_metadata')
        binding=json.loads((job/'metadata-a1/METADATA_SEAL.json').read_text())['spec']
        sealed_spec=json.loads(Path(binding['path']).read_text())
        plan=json.loads((job/'metadata-a1/COMMANDS.json').read_text())
        if sealed_spec != dict(spec,resource_owner=owner_binding) or plan != seal.commands(registry,sealed_spec,binding):raise ValueError('frozen commands/spec differ')
        for path in (Path(binding['path']),job/'metadata-a1/COMMANDS.json',job/'metadata-a1/METADATA_SEAL.json'):frozen[str(path)]=sha(path)
        phases=helper.run_phases(plan,control,shared.process_info,owner,verify,cwd=analysis)
        if phases['status']!='completed' or len(phases['phases'])!=4:raise ValueError('native phases incomplete')
        complete=Path(spec['output'])/'COMPLETE.json';terminal=json.loads(complete.read_text())
        if terminal['spec_sha256']!=binding['sha256'] or terminal['updates']!=0 or terminal['counts']!=registry['counts'] or terminal['panel_rows']!=128 or list(job.rglob('*.pkl')):raise ValueError('native terminal/whole-job zero-PKL failed')
        verify()
        helper.write_new(control/'COMPLETE.json',dict(status='completed',spec=binding,owner=owner_binding,native_complete=dict(path=str(complete),sha256=sha(complete)),analysis_source=spec['analysis_source']['commit'],native_source=spec['native_source']['commit'],wrapper_publication=dict(path=str(args.wrapper_publication),sha256=args.wrapper_publication_sha256),files=frozen,scientific_complete=False))
    except BaseException as error:
        helper.write_new(control/'FAILURE.json',dict(status='failed',error=repr(error),scientific_complete=False));raise


if __name__=='__main__':
    main()
