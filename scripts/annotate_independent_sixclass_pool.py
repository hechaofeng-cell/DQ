#!/usr/bin/env python3
"""Reuse the frozen online/local VLM implementation on pool B only."""
import argparse
import annotate_dual_vlm as adapter
from independent_confirmation_common import BASE

def configure():
    adapter.STUDY=BASE
    adapter.job_path=lambda backend,domain:BASE/'annotations'/backend/domain
    adapter.schema_path=lambda domain:BASE/'configs'/('car_annotation_schema.json' if domain=='car' else 'old_annotation_schema.json')
    adapter.prompt_path=lambda domain:BASE/'prompts'/f'{domain}.txt'

if __name__=='__main__':
    configure()
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['local','online'],required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    for domain in ('old','car'):
        (adapter.run_local if args.backend=='local' else adapter.run_online)(domain,0,args.smoke)
