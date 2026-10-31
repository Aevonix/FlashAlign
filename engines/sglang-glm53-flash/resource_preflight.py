# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Report necessary storage/VRAM lower bounds; not a peak-memory prediction."""
import argparse
import json
from pathlib import Path
import shutil


def bounds(resources, disk_free, gpu_free=(), host_available=None, disk_floor=100*1024**3):
    payload=resources['payload_bytes']
    header=sum(x['header_bytes']+8 for x in resources['shards'].values())
    checkpoint=payload+header
    failures=[]
    if disk_free < checkpoint+disk_floor:
        failures.append('A new full checkpoint copy would cross the required disk floor')
    if gpu_free and (len(gpu_free)!=4 or sum(gpu_free)<payload):
        failures.append('Four GPUs do not have enough aggregate free memory even for stored tensor payloads')
    if host_available is not None and host_available<resources['largest_tensor']['bytes']:
        failures.append('Host free memory is smaller than the largest checkpoint tensor')
    return {'checkpoint_minimum_file_bytes':checkpoint,'required_pre_download_free_bytes':checkpoint+disk_floor,
            'aggregate_gpu_payload_lower_bound_bytes':payload,'ideal_even_payload_per_gpu_bytes':payload/4,
            'largest_cpu_tensor_bytes':resources['largest_tensor']['bytes'],
            'largest_shard_payload_bytes':resources['largest_shard_payload_bytes'],
            'native_mtp_payload_bytes':resources['component_payload_bytes']['native_mtp'],
            'necessary_bounds_pass':not failures,'failures':failures,
            'launch_qualified':False,
            'limits':'Passing these necessary bounds does not establish allocation success. Runtime workspaces, replicated tensors, recurrent/KV caches, communication and graph-free speculative buffers need measured headroom.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--disk-path',type=Path,required=True)
    parser.add_argument('--resources',type=Path,default=Path(__file__).resolve().parent/'resources.json')
    args=parser.parse_args()
    resources=json.loads(args.resources.read_text())
    host=None
    if Path('/proc/meminfo').exists():
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):host=int(line.split()[1])*1024
    print(json.dumps(bounds(resources,shutil.disk_usage(args.disk_path).free,host_available=host),indent=2))


if __name__=='__main__':main()
