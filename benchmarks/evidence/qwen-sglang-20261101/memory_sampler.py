#!/usr/bin/env python3
"""Private fixed-deadline memory sampler; it never starts or signals an engine.

The default GPU domain is NVML device used memory. Explicitly selecting
system_memory_pressure instead measures MemTotal - MemAvailable from Linux
/proc/meminfo; GPU-used memory stays unavailable, and no discrete-GPU memory
gate is established. That system-wide metric is not a replacement estimate of
GPU allocations. It belongs to a separate hardware-specific comparison.
The host domain is the sum of VmRSS
equivalents for the recorded supervisor and its verified descendants, so shared
pages may be counted more than once. These domains are never added together.
Sampling is scheduled every 50 ms, without catch-up bursts. Adequacy requires
no skipped ticks, both measurements on every sample, at least two samples, all
gaps including window edges within the sealed maximum, and a normal stop-file
ending. Actual achieved Hz is reported, without rounding it up to 20.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from types import ModuleType


SCHEMA = 'flashalign.private-memory-sampler.v1'
PERIOD = 0.05
MAX_DURATION = 3600.0
MAX_SAMPLES = 72001
MAX_ERRORS = 256
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
TELEMETRY = ('temperature_c', 'graphics_clock_mhz', 'sm_clock_mhz', 'memory_clock_mhz',
             'power_watts', 'power_limit_watts', 'throttle_reasons')
MEMORY_MODES = ('gpu_used', 'system_memory_pressure')


def _finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and 0 <= value < 10**15
    except OverflowError:
        return False


def _identity(value):
    return (isinstance(value, dict) and set(value) == {'pid', 'start', 'uid'}
            and type(value.get('pid')) is int and value['pid'] > 0
            and isinstance(value.get('start'), str) and value['start'].isdigit()
            and type(value.get('uid')) is int and value['uid'] == os.getuid())


def _private_bytes(path, limit):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_uid != os.getuid()
                or stat.S_IMODE(before.st_mode) != 0o600 or before.st_size > limit):
            raise ValueError('Unsafe private sampler input')
        chunks, remaining = [], limit + 1
        while remaining:
            chunk = os.read(fd, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(fd)
        data = b''.join(chunks)
        if len(data) > limit or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('Sampler input changed while reading')
        return data
    finally:
        os.close(fd)


def read_anchor(path):
    def unique(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('Duplicate process identity field')
            value[key] = item
        return value
    value = json.loads(_private_bytes(path, 4096), object_pairs_hook=unique)
    if not _identity(value):
        raise ValueError('Expected an owned process birth identity')
    return value


def load_helper(path, expected_sha256):
    if not isinstance(expected_sha256, str) or re.fullmatch(r'[0-9a-f]{64}', expected_sha256) is None:
        raise ValueError('A pinned helper SHA-256 is required')
    source = _private_bytes(path, 1024 * 1024)
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError('Supervisor helper hash differs')
    # Execute the exact hashed bytes, not a second path read or a cached .pyc.
    module = ModuleType('private_sampler_verified_supervisor')
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    if not callable(getattr(module, 'identity', None)) or not callable(getattr(module, 'descendants', None)):
        raise ValueError('Pinned helper does not implement process ownership')
    return module


def read_rss(record):
    """Read only a verified PID's resident pages, never its command or content."""
    with (Path('/proc') / str(record['pid']) / 'statm').open() as stream:
        fields = stream.read(4096).split()
    if len(fields) < 2 or not fields[1].isdigit():
        raise ValueError('Invalid resident-page measurement')
    return int(fields[1]) * os.sysconf('SC_PAGE_SIZE')


def read_system_memory(path=Path('/proc/meminfo')):
    """Whole-system memory pressure, with no process/GPU allocation inference."""
    with Path(path).open() as stream:
        text = stream.read(65537)
    if len(text) > 65536:
        raise ValueError('System memory metadata exceeds its bound')
    values = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] not in ('MemTotal:', 'MemAvailable:'):
            continue
        name = fields[0][:-1]
        if name in values or len(fields) != 3 or fields[2] != 'kB' or not fields[1].isdigit():
            raise ValueError('Invalid system memory metadata')
        values[name] = int(fields[1]) * 1024
    total, available = values.get('MemTotal'), values.get('MemAvailable')
    if type(total) is not int or type(available) is not int or not 0 <= available <= total < 2**63 or total == 0:
        raise ValueError('System total/available memory is unavailable')
    return {'mem_total_bytes': total, 'mem_available_bytes': available,
            'system_memory_pressure_bytes': total - available}


def owned_rss(anchor, helper, reader=read_rss):
    if helper.identity(anchor['pid']) != anchor:
        raise ProcessLookupError('Owned supervisor is absent')
    rows = helper.descendants([anchor])
    if not isinstance(rows, list):
        raise ValueError('Invalid owned process snapshot')
    total, counted, seen = 0, 0, set()
    for row in rows:
        if not _identity(row):
            raise ValueError('Process snapshot contains an unowned identity')
        key = (row['pid'], row['start'], row['uid'])
        if key in seen:
            continue
        seen.add(key)
        if helper.identity(row['pid']) != row:
            continue
        try:
            resident = reader(row)
        except (FileNotFoundError, ProcessLookupError):
            if helper.identity(row['pid']) != row:
                continue
            raise
        if helper.identity(row['pid']) != row:
            continue
        if type(resident) is not int or not 0 <= resident < 2**63:
            raise ValueError('Invalid owned resident-memory value')
        total += resident
        counted += 1
    if helper.identity(anchor['pid']) != anchor or anchor['pid'] not in {key[0] for key in seen}:
        raise ProcessLookupError('Owned supervisor disappeared during sampling')
    if not counted or total >= 2**63:
        raise ValueError('Owned memory snapshot is unavailable')
    return total, counted


class NvmlDevice:
    """NVML is imported and initialized only for an explicitly invoked run."""
    def __init__(self, gpu_uuid):
        if not isinstance(gpu_uuid, str) or re.fullmatch(r'GPU-[A-Za-z0-9-]{1,128}', gpu_uuid) is None:
            raise ValueError('An explicit GPU UUID is required')
        self.gpu_uuid = gpu_uuid
        self.nvml = None
        self.handle = None

    def open(self):
        import pynvml
        pynvml.nvmlInit()
        self.nvml = pynvml
        self.handle = pynvml.nvmlDeviceGetHandleByUUID(self.gpu_uuid.encode())
        return self

    def close(self):
        if self.nvml is not None:
            self.nvml.nvmlShutdown()
            self.nvml = None
            self.handle = None

    def _optional(self, function, *args, divisor=1):
        try:
            value = getattr(self.nvml, function)(self.handle, *args)
            if divisor != 1:
                value /= divisor
            return value if _finite(value) else None
        except Exception:
            return None

    def telemetry_snapshot(self):
        if self.nvml is None or self.handle is None:
            raise RuntimeError('NVML device is not initialized')
        value = {name: None for name in TELEMETRY}
        temperature = getattr(self.nvml, 'NVML_TEMPERATURE_GPU', None)
        if temperature is not None:
            value['temperature_c'] = self._optional('nvmlDeviceGetTemperature', temperature)
        for name, kind in (('graphics_clock_mhz', 'NVML_CLOCK_GRAPHICS'),
                           ('sm_clock_mhz', 'NVML_CLOCK_SM'), ('memory_clock_mhz', 'NVML_CLOCK_MEM')):
            clock = getattr(self.nvml, kind, None)
            if clock is not None:
                value[name] = self._optional('nvmlDeviceGetClockInfo', clock)
        value['power_watts'] = self._optional('nvmlDeviceGetPowerUsage', divisor=1000)
        value['power_limit_watts'] = self._optional('nvmlDeviceGetEnforcedPowerLimit', divisor=1000)
        value['throttle_reasons'] = self._optional('nvmlDeviceGetCurrentClocksThrottleReasons')
        return value


    def snapshot(self):
        if self.nvml is None or self.handle is None:
            raise RuntimeError('NVML device is not initialized')
        used = self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used
        if type(used) is not int or not 0 <= used < 2**63:
            raise ValueError('Invalid NVML device used memory')
        return {'gpu_used_bytes': used, **self.telemetry_snapshot()}


def sample(anchor, helper, device, *, stop_requested, max_duration_seconds,
           max_gap_seconds=0.10, clock=time.monotonic, sleep=time.sleep, rss_reader=read_rss,
           memory_mode='gpu_used', system_reader=read_system_memory):
    """Pure orchestration with injected readers/clocks; it never sends signals."""
    if not _identity(anchor):
        raise ValueError('Invalid owned process identity')
    if not _finite(max_duration_seconds) or not 0 < max_duration_seconds <= MAX_DURATION:
        raise ValueError('Sampling duration must be positive and bounded')
    if not _finite(max_gap_seconds) or not PERIOD <= max_gap_seconds <= 1:
        raise ValueError('A bounded sealed maximum sampling gap is required')
    if memory_mode not in MEMORY_MODES:
        raise ValueError('Unsupported explicit memory sampling mode')
    primary_domain = memory_mode
    primary_field = 'gpu_used_bytes' if memory_mode == 'gpu_used' else 'system_memory_pressure_bytes'
    started = clock()
    if not _finite(started):
        raise ValueError('Invalid monotonic clock')
    deadline, tick, skipped = started, 0, 0
    rows, errors = [], []
    error_count = 0
    reason = None

    def record_error(domain, error):
        nonlocal error_count
        error_count += 1
        if len(errors) < MAX_ERRORS:
            errors.append({'sample_index': len(rows), 'domain': domain, 'error_type': type(error).__name__})

    while True:
        now = clock()
        if not _finite(now) or now < started:
            raise ValueError('Monotonic clock moved backwards')
        if now < deadline:
            sleep(deadline - now)
            now = clock()
        try:
            if stop_requested():
                reason = 'stop_file'
                break
            if helper.identity(anchor['pid']) != anchor:
                reason = 'anchor_gone'
                break
        except Exception as error:
            record_error('control', error)
            reason = 'control_error'
            break
        if now - started >= max_duration_seconds:
            reason = 'maximum_duration'
            break
        if len(rows) >= MAX_SAMPLES:
            reason = 'sample_budget'
            break
        row = {'t_monotonic': now, 'gpu_used_bytes': None, 'system_memory_pressure_bytes': None,
               'mem_total_bytes': None, 'mem_available_bytes': None, 'host_rss_bytes': None,
               'owned_process_count': None, **{name: None for name in TELEMETRY}}
        try:
            if memory_mode == 'gpu_used':
                device_value = device.snapshot()
                used = device_value.get('gpu_used_bytes')
                if type(used) is not int or not 0 <= used < 2**63:
                    raise ValueError('Missing device used-memory measurement')
                row['gpu_used_bytes'] = used
            else:
                memory = system_reader()
                total, available = memory.get('mem_total_bytes'), memory.get('mem_available_bytes')
                if (type(total) is not int or type(available) is not int
                        or not 0 <= available <= total < 2**63 or total == 0
                        or memory.get('system_memory_pressure_bytes') != total - available):
                    raise ValueError('Invalid system memory pressure measurement')
                row.update({name: memory[name] for name in ('mem_total_bytes', 'mem_available_bytes', primary_field)})
                # Telemetry is optional and does not call the unsupported GPU
                # memory API. Its absence never creates a numeric zero.
                try:
                    device_value = device.telemetry_snapshot() if hasattr(device, 'telemetry_snapshot') else {}
                except Exception:
                    device_value = {}
            for key in TELEMETRY:
                value = device_value.get(key)
                row[key] = value if _finite(value) else None
        except Exception as error:
            record_error(primary_domain, error)
        try:
            row['host_rss_bytes'], row['owned_process_count'] = owned_rss(anchor, helper, rss_reader)
        except Exception as error:
            record_error('host_rss', error)
        rows.append(row)
        tick += 1
        deadline = started + tick * PERIOD
        after = clock()
        if after > deadline:
            missed = math.floor((after - deadline) / PERIOD) + 1
            skipped += missed
            tick += missed
            deadline = started + tick * PERIOD
    ended = clock()
    if not _finite(ended) or ended < started or rows and ended < rows[-1]['t_monotonic']:
        raise ValueError('Invalid sampling endpoint clock')
    gaps = ([rows[0]['t_monotonic'] - started]
            + [b['t_monotonic'] - a['t_monotonic'] for a, b in zip(rows, rows[1:])]
            + [ended - rows[-1]['t_monotonic']]) if rows else [ended - started]
    maximum_gap = max(gaps)
    valid_primary = [row[primary_field] for row in rows if row[primary_field] is not None]
    valid_host = [row['host_rss_bytes'] for row in rows if row['host_rss_bytes'] is not None]
    adequate = (reason == 'stop_file' and len(rows) >= 2 and skipped == 0 and error_count == 0
                and maximum_gap <= max_gap_seconds
                and len(valid_primary) == len(valid_host) == len(rows))
    observed_rate = ((len(rows) - 1) / (rows[-1]['t_monotonic'] - rows[0]['t_monotonic'])
                     if len(rows) >= 2 and rows[-1]['t_monotonic'] > rows[0]['t_monotonic'] else None)
    return {'schema': SCHEMA, 'anchor': anchor, 'target_hz': 20, 'memory_mode': memory_mode,
            'started_monotonic': started, 'ended_monotonic': ended, 'stop_reason': reason,
            'sample_count': len(rows), 'scheduled_ticks': tick, 'skipped_ticks': skipped,
            'actual_hz_between_first_and_last': observed_rate, 'observed_max_gap_seconds': maximum_gap,
            'sealed_max_gap_seconds': max_gap_seconds, 'sampling_adequate': adequate,
            'domains': {
                primary_domain: {'kind': 'sampled', 'peak_bytes': max(valid_primary) if valid_primary else None,
                             'physical_metric': ('NVML device used memory' if memory_mode == 'gpu_used' else
                                                 'Linux MemTotal minus MemAvailable, whole-system memory pressure'),
                             'valid_samples': len(valid_primary), 'measurement_adequate': adequate,
                             'sample_hz': 20, 'sample_count': len(rows), 'max_gap_seconds': maximum_gap},
                'host_rss': {'kind': 'sampled', 'peak_bytes': max(valid_host) if valid_host else None,
                             'valid_samples': len(valid_host), 'measurement_adequate': adequate,
                             'sample_hz': 20, 'sample_count': len(rows), 'max_gap_seconds': maximum_gap}},
            'definitions': {'gpu_used': 'NVML device used bytes, not allocator-reserved or live-tensor bytes',
                            'system_memory_pressure': 'whole-system MemTotal - MemAvailable; not GPU-only allocation and not additive with process RSS',
                            'host_rss': 'sum of RSS for the birth-verified supervisor and its verified descendants; shared pages may be counted repeatedly',
                            'cadence': '50 ms monotonic deadlines; missed deadlines are skipped and recorded, never catch-up sampled',
                            'sample_hz_field': 'scheduled rate; achieved rate is reported separately',
                            'actual_hz': '(n-1)/(last sample timestamp-first sample timestamp), without rounding to target',
                            'peak_limit': 'sampled maxima can miss short peaks; startup and overlap experiments are separate',
                            'domains_added_together': False},
            'unavailable_domains': ({'gpu_used': {'peak_bytes': None, 'gate': 'unestablished',
                'reason': 'Not measured in explicit system_memory_pressure mode; discrete-GPU comparison remains separate'}}
                if memory_mode == 'system_memory_pressure' else {}),
            'errors': errors, 'error_count': error_count, 'errors_truncated': error_count > len(errors),
            'samples': rows}


def sample_v2(anchor, helper, device, *, stop_requested, stop_endpoint_ns,
              max_duration_seconds, max_gap_seconds, minimum_paired_coverage,
              memory_mode='gpu_used', system_reader=read_system_memory,
              clock_ns=time.monotonic_ns, sleep=time.sleep, rss_reader=read_rss):
    """Explicit request-stop window with integer deadlines; all raw rows survive.

    This is a new protocol. It never reinterprets a v1 receipt. The requested
    stop endpoint is recorded by the collector before setting its stop event.
    Sampling adequacy describes coverage, not capture of the true memory peak.
    """
    period_ns = 50_000_000
    if not _identity(anchor):raise ValueError('Invalid owned process identity')
    if not _finite(max_duration_seconds) or not 0 < max_duration_seconds <= MAX_DURATION:
        raise ValueError('Sampling duration must be positive and bounded')
    if not _finite(max_gap_seconds) or not PERIOD <= max_gap_seconds <= 1:
        raise ValueError('A bounded sealed maximum sampling gap is required')
    if (type(minimum_paired_coverage) not in (float, int)
            or not math.isfinite(minimum_paired_coverage) or not 0 < minimum_paired_coverage <= 1):
        raise ValueError('A finite coverage fraction in (0, 1] is required')
    if memory_mode not in MEMORY_MODES:raise ValueError('Unsupported memory mode')
    def now_ns():
        value=clock_ns()
        if type(value) is not int or not 0 <= value < 2**63:raise ValueError('Invalid monotonic ns clock')
        return value
    start=now_ns();slot=0;rows=[];errors=[];error_count=0;skipped=[];reason=None
    primary_field='gpu_used_bytes' if memory_mode=='gpu_used' else 'system_memory_pressure_bytes'
    def error(domain, exc):
        nonlocal error_count
        error_count+=1
        if len(errors)<MAX_ERRORS:errors.append({'sample_index':len(rows),'domain':domain,'error_type':type(exc).__name__})
    while True:
        deadline=start+slot*period_ns;now=now_ns()
        if now<start:raise ValueError('Monotonic clock moved backwards')
        if now<deadline:sleep((deadline-now)/1e9);now=now_ns()
        try:
            if stop_requested():reason='stop_file';break
            if helper.identity(anchor['pid'])!=anchor:reason='anchor_gone';break
        except Exception as exc:error('control',exc);reason='control_error';break
        now=now_ns()  # Control/identity reads may themselves cross the endpoint.
        if now-start >= int(max_duration_seconds*1e9):reason='maximum_duration';break
        if len(rows)>=MAX_SAMPLES:reason='sample_budget';break
        row={'slot_index':slot,'scheduled_ns':deadline,'started_ns':now,'completed_ns':None,
             't_monotonic':now/1e9,'gpu_used_bytes':None,'system_memory_pressure_bytes':None,
             'mem_total_bytes':None,'mem_available_bytes':None,'host_rss_bytes':None,
             'owned_process_count':None,**{name:None for name in TELEMETRY}}
        try:
            if memory_mode=='gpu_used':
                values=device.snapshot();used=values.get('gpu_used_bytes')
                if type(used) is not int or not 0<=used<2**63:raise ValueError('Missing device memory')
                row[primary_field]=used
            else:
                values={};memory=system_reader();total=memory.get('mem_total_bytes');available=memory.get('mem_available_bytes')
                if (type(total) is not int or type(available) is not int or not 0<=available<=total<2**63
                        or total==0 or memory.get(primary_field)!=total-available):raise ValueError('Invalid system memory')
                row.update({name:memory[name] for name in ('mem_total_bytes','mem_available_bytes',primary_field)})
                try:values=device.telemetry_snapshot() if hasattr(device,'telemetry_snapshot') else {}
                except Exception:values={}
            for name in TELEMETRY:row[name]=values.get(name) if _finite(values.get(name)) else None
        except Exception as exc:error(memory_mode,exc)
        try:row['host_rss_bytes'],row['owned_process_count']=owned_rss(anchor,helper,rss_reader)
        except Exception as exc:error('host_rss',exc)
        completed=now_ns()
        if completed<now:raise ValueError('Monotonic clock moved backwards during read')
        row['completed_ns']=completed;rows.append(row);slot+=1
        # Equality remains eligible: only deadlines strictly before read
        # completion are skipped, so a read ending on a deadline can sample it.
        next_slot=max(slot,(completed-start+period_ns-1)//period_ns)
        if next_slot>slot:
            skipped.append({'first_slot':slot,'end_slot_exclusive':next_slot,
                            'first_deadline_ns':start+slot*period_ns,'read_completed_ns':completed})
            slot=next_slot
    observed=now_ns()
    requested=stop_endpoint_ns() if reason=='stop_file' else None
    if requested is not None and (type(requested) is not int or not 0<=requested<=observed):
        raise ValueError('Invalid requested stop endpoint')
    end=requested if requested is not None else observed
    eligible=0 if end<=start else ((end-start-1)//period_ns)+1
    inside=[r for r in rows if r['scheduled_ns']<end and r['started_ns']<end]
    paired=[r for r in inside if r[primary_field] is not None and r['host_rss_bytes'] is not None]
    paired_slots={r['slot_index'] for r in paired}
    if len(paired_slots)!=len(paired):raise ValueError('Duplicate assigned sample slot')
    coverage=len(paired)/eligible if eligible else 0.0
    if coverage>1:raise ValueError('More valid samples than eligible deadlines')
    points=[start,*[r['started_ns'] for r in paired],max(start,end)]
    gaps_ns=[b-a for a,b in zip(points,points[1:])]
    if any(g<0 for g in gaps_ns):raise ValueError('Sample timestamps are unordered')
    max_gap=max(gaps_ns,default=0)/1e9
    adequate=(reason=='stop_file' and requested is not None and len(paired)>=2 and error_count==0
              and coverage>=minimum_paired_coverage and max_gap<=max_gap_seconds)
    domains={}
    for name,field in ((memory_mode,primary_field),('host_rss','host_rss_bytes')):
        values=[r[field] for r in inside if r[field] is not None]
        domains[name]={'kind':'sampled','peak_bytes':max(values) if values else None,
            'valid_samples':len(values),'measurement_adequate':adequate,'sample_hz':20,
            'sample_count':len(inside),'max_gap_seconds':max_gap,
            'coverage_protocol':'request-stop-coverage-v2','eligible_scheduled_ticks':eligible,
            'paired_valid_samples':len(paired),'paired_coverage':coverage,
            'minimum_paired_coverage':minimum_paired_coverage,'error_count':error_count}
    times=[r['started_ns'] for r in paired]
    actual_hz=(len(times)-1)*1e9/(times[-1]-times[0]) if len(times)>=2 and times[-1]>times[0] else None
    return {'schema':'flashalign.private-memory-sampler.v2','anchor':anchor,'target_hz':20,'period_ns':period_ns,
        'memory_mode':memory_mode,'started_ns':start,'requested_stop_ns':requested,'observed_stop_ns':observed,
        'started_monotonic':start/1e9,'ended_monotonic':end/1e9,'stop_reason':reason,
        'sample_count':len(rows),'eligible_sample_count':len(inside),'excluded_sample_count':len(rows)-len(inside),
        'eligible_scheduled_ticks':eligible,'scheduled_ticks':eligible,'paired_valid_samples':len(paired),
        'paired_coverage':coverage,'sealed_minimum_paired_coverage':minimum_paired_coverage,
        'skipped_ticks':eligible-len({r['slot_index'] for r in inside}),
        'scheduler_skipped_ticks':sum(r['end_slot_exclusive']-r['first_slot'] for r in skipped),
        'skipped_deadline_ranges':skipped,'actual_hz_between_first_and_last':actual_hz,
        'gap_ns':gaps_ns,'observed_max_gap_seconds':max_gap,'sealed_max_gap_seconds':max_gap_seconds,
        'straddling_samples':sum(r['completed_ns']>end for r in inside),'sampling_adequate':adequate,'domains':domains,
        'definitions':{'window':'[sampler start, collector stop request), integer monotonic ns',
            'coverage':'paired-valid unique assigned slots with sample start inside window / ideal deadlines inside window',
            'excluded_rows':'raw rows after requested endpoint remain in samples and are excluded from window metrics',
            'gap':'start/paired sample-start/stop-request edge intervals; observed stop-poll tail recorded separately',
            'peak_limit':'sampled maxima can miss transient peaks; coverage does not establish true peak capture',
            'host_rss':'sum of birth-verified process RSS; shared pages may be counted repeatedly','domains_added_together':False},
        'unavailable_domains':({'gpu_used':{'peak_bytes':None,'gate':'unestablished'}} if memory_mode=='system_memory_pressure' else {}),
        'errors':errors,'error_count':error_count,'errors_truncated':error_count>len(errors),'samples':rows}


def validate_v2_receipt(receipt, *, max_gap_seconds, minimum_paired_coverage):
    """Recompute the coverage gate from retained rows before analysis."""
    def need(value):
        if not value:raise ValueError('Inconsistent v2 memory receipt')
    need(isinstance(receipt,dict) and receipt.get('schema')=='flashalign.private-memory-sampler.v2')
    start,end,observed=(receipt.get(k) for k in ('started_ns','requested_stop_ns','observed_stop_ns'))
    need(all(type(x) is int and 0<=x<2**63 for x in (start,end,observed)) and start<=end<=observed)
    need(receipt.get('period_ns')==50_000_000 and receipt.get('sealed_max_gap_seconds')==max_gap_seconds
         and receipt.get('sealed_minimum_paired_coverage')==minimum_paired_coverage)
    rows=receipt.get('samples');need(isinstance(rows,list) and len(rows)<=MAX_SAMPLES)
    primary='gpu_used_bytes' if receipt.get('memory_mode')=='gpu_used' else 'system_memory_pressure_bytes'
    need(receipt.get('memory_mode') in MEMORY_MODES)
    seen=set();last=start;missing=0;expected_slot=0;expected_ranges=[]
    for row in rows:
        need(isinstance(row,dict))
        slot,scheduled,began,completed=(row.get(k) for k in ('slot_index','scheduled_ns','started_ns','completed_ns'))
        need(all(type(x) is int and x>=0 for x in (slot,scheduled,began,completed))
             and slot==expected_slot and slot not in seen and scheduled==start+slot*50_000_000 and scheduled<=began<=completed
             and last<=began<=completed<=observed)
        seen.add(slot);last=completed
        expected_slot=max(slot+1,(completed-start+49_999_999)//50_000_000)
        if expected_slot>slot+1:
            expected_ranges.append({'first_slot':slot+1,'end_slot_exclusive':expected_slot,
                'first_deadline_ns':start+(slot+1)*50_000_000,'read_completed_ns':completed})
        for field in (primary,'host_rss_bytes'):
            need(row.get(field) is None or type(row[field]) is int and 0<=row[field]<2**63)
            missing+=row.get(field) is None
    inside=[r for r in rows if r['scheduled_ns']<end and r['started_ns']<end]
    paired=[r for r in inside if r[primary] is not None and r['host_rss_bytes'] is not None]
    eligible=0 if end<=start else (end-start-1)//50_000_000+1
    fraction=len(paired)/eligible if eligible else 0.0
    points=[start,*[r['started_ns'] for r in paired],end];gaps=[b-a for a,b in zip(points,points[1:])]
    gap=max(gaps,default=0)/1e9
    need(receipt.get('sample_count')==len(rows) and receipt.get('eligible_sample_count')==len(inside)
         and receipt.get('excluded_sample_count')==len(rows)-len(inside)
         and receipt.get('eligible_scheduled_ticks')==eligible and receipt.get('paired_valid_samples')==len(paired)
         and receipt.get('paired_coverage')==fraction and receipt.get('gap_ns')==gaps
         and receipt.get('observed_max_gap_seconds')==gap and receipt.get('scheduled_ticks')==eligible
         and receipt.get('skipped_ticks')==eligible-len(inside)
         and receipt.get('skipped_deadline_ranges')==expected_ranges
         and receipt.get('scheduler_skipped_ticks')==sum(r['end_slot_exclusive']-r['first_slot'] for r in expected_ranges)
         and receipt.get('straddling_samples')==sum(r['completed_ns']>end for r in inside))
    errors=receipt.get('error_count');error_rows=receipt.get('errors')
    need(type(errors) is int and errors>=missing and isinstance(error_rows,list)
         and len(error_rows)==min(errors,MAX_ERRORS)
         and receipt.get('errors_truncated') is (errors>MAX_ERRORS))
    adequate=(receipt.get('stop_reason')=='stop_file' and len(paired)>=2 and errors==0
              and fraction>=minimum_paired_coverage and gap<=max_gap_seconds)
    need(receipt.get('sampling_adequate') is adequate)
    for domain,field in ((receipt['memory_mode'],primary),('host_rss','host_rss_bytes')):
        value=receipt.get('domains',{}).get(domain);need(isinstance(value,dict))
        values=[r[field] for r in inside if r[field] is not None]
        need(value.get('peak_bytes')==(max(values) if values else None) and value.get('valid_samples')==len(values)
             and value.get('kind')=='sampled' and value.get('sample_hz')==20 and value.get('sample_count')==len(inside)
             and value.get('measurement_adequate') is adequate and value.get('paired_valid_samples')==len(paired)
             and value.get('eligible_scheduled_ticks')==eligible and value.get('paired_coverage')==fraction
             and value.get('error_count')==errors and value.get('max_gap_seconds')==gap
             and value.get('minimum_paired_coverage')==minimum_paired_coverage
             and value.get('coverage_protocol')=='request-stop-coverage-v2')
    return adequate


def _private_directory(path):
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError('Sampler control directory must be private and owned')


def stop_file_exists(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600):
        raise ValueError('Unsafe sampler stop file')
    return True


def write_receipt(path, value):
    _private_directory(path.parent)
    encoded = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise ValueError('Sampler receipt exceeds its size bound')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--process-record', type=Path, required=True)
    parser.add_argument('--supervisor-helper', type=Path, required=True)
    parser.add_argument('--supervisor-sha256', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--stop-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-duration-seconds', type=float, required=True)
    parser.add_argument('--max-gap-seconds', type=float, required=True)
    parser.add_argument('--memory-mode', choices=MEMORY_MODES, default='gpu_used')
    args = parser.parse_args()
    _private_directory(args.stop_file.parent)
    _private_directory(args.output.parent)
    if args.output.exists():
        raise ValueError('Sampler output already exists')
    anchor = read_anchor(args.process_record)
    helper = load_helper(args.supervisor_helper, args.supervisor_sha256)
    device = NvmlDevice(args.gpu_uuid)
    began = time.monotonic()
    try:
        device.open()
        result = sample(anchor, helper, device, stop_requested=lambda: stop_file_exists(args.stop_file),
                        max_duration_seconds=args.max_duration_seconds, max_gap_seconds=args.max_gap_seconds,
                        memory_mode=args.memory_mode)
    except Exception as error:
        result = {'schema': SCHEMA, 'anchor': anchor, 'target_hz': 20, 'sampling_adequate': False,
                  'memory_mode': args.memory_mode,
                  'stop_reason': 'sampler_error', 'error_count': 1,
                  'started_monotonic': began, 'ended_monotonic': time.monotonic(),
                  'observed_max_gap_seconds': None, 'sealed_max_gap_seconds': args.max_gap_seconds,
                  'domains': {name: {'kind': 'sampled', 'peak_bytes': None, 'valid_samples': 0,
                                    'measurement_adequate': False} for name in (args.memory_mode, 'host_rss')},
                  'unavailable_domains': ({'gpu_used': {'peak_bytes': None, 'gate': 'unestablished',
                      'reason': 'Not measured in explicit system_memory_pressure mode; discrete-GPU comparison remains separate'}}
                      if args.memory_mode == 'system_memory_pressure' else {}),
                  'errors': [{'domain': 'sampler', 'error_type': type(error).__name__}], 'samples': [], 'sample_count': 0}
    finally:
        try:
            device.close()
        except Exception as error:
            result['sampling_adequate'] = False
            result['error_count'] += 1
            result['errors'].append({'domain': 'nvml_shutdown', 'error_type': type(error).__name__})
    result.update(gpu_uuid=args.gpu_uuid, supervisor_sha256=args.supervisor_sha256,
                  sampler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_receipt(args.output, result)
    print(json.dumps({'receipt_written': True, 'sampling_adequate': result['sampling_adequate'],
                      'sample_count': result['sample_count'], 'stop_reason': result['stop_reason']}))
    return 0 if result['sampling_adequate'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
