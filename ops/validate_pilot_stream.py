"""Tiny CPU-only parity test: not a GPU training or performance benchmark."""
import socket
import threading
import torch
import exposure.pilot_train as pilot_module
from ops.protocol import recv_json, send_json
from exposure.pilot_train import train, _resume_parts
from ops.conditional_stream import serve_requests
from ops.pilot_stream import train_via_stream

RULES = {'red': [{'type': 'attribute', 'variable': 'x', 'field': 'color', 'value': 'red'}],
         'circle': [{'type': 'attribute', 'variable': 'x', 'field': 'shape', 'value': 'circle'}]}
ARGS = {'batch_size': 2, 'seed': 19, 'device': 'cpu', 'width': 4, 'timesteps': 4,
        'lr': 1e-4, 'dropout': .1, 'ema_decay': .99, 'max_attempts': 10000,
        'deterministic': True}


def tcp_pair():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0)); listener.listen(1)
        consumer = socket.create_connection(listener.getsockname(), timeout=30)
        producer, _ = listener.accept()
    return producer, consumer


def streamed(*, steps, resume=None, batch_size=None):
    producer, consumer = tcp_pair(); errors = []
    def work():
        try:
            with producer: serve_requests(producer, RULES)
        except EOFError: pass
        except Exception as exc: errors.append(repr(exc))
    thread = threading.Thread(target=work); thread.start()
    try:
        with consumer:
            arguments = {**ARGS, **({'batch_size': batch_size} if batch_size else {})}
            result = train_via_stream(RULES, None, consumer, steps=steps, resume=resume, **arguments)
    finally:
        thread.join(10)
    if thread.is_alive() or errors: raise RuntimeError(f'producer failure: {errors}')
    return result


def equal(a, b):
    for section in ('model', 'ema'):
        for key in a[section]:
            if not torch.equal(a[section][key], b[section][key]):
                raise AssertionError(f'{section} {key} differs')
    for key in ('condition_counts', 'realized_rule_counts_by_condition', 'sampling_attempts', 'last_loss'):
        if a['report'][key] != b['report'][key]: raise AssertionError(f'report {key} differs')
    if a['rng']['python'] != b['rng']['python'] or not torch.equal(a['rng']['torch_cpu'], b['rng']['torch_cpu']):
        raise AssertionError('RNG state differs')


def main():
    torch.set_num_threads(1)
    baseline, _ = train(RULES, None, steps=2, **ARGS)
    remote, _ = streamed(steps=2)
    initial, _ = streamed(steps=1)
    resumed, _ = streamed(steps=1, resume=initial)
    equal(baseline, remote); equal(remote, resumed)
    try: _resume_parts(initial)
    except ValueError: pass
    else: raise AssertionError('standard v3 trainer accepted v4 stream checkpoint')
    # A bad server reply must fail without leaving the global renderer patched.
    original = pilot_module.render_conditioned
    producer, consumer = tcp_pair(); errors = []
    def bad_server():
        try:
            with producer:
                recv_json(producer)
                send_json(producer, {'kind': 'invalid'})
        except Exception as exc: errors.append(repr(exc))
    thread = threading.Thread(target=bad_server); thread.start()
    with consumer:
        try: train_via_stream(RULES, None, consumer, steps=1, **ARGS)
        except ValueError: pass
        else: raise AssertionError('bad server response accepted')
    thread.join(10)
    if errors or thread.is_alive() or pilot_module.render_conditioned is not original:
        raise AssertionError(f'renderer not restored: {errors}')
    # Lock must have been released too.
    streamed(steps=1)
    print('PASS: CPU baseline/stream and 1+1 resume model/EMA/loss/counts/RNG parity; v3 rejects v4; failure restores renderer and lock')


if __name__ == '__main__': main()
