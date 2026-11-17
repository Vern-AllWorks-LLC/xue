"""End-to-end feature check for an assembled Xue distribution.

Run with the distribution's own interpreter (any working directory):

    <pkg>/bin/python3 -I tools/feature_check.py      (Linux / macOS)
    <pkg>\\python.exe  -I tools\\feature_check.py     (Windows)

The unit tests in xue/tests cover the library APIs; this script covers what
they cannot: the start-up hook (.pth) and everything it switches on in a
fresh interpreter -- the xue-unicode source codec, --strict / XUE_STRICT,
--llm / XUE_LLM_HOOK against mock HTTP, OpenAI and Unix-socket backends --
plus sandboxed imports, package signing, Secret redaction, the C
accelerators, and threaded use of the accelerators on free-threaded builds.
Exits non-zero if any check fails.
"""
import http.server
import json
import os
import socket
import subprocess
import sys
import sysconfig
import tempfile
import textwrap
import threading

PY = sys.executable
FT = bool(sysconfig.get_config_var('Py_GIL_DISABLED'))
failures = []


def check(name, cond, detail=''):
    print(('ok    ' if cond else 'FAIL  ') + name + (f'  [{detail}]' if detail and not cond else ''))
    if not cond:
        failures.append(name)


def run(code, *args, env=None, flags=('-I',), cwd=None):
    """Run code (or a script path) in a fresh interpreter; return CompletedProcess."""
    e = {k: v for k, v in os.environ.items() if not k.startswith('XUE_')}
    e.update(env or {})
    cmd = [PY, *flags] + (['-c', code] if not os.path.isfile(code) else [code]) + list(args)
    return subprocess.run(cmd, env=e, cwd=cwd, capture_output=True, text=True,
                          encoding='utf-8', timeout=120)


# ---------------------------------------------------------------- mock LLMs
class _Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path.startswith('/v1/chat/completions'):
            ok = (self.headers.get('Authorization') == 'Bearer test-key'
                  and body['messages'][-1]['role'] == 'user')
            out = {'choices': [{'message': {'content': 'MOCK-OPENAI ' + ('auth-ok' if ok else 'auth-bad')}}]}
        else:
            out = {'response': 'MOCK-HTTP model=%s has-traceback=%s'
                   % (body.get('model'), 'Traceback' in body.get('prompt', ''))}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def start_http():
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, 'http://127.0.0.1:%d' % srv.server_address[1]


def start_unix(path):
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(4)

    def serve():
        while True:
            try:
                c, _ = srv.accept()
            except OSError:
                return
            with c:
                n = int.from_bytes(c.recv(4), 'big')
                buf = b''
                while len(buf) < n:
                    buf += c.recv(n - len(buf))
                req = json.loads(buf)
                data = json.dumps({'text': 'MOCK-UNIX got=%s' % ('prompt' in req)}).encode()
                c.sendall(len(data).to_bytes(4, 'big') + data)

    threading.Thread(target=serve, daemon=True).start()
    return srv


# ------------------------------------------------------------------ checks
def main():
    print(sys.version)
    tmp = tempfile.mkdtemp(prefix='xue-check-')

    # 1. Start-up hook and package metadata.
    r = run("import sys, xue; print('xue.sitecustomize_xue' in sys.modules, xue.__version__, xue.__based_on__)")
    check('start-up hook loads at interpreter start', r.stdout.startswith('True'), r.stdout + r.stderr)
    check('xue.__based_on__ matches the interpreter',
          r.stdout.strip().endswith('CPython %d.%d.%d' % sys.version_info[:3]), r.stdout)
    r = run("import sys; print('xue.sitecustomize_xue' in sys.modules)", flags=('-I', '-S'))
    check('python -S skips the hook', r.stdout.strip() == 'False', r.stdout + r.stderr)

    # 2. xue-unicode source codec (registered by the hook).
    src = os.path.join(tmp, 'uni.py')
    with open(src, 'w', encoding='utf-8') as f:
        f.write(textwrap.dedent('''\
            # coding: xue-unicode
            import math
            x ← 6
            y ← 3
            ok ← x ≥ y ∧ y ≠ 0 ∧ x ≤ 10 ∧ ¬ (x ≡ y) ∧ 3 ∈ [1, 2, 3] ∧ 4 ∉ [1]
            print(ok, x × y, x ÷ y, √(16.0), π, ∞)
            '''))
    r = run(src)
    check('# coding: xue-unicode operators', r.stdout.split() == ['True', '18', '2.0', '4.0', '3.141592653589793', 'inf'],
          r.stdout + r.stderr)

    # 3. Strict mode via environment and --strict.
    strict_src = os.path.join(tmp, 'strict_demo.py')
    with open(strict_src, 'w', encoding='utf-8') as f:
        f.write(textwrap.dedent('''\
            import sys
            from xue import strict
            def label(w: float) -> str:
                return 'w=%s' % (w,)
            strict.strict_module(__name__)
            print('enabled', strict.is_enabled(), 'argv', sys.argv[1:])
            try:
                label('a')
                print('no-error')
            except strict.StrictTypeError as e:
                print('caught', type(e).__name__)
            '''))
    r = run(strict_src, env={'XUE_STRICT': '1'})
    check('XUE_STRICT=1 enforces annotations', 'enabled True' in r.stdout and 'caught StrictTypeError' in r.stdout,
          r.stdout + r.stderr)
    r = run(strict_src, '--strict', 'keep')
    check('--strict enables strict mode and is removed from argv',
          "enabled True argv ['keep']" in r.stdout and 'caught' in r.stdout, r.stdout + r.stderr)
    r = run(strict_src)
    check('strict mode off by default', 'enabled False' in r.stdout and 'no-error' in r.stdout, r.stdout + r.stderr)

    # 4. LLM hook: HTTP backend through the start-up hook (env), --llm, and the OpenAI / Unix backends.
    srv, base = start_http()
    boom = os.path.join(tmp, 'boom.py')
    with open(boom, 'w', encoding='utf-8') as f:
        f.write('import sys\nprint("argv", sys.argv[1:])\n1 / 0\n')
    llm_env = {'XUE_LLM_HOOK': '1', 'XUE_LLM_BACKEND': 'http',
               'XUE_LLM_URL': base + '/api/generate', 'XUE_LLM_MODEL': 'mock'}
    r = run(boom, env=llm_env)
    check('XUE_LLM_HOOK explains unhandled exceptions',
          'ZeroDivisionError' in r.stderr and 'MOCK-HTTP model=mock has-traceback=True' in r.stdout,
          r.stdout + r.stderr)
    env2 = dict(llm_env)
    del env2['XUE_LLM_HOOK']
    r = run(boom, '--llm', 'x', env=env2)
    check('--llm installs the hook and is removed from argv',
          "argv ['x']" in r.stdout and 'MOCK-HTTP' in r.stdout, r.stdout + r.stderr)
    r = run("from xue import llmhook as L\n"
            "L.configure(backend='openai', url=%r, api_key='test-key', model='m')\n"
            "print(L.ask('hi', context={'n': 1}))" % (base + '/v1/chat/completions'))
    check('llmhook OpenAI-compatible backend', 'MOCK-OPENAI auth-ok' in r.stdout, r.stdout + r.stderr)
    r = run("from xue import llmhook as L\nprint(L.ask('hi'))")
    check('llmhook disabled by default', 'not enabled' in r.stdout, r.stdout + r.stderr)
    if hasattr(socket, 'AF_UNIX') and os.name != 'nt':
        sock = os.path.join(tmp, 'llmd.sock')
        usrv = start_unix(sock)
        r = run("from xue import llmhook as L\n"
                "L.configure(backend='unix', socket_path=%r)\n"
                "print(L.suggest_fix('x = 1/0', 'ZeroDivisionError'))" % sock)
        check('llmhook Unix-socket backend', 'MOCK-UNIX got=True' in r.stdout, r.stdout + r.stderr)
        usrv.close()
    srv.shutdown()

    # 5. Sandbox and package signing.
    pkgdir = os.path.join(tmp, 'pkgs')
    os.makedirs(os.path.join(pkgdir, 'netplugin'))
    with open(os.path.join(pkgdir, 'netplugin', '__init__.py'), 'w') as f:
        f.write('import ctypes\n')
    with open(os.path.join(pkgdir, 'mathplugin.py'), 'w') as f:
        f.write('def sq(x):\n    return x * x\n')
    r = run(textwrap.dedent('''\
        import sys
        sys.path.insert(0, %r)
        from xue.sandbox import (sandboxed_import, sandbox, SandboxViolation, POLICY_STRICT,
                                 register_trusted_hash, require_signed, verify_package,
                                 SignatureError, _compute_package_hash)
        try:
            sandboxed_import('netplugin', policy=POLICY_STRICT); print('not-blocked')
        except SandboxViolation as e:
            print('blocked', e.capability)
        m = sandboxed_import('mathplugin', policy=POLICY_STRICT); print('sq', m.sq(7))
        try:
            with sandbox(allow_subprocess=False):
                import pty
            print('not-blocked')
        except SandboxViolation as e:
            print('blocked', e.capability)
        import os, mathplugin
        h = _compute_package_hash(mathplugin.__file__)
        register_trusted_hash('mathplugin', h); require_signed('mathplugin'); print('signed-ok')
        register_trusted_hash('mathplugin', '0' * 64)
        try:
            require_signed('mathplugin'); print('tamper-missed')
        except SignatureError:
            print('tamper-detected')
        ''' % pkgdir))
    check('sandboxed_import blocks ctypes', 'blocked ctypes/ffi' in r.stdout, r.stdout + r.stderr)
    check('sandboxed_import allows pure modules', 'sq 49' in r.stdout, r.stdout + r.stderr)
    check('sandbox() context blocks subprocess modules', 'blocked subprocess' in r.stdout, r.stdout + r.stderr)
    check('require_signed accepts / rejects hashes',
          'signed-ok' in r.stdout and 'tamper-detected' in r.stdout, r.stdout + r.stderr)

    # 6. Secret, Result, contracts, dispatch smoke (APIs are unit-tested; this is the integrated path).
    r = run(textwrap.dedent('''\
        import json, pickle
        from xue.secret import Secret
        from xue.result import Ok, Err, Some, Nothing
        from xue.contracts import requires, ContractViolation
        from xue.dispatch import multimethod
        s = Secret('sk-123')
        out = [repr(s), str(s), f'{s}', s.expose()]
        for f in (lambda: pickle.dumps(s), lambda: json.dumps(s)):
            try:
                f(); out.append('leak')
            except TypeError:
                out.append('blocked')
        out.append(Ok(2).map(lambda v: v * 3).unwrap())
        out.append(Err('e').unwrap_or(5))
        @requires(lambda x: x > 0)
        def f(x): return x
        try:
            f(-1); out.append('no-contract')
        except ContractViolation:
            out.append('contract')
        @multimethod
        def add(a: int, b: int): return 'int'
        @multimethod
        def add(a: str, b: str): return 'str'
        out += [add(1, 2), add('a', 'b')]
        print(out)
        '''))
    check('Secret redaction / Result / contracts / dispatch',
          r.stdout.strip() == "['Secret(****)', 'Secret(****)', 'Secret(****)', 'sk-123', 'blocked', "
          "'blocked', 6, 5, 'contract', 'int', 'str']", r.stdout + r.stderr)

    # 7. C accelerators: loaded, and correct against the pure-Python reference.
    r = run(textwrap.dedent('''\
        import sys, warnings
        warnings.simplefilter('ignore', RuntimeWarning)
        from xue import units, autodiff, tensor
        accel = sorted(m for m in sys.modules if m.endswith('_accel'))
        print(accel)
        from xue.units import meters, seconds
        print((10 * meters / (2 * seconds)))
        from xue.autodiff import grad
        print(grad(lambda x: x * x * x)(2.0))
        from xue.tensor import Tensor
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        print((a @ a).tolist())
        '''))
    lines = r.stdout.splitlines()
    check('all three C accelerators load', len(lines) > 0 and lines[0].count('_accel') == 3, r.stdout + r.stderr)
    check('accelerated units / autodiff / tensor results',
          len(lines) == 4 and lines[2] in ('12.0', '12') and lines[3] == '[[7.0, 10.0], [15.0, 22.0]]',
          r.stdout + r.stderr)

    # 8. Free-threaded builds: hammer the accelerators from several threads with the GIL off.
    if FT:
        r = run(textwrap.dedent('''\
            import sys, threading
            from xue.tensor import Tensor
            from xue.autodiff import grad
            from xue.units import meters
            errs = []
            def work():
                try:
                    for _ in range(300):
                        t = Tensor([[1.0, 2.0], [3.0, 4.0]])
                        assert (t @ t).tolist() == [[7.0, 10.0], [15.0, 22.0]]
                        assert grad(lambda x: x * x)(3.0) == 6.0
                        assert ((2 * meters) + (3 * meters)).value == 5
                except Exception as e:
                    errs.append(repr(e))
            ts = [threading.Thread(target=work) for _ in range(8)]
            [t.start() for t in ts]; [t.join() for t in ts]
            print('gil', sys._is_gil_enabled(), 'errors', errs)
            '''), flags=('-I', '-X', 'gil=0'))
        check('free-threaded: accelerators under 8 threads with -X gil=0',
              'gil False errors []' in r.stdout, r.stdout + r.stderr)

    print()
    if failures:
        print('%d check(s) FAILED: %s' % (len(failures), ', '.join(failures)))
        sys.exit(1)
    print('all feature checks passed')


if __name__ == '__main__':
    main()
