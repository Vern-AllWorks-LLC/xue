"""Turn an unpacked relocatable CPython tree into a Xue distribution.

Run with the interpreter being assembled, from the repository root:

    <pkg>/bin/python3 tools/assemble_dist.py <pkg>      (Linux / macOS)
    <pkg>\\python.exe  tools\\assemble_dist.py <pkg>     (Windows)

It installs the `xue` package into the standard library, adds the Xue
start-up hook to site.py (same change as the xue-python fork), copies the
licence and guide, and prints KEY=VALUE lines (paths relative to the current
directory) for the C-extension build step.
"""
import os
import shutil
import sys
import sysconfig

HOOK = """    # xue-python: initialize extensions (unicode ops, strict mode, llm hook)
    try:
        import xue.sitecustomize_xue
    except ImportError:
        pass
"""
ANCHOR = """    if ENABLE_USER_SITE:
        execusercustomize()
"""


def rel(path):
    return os.path.relpath(path).replace(os.sep, '/')


def main(pkg):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stdlib = sysconfig.get_path('stdlib')

    dest = os.path.join(stdlib, 'xue')
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(os.path.join(repo, 'xue-modules'), dest,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))

    site_py = os.path.join(stdlib, 'site.py')
    with open(site_py, encoding='utf-8') as f:
        src = f.read()
    if 'import xue.sitecustomize_xue' not in src:
        assert src.count(ANCHOR) == 1, 'site.py anchor not found exactly once'
        src = src.replace(ANCHOR, ANCHOR + HOOK)
        with open(site_py, 'w', encoding='utf-8', newline='\n') as f:
            f.write(src)

    for name in ('LICENSE.xue', 'xue-guide.html', 'xue-guide.pdf'):
        if os.path.exists(os.path.join(repo, name)):
            shutil.copy(os.path.join(repo, name), pkg)

    ft = bool(sysconfig.get_config_var('Py_GIL_DISABLED'))
    major, minor = sys.version_info[:2]
    print('XUEDIR=' + rel(dest))
    print('PYINC=' + rel(sysconfig.get_path('include')))
    print('PYEXT=' + sysconfig.get_config_var('EXT_SUFFIX'))
    print('PYFT=' + ('1' if ft else '0'))
    print('PYLIB=python%d%d%s' % (major, minor, 't' if ft else ''))


if __name__ == '__main__':
    main(sys.argv[1])
