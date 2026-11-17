#!/bin/sh
# Create the xue-python source tree: a pristine CPython release plus the Xue
# changes (the `xue` package in Lib/ and the start-up hook in site.main()).
#
#   tools/make_fork.sh [VERSION]        default VERSION: 3.15.2
#
# Downloads Python-VERSION.tgz into archives/ (unless already there), unpacks
# it to xue-python/ (an existing xue-python/ is moved to xue-python.prev/),
# and applies the Xue changes. Then build it like any CPython:
#
#   cd xue-python && ./configure --prefix=/opt/xue && make -j && make install
set -eu

VERSION="${1:-3.15.2}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TGZ="archives/Python-$VERSION.tgz"
mkdir -p archives
[ -f "$TGZ" ] || curl -fL -o "$TGZ" "https://www.python.org/ftp/python/$VERSION/Python-$VERSION.tgz"

if [ -e xue-python ]; then
    rm -rf xue-python.prev
    mv xue-python xue-python.prev
    echo "previous tree moved to xue-python.prev/"
fi
tar -xzf "$TGZ"
mv "Python-$VERSION" xue-python

# The xue package becomes part of the standard library.
cp -R xue-modules xue-python/Lib/xue
find xue-python/Lib/xue -name __pycache__ -prune -exec rm -rf {} +

# Start-up hook: import xue.sitecustomize_xue at the end of site.main(),
# after sitecustomize/usercustomize (skipped with -S, like the rest of site).
python3 - xue-python/Lib/site.py <<'EOF'
import re, sys
path = sys.argv[1]
src = open(path, encoding='utf-8').read()
hook = ("    # xue-python: initialize extensions (unicode ops, strict mode, llm hook)\n"
        "    try:\n"
        "        import xue.sitecustomize_xue\n"
        "    except ImportError:\n"
        "        pass\n")
anchor = re.search(r"^    if ENABLE_USER_SITE:\n        execusercustomize\(\)\n", src, re.M)
if not anchor:
    sys.exit("site.py: could not find the end of site.main(); patch it by hand")
src = src[:anchor.end()] + hook + src[anchor.end():]
open(path, 'w', encoding='utf-8').write(src)
print("patched", path)
EOF

echo "xue-python/ is CPython $VERSION + Xue. Build it with ./configure && make (add --disable-gil for free-threaded)."
