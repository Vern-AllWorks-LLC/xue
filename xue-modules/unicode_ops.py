"""
Unicode operator aliases for scientific Python code.

Registers a custom codec 'xue-unicode' that translates Unicode mathematical
operators to their Python equivalents during source parsing.

Supported mappings:
    ≤  →  <=          ≥  →  >=          ≠  →  !=
    ×  →  *           ÷  →  /           ¬  →  not
    ∧  →  and         ∨  →  or          ←  →  =
    ≡  →  ==          ∈  →  in          ∉  →  not in
    ≔  →  :=          √  →  math.sqrt
    ∞  →  float('inf')
    π  →  3.141592653589793
    τ  →  6.283185307179586
    ε  →  1e-10

Usage:
    # coding: xue-unicode

    α = 0.01
    if x ≤ 10 ∧ y ≠ 0:
        result = x ÷ y

Or activate globally:
    import xue.unicode_ops
    xue.unicode_ops.register()
"""

from __future__ import annotations
import codecs
import io

# Unicode → Python source mappings
# Only map operators, not identifiers (Python already supports Unicode identifiers)
_OPERATOR_MAP = {
    '\u2264': '<=',       # ≤
    '\u2265': '>=',       # ≥
    '\u2260': '!=',       # ≠
    '\u2261': '==',       # ≡
    '\u00d7': '*',        # ×
    '\u00f7': '/',        # ÷
    '\u00ac': 'not ',     # ¬
    '\u2227': ' and ',    # ∧
    '\u2228': ' or ',     # ∨
    '\u2190': '=',        # ←
    '\u2254': ':=',       # ≔
    '\u2208': ' in ',     # ∈
    '\u2209': ' not in ', # ∉
    '\u221e': "float('inf')",  # ∞
    '\u221a': 'math.sqrt',     # √ (used as function prefix)
}

# Mathematical constants — these are identifiers, not operators.
# Python already allows them as variable names, but we can provide defaults.
_CONSTANT_MAP = {
    '\u03c0': '3.141592653589793',    # π
    '\u03c4': '6.283185307179586',    # τ
}


def _translate_source(source: str) -> str:
    """Replace Unicode operators with their Python equivalents."""
    for unicode_char, replacement in _OPERATOR_MAP.items():
        source = source.replace(unicode_char, replacement)
    return source


class XueUnicodeCodec(codecs.Codec):
    """Codec that translates Unicode operators during source decoding."""

    def decode(self, data: bytes, errors: str = 'strict'):
        # First decode as UTF-8
        text, length = codecs.utf_8_decode(data, errors)
        # Then translate Unicode operators
        translated = _translate_source(text)
        return translated, length

    def encode(self, text: str, errors: str = 'strict'):
        return codecs.utf_8_encode(text, errors)


class XueIncrementalDecoder(codecs.IncrementalDecoder):
    def decode(self, data: bytes, final: bool = False):
        text = data.decode('utf-8', self.errors)
        return _translate_source(text)


class XueIncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, text: str, final: bool = False):
        return text.encode('utf-8', self.errors)


class XueStreamReader(XueUnicodeCodec, codecs.StreamReader):
    pass


class XueStreamWriter(XueUnicodeCodec, codecs.StreamWriter):
    pass


def _search_function(name: str):
    if name in ('xue-unicode', 'xue_unicode'):
        return codecs.CodecInfo(
            name='xue-unicode',
            encode=XueUnicodeCodec().encode,
            decode=XueUnicodeCodec().decode,
            incrementalencoder=XueIncrementalEncoder,
            incrementaldecoder=XueIncrementalDecoder,
            streamreader=XueStreamReader,
            streamwriter=XueStreamWriter,
        )
    return None


_registered = False


def register():
    """Register the xue-unicode codec. Call once at interpreter startup."""
    global _registered
    if not _registered:
        codecs.register(_search_function)
        _registered = True


def translate_file(path: str) -> str:
    """Read a Python source file and return it with Unicode operators translated."""
    with open(path, 'r', encoding='utf-8') as f:
        source = f.read()
    return _translate_source(source)
