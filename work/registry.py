import glyphs as G
from lib import *

# base glyph -> list of recipes (first = default, others = contextual alternates)
LETTERS = {
    'A': [G.orig('A2'), G.orig('A4'), G.orig('A1')],
    'B': [G.B, G.B], 'C': [G.orig('C2'), G.orig('C3'), G.orig('C4')],
    'D': [G.D, G.D], 'E': [G.E, G.E], 'F': [G.orig('F')], 'G': [G.G, G.G],
    'H': [G.orig('H'), G.orig('H', flip=True)], 'I': [G.orig('I'), G.orig('I', flip=True)],
    'J': [G.J, G.J], 'K': [G.K, G.K], 'L': [G.L, G.L], 'M': [G.M, G.M], 'N': [G.orig('N')],
    'O': [G.orig('O1'), G.orig('O1', flip=True)], 'P': [G.P, G.P], 'Q': [G.Q, G.Q],
    'R': [G.orig('R1'), G.orig('R2')], 'S': [G.orig('S2'), G.orig('S1')],
    'T': [G.T, G.T], 'U': [G.U, G.U], 'V': [G.V, G.V], 'W': [G.W, G.W], 'X': [G.X, G.X],
    'Y': [G.Y, G.Y], 'Z': [G.Z, G.Z],
}
ROUND = set('OCSGQ')   # get overshoot (originals)
DIGITS = {'zero': G.zero, 'one': G.one, 'two': G.two, 'three': G.three, 'four': G.four,
          'five': G.five, 'six': G.six, 'seven': G.seven, 'eight': G.eight, 'nine': G.nine}
PUNCT = {n: getattr(G, n) for n in ['period', 'comma', 'colon', 'semicolon', 'exclam', 'question',
         'hyphen', 'endash', 'emdash', 'quotesingle', 'quotedbl', 'quoteleft', 'quoteright',
         'quotedblleft', 'quotedblright', 'parenleft', 'parenright', 'slash', 'plus', 'equal',
         'guillemotleft', 'guillemotright', 'exclamdown', 'questiondown', 'dollar',
         'ordmasculine', 'ordfeminine', 'asterisk', 'degree', 'percent',
         'numbersign', 'underscore', 'bullet', 'ellipsis', 'ampersand', 'periodcentered']}
ACCENTS = {n: getattr(G, n) for n in ['acute', 'grave', 'tilde', 'circumflex', 'dieresis', 'cedilla']}

def variant_name(base, i):
    return base if i == 0 else f'{base}.alt{i}'

def build_all():
    out = {}
    for base, recipes in LETTERS.items():
        for i, r in enumerate(recipes):
            nm = variant_name(base, i)
            m = r(nm)
            if getattr(r, 'original', False) and base in ROUND:
                m = G.overshoot(m)
            out[nm] = m
    for d in (DIGITS, PUNCT, ACCENTS):
        for nm, r in d.items():
            out[nm] = r(nm)
    return out
