"""Room text features; structured columns alone supply bed configuration."""
import re
import numpy as np
import pandas as pd

FAMILIES = {'layout': ['suite',
            'junior_suite',
            'studio',
            'apartment',
            'villa',
            'connecting',
            'explicit_bedrooms',
            'explicit_area_m2'],
 'marketing': ['standard',
               'deluxe',
               'superior',
               'premium',
               'executive',
               'club',
               'family',
               'classic',
               'luxury',
               'economy'],
 'accessibility': ['hearing', 'mobility', 'accessible', 'roll_in_shower', 'bathtub'],
 'room_amenities': ['kitchenette',
                    'kitchen',
                    'fridge',
                    'microwave',
                    'balcony',
                    'terrace',
                    'non_smoking',
                    'smoking',
                    'pet_friendly'],
 'view_location': ['city_view',
                   'ocean_view',
                   'pool_view',
                   'courtyard_view',
                   'mountain_view',
                   'partial_view',
                   'corner',
                   'high_floor',
                   'run_of_house'],
 'format_metadata': ['text_length']}

PATTERNS = {'suite': '\\bsuite\\b',
 'junior_suite': '\\bjunior suite\\b',
 'studio': '\\bstudio\\b',
 'apartment': '\\bapartment\\b',
 'villa': '\\bvilla\\b',
 'standard': '\\bstandard\\b',
 'deluxe': '\\bdeluxe\\b',
 'superior': '\\bsuperior\\b',
 'premium': '\\bpremium\\b',
 'executive': '\\bexecutive\\b',
 'club': '\\bclub\\b',
 'family': '\\bfamily\\b',
 'classic': '\\bclassic\\b',
 'luxury': '\\bluxury\\b',
 'economy': '\\beconomy\\b',
 'hearing': '\\bhearing\\b',
 'mobility': '\\bmobility\\b',
 'accessible': '\\baccessible\\b',
 'roll_in_shower': 'roll[ -]?in',
 'bathtub': '\\b(?:bathtub|tub)\\b',
 'kitchenette': '\\bkitchenette\\b',
 'kitchen': '\\bkitchen\\b',
 'fridge': '\\b(?:mini[ -]?fridge|refrigerator|fridge)\\b',
 'microwave': '\\bmicrowave\\b',
 'balcony': '\\bbalcony\\b',
 'terrace': '\\bterrace\\b',
 'city_view': '\\bcity view\\b',
 'ocean_view': '\\b(?:ocean|sea) view\\b',
 'pool_view': '\\bpool view\\b',
 'courtyard_view': '\\bcourtyard view\\b',
 'mountain_view': '\\bmountain view\\b',
 'partial_view': '\\bpartial\\b',
 'corner': '\\bcorner\\b',
 'high_floor': '\\bhigh floor\\b',
 'run_of_house': 'run of (?:the )?house',
 'non_smoking': '\\bnon[ -]?smoking\\b',
 'smoking': '(?<!non-)(?<!non )\\bsmoking\\b',
 'pet_friendly': '\\bpet friendly\\b',
 'connecting': '\\bconnect(?:ing|ed)\\b'}


COMPILED = {name: re.compile(pattern) for name, pattern in PATTERNS.items()}
NUMBER_MAP = {'one': '1', 'two': '2', 'three': '3', 'four': '4'}
NUMBER_PATTERN = re.compile(r'\b(one|two|three|four)\b')
# Redact recognized bed configurations before parsing, length, or vectorization.
# Bedroom counts and floor area remain separate room-layout attributes.
_QTY = r'(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a|an)'
_TYPE = (r'(?:(?:cal(?:ifornia)?|super|eastern|western)[ -]*)?kings?'
         r'|queens?|twins?(?:[ -]*xl)?|singles?|doubles?|fulls?'
         r'|sofa[ -]?beds?|sofabds?|sleeper[ -]?sofas?|sofas?|sleepers?|bunk(?:s|ing|house)?'
         r'|murphy|futons?|trundles?|roll[ -]?aways?|pull[ -]?outs?|day[ -]?beds?'
         r'|semi[ -]?double|kng|kgs?|qns?|dbl|dble|db|twns?|sgls?|kb|qb|tb')
BED_PHRASE = re.compile(
    rf'\b(?:{_QTY}\s*(?:[x×]\s*)?)?'
    rf'(?:(?:extra[ -]?large|extra[ -]?long|large|small|oversized|top|bottom|long)\s+)?(?:{_TYPE}|beds?)'
    rf'(?:[ -]*(?:size|sized))?(?:\s*beds?)?'
    rf'(?:\s*(?:[x×:]\s*){_QTY})?\b', re.IGNORECASE,
)
BED_CODE = re.compile(rf'\b{_QTY}\s*[x×]?\s*(?:kb|qb|db|tb|k|q)\b', re.IGNORECASE)


def room_text_without_beds(text):
    """Remove recognized bed phrases, quantities and alternatives from room text.

    This is conservative lexical redaction, not a guarantee about every unseen
    language/abbreviation. No bed type or count is ever inferred from the result.
    """
    if pd.isna(text):
        return ''
    if not isinstance(text, str):
        raise TypeError('Description must be a string or a missing value.')
    text = re.sub(r'\\[nrt]', ' ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text).lower()
    text = BED_CODE.sub('\x1f', BED_PHRASE.sub('\x1f', text))
    # Supplier concatenations such as King1, KingSG and HearinSofa.
    text = re.sub(r'\b(?:(?:king|queen|twin|sofa|bunk|double|single)\w*|\w*(?:queen|twin|sofa|bunk|double|single))\b', '\x1f', text)
    text = re.sub(r'\x1f(?:\s*(?:and|or|[,/+&])\s*\x1f)+', '\x1f', text)
    text = text.replace('\x1f', ' ')
    text = re.sub(r'\(\s*\)|\[\s*\]', ' ', text)
    text = re.sub(r'\s*[,;|/]+\s*', ' ', text)
    return re.sub(r'\s+', ' ', text).strip(' ,;:+/|-&')


def normalize_description(text):
    """Normalize room text after bed-configuration redaction."""
    return NUMBER_PATTERN.sub(lambda match: NUMBER_MAP[match.group()], room_text_without_beds(text))


def parse_description(txt):
    """Extract 41 non-bed attributes and bed-redacted length; unknown values stay NaN."""
    t = normalize_description(txt)
    f = {k: int(bool(p.search(t))) for k, p in COMPILED.items()}
    f['text_length'] = len(t)
    m = re.search(r'\b([1-9])[ -]bedrooms?\b', t)
    f['explicit_bedrooms'] = float(m.group(1)) if m else np.nan
    m = re.search(r'\b([0-9]+(?:\.[0-9]+)?)\s*(?:square m(?:eters?)?\b|sq\.?\s*m\b|m²)', t)
    area = float(m.group(1)) if m else np.nan
    if not m:
        m = re.search(r'\b([0-9]+(?:\.[0-9]+)?)\s*(?:square (?:ft|feet)\b|sq\.?\s*ft\b)', t)
        if m:
            area = float(m.group(1)) * 0.092903
    f['explicit_area_m2'] = area if 5 <= area <= 2000 else np.nan
    return f
