"""
Hardcoded values and mappings for the specific dataset and accompanying task.
"""

# Newspaper name to newspaper alignment
NEWSPAPER_TO_COMPAS = {
    'Mail & Guardian': "left",
    'Sydney Morning Herald (Australia)': "left",
    'The Age (Melbourne, Australia)': "left",
    'The Australian': "right",
    'The Hindu': "left",
    'The New York Times': "left",
    'The Times (South Africa)': "right",
    'The Times of India (TOI)': "right",
    'The Washington Post': "left",
}

# Newspaper name to country code
NEWSPAPER_TO_COUNTRY = {
    'Sydney Morning Herald (Australia)': "australia",
    'The Australian': "australia",
    'The Age (Melbourne, Australia)': "australia",
    'The Hindu': "india",
    'The Times of India (TOI)': "india",
    'Mail & Guardian': "south_africa",
    'The Times (South Africa)': "south_africa",
    'The New York Times': "us",
    'The Washington Post': "us",
}

# Country codename to printable label
COUNTRY_TO_PRETTY = {
    "us": "US",
    "india": "India",
    "south_africa": "South Africa",
    "australia": "Australia",
}

# Input variables
X_KEYS_LIST = ["headline", "body"]
X_KEYS = set(X_KEYS_LIST)

# Output variables
Y_KEYS_LIST = [
    "newspaper", "ncountry", "ncompas",
    "month", "year", "subject", "geographic",
]
Y_KEYS = set(Y_KEYS_LIST)

# Output variables which are single-output
Y_KEYS_LOCAL = Y_KEYS - {"subject", "geographic"}

# This may be unnecessary because since Python 3.8, Python guarantees
# stable ordering based on insertion order.
Y_KEYS_FIXED = list(Y_KEYS)
Y_KEYS_LOCAL_FIXED = list(Y_KEYS_LOCAL)

# Single-character code for each output variable 
Y_KEYS_TO_CODE = {
    "newspaper": "n",
    "ncountry": "c",
    "ncompas": "o",
    "month": "m",
    "year": "y",
    "subject": "s",
    "geographic": "g",
}
# Reverse code to variable codes
CODE_TO_Y_KEYS = {v: k for k, v in Y_KEYS_TO_CODE.items()}

# Variable code to printable label
Y_KEYS_PRETTY = {
    "newspaper": "Newspaper",
    "ncountry": "News. country",
    "ncompas": "News. align.",
    "month": "Month",
    "year": "Year",
    "subject": "Subject",
    "geographic": "Geographic",
}
