"""
Hardcoded 
"""

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

COUNTRY_TO_PRETTY = {
    "us": "US",
    "india": "India",
    "south_africa": "South Africa",
    "australia": "Australia",
}


X_KEYS = {"headline", "body"}
Y_KEYS = {
    "newspaper", "ncountry", "ncompas",
    "month", "year", "subject", "geographic"
}
Y_KEYS_LOCAL = Y_KEYS - {"subject", "geographic"}

Y_KEYS_FIXED = list(Y_KEYS)
Y_KEYS_LOCAL_FIXED = list(Y_KEYS_LOCAL)

Y_KEYS_TO_CODE = {
    "newspaper": "n",
    "ncountry": "c",
    "ncompas": "o",
    "month": "m",
    "year": "y",
    "subject": "s",
    "geographic": "g",
}
CODE_TO_Y_KEYS = {v: k for k, v in Y_KEYS_TO_CODE.items()}
Y_KEYS_PRETTY = {
    "newspaper": "Newspaper",
    "ncountry": "News. country",
    "ncompas": "News. align.",
    "month": "Month",
    "year": "Year",
    "subject": "Subject",
    "geographic": "Geographic",
}
