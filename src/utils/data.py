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


# Labels
LABEL_NAMES = {
    "newspaper": ['Mail & Guardian', 'Sydney Morning Herald (Australia)',
 'The Age (Melbourne, Australia)', 'The Australian', 'The Hindu',
 'The New York Times','The Times (South Africa)',
 'The Times of India (TOI)', 'The Washington Post'],
    
    "ncountry": ['australia', 'india', 'south_africa', 'us'],
    
    "ncompas": ['left', 'right'],
    
    "month": ['April', 'August', 'December', 'July', 'March', 'November', 'October'],
    
    "year": ['1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004',
 '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014',
 '2015', '2016', '2017', '2018'],
    
    "subject": ['AGREEMENTS', 'AIR POLLUTION', 'AIR QUALITY REGULATION', 'ARMED FORCES',
 'ASSOCIATIONS & ORGANIZATIONS', 'BUSINESS NEWS', 'CAMPAIGNS & ELECTIONS',
 'CHILDREN', 'CHRISTMAS', 'CLIMATE CHANGE',
 'CLIMATE CHANGE REGULATION & POLICY', 'CLIMATOLOGY',
 'CONFERENCES & CONVENTIONS', 'CONSERVATION',
 'CRIME, LAW ENFORCEMENT & CORRECTIONS', 'DEATH & DYING',
 'DEVELOPING COUNTRIES', 'EARTH & ATMOSPHERIC SCIENCE',
 'ECONOMIC CONDITIONS', 'ECONOMIC NEWS', 'ECONOMICS',
 'ECONOMY & ECONOMIC INDICATORS', 'ELECTIONS', 'EMERGING MARKETS',
 'EMISSIONS', 'EMISSIONS CREDITS', 'ENERGY & ENVIRONMENT',
 'ENVIRONMENT & NATURAL RESOURCES',
 'ENVIRONMENTAL & WILDLIFE ORGANIZATIONS', 'ENVIRONMENTAL DEPARTMENTS',
 'ENVIRONMENTAL TREATIES & AGREEMENTS', 'ENVIRONMENTALISM', 'EUROPEAN UNION',
 'EXECUTIVES', 'GLOBAL WARMING', 'GOVERNMENT & PUBLIC ADMINISTRATION',
 'GOVERNMENT ADVISORS & MINISTERS', 'GREENHOUSE GASES',
 'HEADS OF STATE & GOVERNMENT', 'INTERNATIONAL ECONOMIC ORGANIZATIONS',
 'INTERNATIONAL RELATIONS', 'INTERNATIONAL RELATIONS & NATIONAL SECURITY',
 'INTERNATIONAL TRADE', 'INTERVIEWS', 'INVESTIGATIONS', 'ISLANDS & REEFS',
 'LEGISLATION', 'LEGISLATIVE BODIES', 'MAMMALS', 'MANAGERS & SUPERVISORS',
 'MUSLIMS & ISLAM', 'NEGATIVE PERSONAL NEWS', 'OIL & GAS PRICES',
 'POLITICAL PARTIES', 'POLITICS', 'POLLUTION & ENVIRONMENTAL IMPACTS',
 'PRICES', 'PRIME MINISTERS', 'PUBLIC POLICY', 'REGIONAL & LOCAL GOVERNMENTS',
 'RELIGION', 'REPORTS, REVIEWS & SECTIONS', 'SCIENCE & TECHNOLOGY',
 'STATE DEPARTMENTS & FOREIGN SERVICES', 'TALKS & MEETINGS',
 'TAXES & TAXATION', 'TERRORISM', 'TREATIES & AGREEMENTS', 'TRENDS & EVENTS',
 'TYPES OF GOVERNMENT', 'UNITED NATIONS', 'UNITED NATIONS INSTITUTIONS',
 'US FEDERAL GOVERNMENT', 'US PRESIDENTIAL CANDIDATES 2008',
 'US PRESIDENTIAL CANDIDATES 2012', 'US PRESIDENTS', 'US REPUBLICAN PARTY',
 'WAR & CONFLICT', 'WEATHER', 'WRITERS', 'Weather/Greenhouse Effect'],
    
    "geographic": ['ADELAIDE, AUSTRALIA', 'AFGHANISTAN', 'AFRICA', 'ASIA', 'ATLANTIC OCEAN',
 'AUSTRALIA', 'AUSTRALIAN CAPITAL TERRITORY', 'BEIJING, CHINA', 'BRAZIL',
 'BRISBANE, AUSTRALIA', 'CALIFORNIA, USA', 'CANADA', 'CANBERRA, AUSTRALIA',
 'CHINA', 'DISTRICT OF COLUMBIA, USA', 'EARTH', 'EGYPT', 'ENGLAND', 'EUROPE',
 'EUROPEAN UNION MEMBER STATES', 'FLORIDA, USA', 'FRANCE', 'GERMANY', 'INDIA',
 'INDIAN OCEAN', 'INDONESIA', 'IRAN, ISLAMIC REPUBLIC OF', 'IRAQ', 'ISRAEL',
 'JAPAN', "KOREA, DEMOCRATIC PEOPLE'S REPUBLIC OF", 'KOREA, REPUBLIC OF',
 'LONDON, ENGLAND', 'LOS ANGELES, CA, USA', 'MALAYSIA', 'MARYLAND, USA',
 'MELBOURNE, AUSTRALIA', 'MEXICO', 'MIDDLE EAST',
 'MOSCOW, RUSSIAN FEDERATION', 'MUMBAI, MAHARASHTRA, INDIA',
 'NEW DELHI, INDIA', 'NEW SOUTH WALES, AUSTRALIA', 'NEW YORK, NY, USA',
 'NEW YORK, USA', 'NEW ZEALAND', 'NORTH AMERICA', 'NORTH CENTRAL CHINA',
 'NORTHERN TERRITORY, AUSTRALIA', 'PACIFIC OCEAN', 'PAKISTAN',
 'PARIS, FRANCE', 'PERTH, AUSTRALIA', 'PHILIPPINES', 'QUEENSLAND, AUSTRALIA',
 'RUSSIAN FEDERATION', 'SAUDI ARABIA', 'SOUTH AFRICA',
 'SOUTH AUSTRALIA, AUSTRALIA', 'SOUTH CHINA SEA', 'STATE OF PALESTINE',
 'SYDNEY, AUSTRALIA', 'SYRIA', 'TASMANIA, AUSTRALIA', 'TEXAS, USA',
 'TOKYO, JAPAN', 'UNITED KINGDOM', 'UNITED STATES', 'VICTORIA, AUSTRALIA',
 'VIRGINIA, USA', 'WESTERN AUSTRALIA, AUSTRALIA'],
}

from sklearn.preprocessing import MultiLabelBinarizer
LABEL_BINARIZERS = {k: MultiLabelBinarizer().fit([[vv] for vv in v])
                    for k,v in LABEL_NAMES.items()}
