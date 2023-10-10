'''
**************** PLEASE READ ***************

Script that reads in spam and ham messages and converts each training example
into a feature vector

Code intended for UC Berkeley course CS 189/289A: Machine Learning

Requirements:
-scipy ('pip install scipy')

To add your own features, create a function that takes in the raw text and
word frequency dictionary and outputs a int or float. Then add your feature
in the function 'def generate_feature_vector'

The output of your file will be a .mat file. The data will be accessible using
the following keys:
    -'training_data'
    -'training_labels'
    -'test_data'

Please direct any bugs to kevintee@berkeley.edu
'''

from collections import defaultdict
import glob
import re
import scipy.io
import numpy as np
import pdb

NUM_TRAINING_EXAMPLES = 4172
NUM_TEST_EXAMPLES = 1000

BASE_DIR = '../data/'
SPAM_DIR = 'spam/'
HAM_DIR = 'ham/'
TEST_DIR = 'test/'

# ************* Features *************

# Features that look for certain words
def freq_pain_feature(text, freq):
    return float(freq['pain'])

def freq_private_feature(text, freq):
    return float(freq['private'])

def freq_bank_feature(text, freq):
    return float(freq['bank'])

def freq_money_feature(text, freq):
    return float(freq['money'])

def freq_drug_feature(text, freq):
    return float(freq['drug'])

def freq_spam_feature(text, freq):
    return float(freq['spam'])

def freq_prescription_feature(text, freq):
    return float(freq['prescription'])

def freq_creative_feature(text, freq):
    return float(freq['creative'])

def freq_height_feature(text, freq):
    return float(freq['height'])

def freq_featured_feature(text, freq):
    return float(freq['featured'])

def freq_differ_feature(text, freq):
    return float(freq['differ'])

def freq_width_feature(text, freq):
    return float(freq['width'])

def freq_other_feature(text, freq):
    return float(freq['other'])

def freq_energy_feature(text, freq):
    return float(freq['energy'])

def freq_business_feature(text, freq):
    return float(freq['business'])

def freq_message_feature(text, freq):
    return float(freq['message'])

def freq_volumes_feature(text, freq):
    return float(freq['volumes'])

def freq_revision_feature(text, freq):
    return float(freq['revision'])

def freq_path_feature(text, freq):
    return float(freq['path'])

def freq_meter_feature(text, freq):
    return float(freq['meter'])

def freq_memo_feature(text, freq):
    return float(freq['memo'])

def freq_planning_feature(text, freq):
    return float(freq['planning'])

def freq_pleased_feature(text, freq):
    return float(freq['pleased'])

def freq_record_feature(text, freq):
    return float(freq['record'])

def freq_out_feature(text, freq):
    return float(freq['out'])

# Features that look for certain characters
def freq_semicolon_feature(text, freq):
    return text.count(';')

def freq_dollar_feature(text, freq):
    return text.count('$')

def freq_sharp_feature(text, freq):
    return text.count('#')

def freq_exclamation_feature(text, freq):
    return text.count('!')

def freq_para_feature(text, freq):
    return text.count('(')

def freq_bracket_feature(text, freq):
    return text.count('[')

def freq_and_feature(text, freq):
    return text.count('&')

# --------- Add your own feature methods ----------
def example_feature(text, freq):
    return int('example' in text)

# Generates a feature vector
def generate_feature_vector(text, freq):
    feature = []
    feature.append(freq_pain_feature(text, freq))
    feature.append(freq_private_feature(text, freq))
    feature.append(freq_bank_feature(text, freq))
    feature.append(freq_money_feature(text, freq))
    feature.append(freq_drug_feature(text, freq))
    feature.append(freq_spam_feature(text, freq))
    feature.append(freq_prescription_feature(text, freq))
    feature.append(freq_creative_feature(text, freq))
    feature.append(freq_height_feature(text, freq))
    feature.append(freq_featured_feature(text, freq))
    feature.append(freq_differ_feature(text, freq))
    feature.append(freq_width_feature(text, freq))
    feature.append(freq_other_feature(text, freq))
    feature.append(freq_energy_feature(text, freq))
    feature.append(freq_business_feature(text, freq))
    feature.append(freq_message_feature(text, freq))
    feature.append(freq_volumes_feature(text, freq))
    feature.append(freq_revision_feature(text, freq))
    feature.append(freq_path_feature(text, freq))
    feature.append(freq_meter_feature(text, freq))
    feature.append(freq_memo_feature(text, freq))
    feature.append(freq_planning_feature(text, freq))
    feature.append(freq_pleased_feature(text, freq))
    feature.append(freq_record_feature(text, freq))
    feature.append(freq_out_feature(text, freq))
    feature.append(freq_semicolon_feature(text, freq))
    feature.append(freq_dollar_feature(text, freq))
    feature.append(freq_sharp_feature(text, freq))
    feature.append(freq_exclamation_feature(text, freq))
    feature.append(freq_para_feature(text, freq))
    feature.append(freq_bracket_feature(text, freq))
    feature.append(freq_and_feature(text, freq))

    # --------- Add your own features here ---------
    # Make sure type is int or float

    return feature
wordlst = ['!', '"', '#', '%', '&', "'", '(', ')', ',', '-', '.', '/',  ':', ';', '=', '>', '?', '??', '???', '????', '?????', '@',  '^', '_', '`', 'a', 'abdv', 'accessories', 'acton', 'actuals', 'adjustment', 'adjustments', 'adobe', 'aeor', 'aep', 'aepin', 'aerofoam', 'afternoon', 'agua', 'aimee', 'albrecht', 'allocated', 'allocation', 'allocations', 'alt', 'am', 'ami', 'and', 'anita', 'anticipates', 'any', 'aol', 'aopen', 'apc', 'approx', 'april', 'archer', 'are', 'artprice', 'as', 'assigned', 'assignment', 'assurance', 'at', 'atleast', 'attached', 'austin', 'availabilities', 'avails', 'avila', 'bammel', 'baseload', 'baumbach', 'be', 'beaty', 'beaumont', 'been', 'bellamy', 'bgcolor', 'billed', 'biz', 'boas', 'bob', 'border', 'br', 'braband', 'brad', 'brandywine', 'brazos', 'brenda', 'brent', 'brian', 'briley', 'bruce', 'bryan', 'bryce', 'btu', 'but', 'buyback', 'buyer', 'by', 'call', 'calpine', 'camp', 'can', 'canon', 'carlos', 'carthage', 'cass', 'cc', 'cdnow', 'cec', 'cellpadding', 'centana', 'cernosek', 'cf', 'change', 'charlene', 'cheryl', 'chokshi', 'chris', 'christy', 'cia', 'cialis', 'cilco', 'cleburne', 'clem', 'clickathome', 'clynes', 'coastal', 'color', 'com', 'comments', 'complaints', 'computron', 'comstock', 'congress', 'connie', 'connor', 'consemiu', 'construed', 'contract', 'copano', 'corel', 'cornhusker', 'corp', 'corrected', 'correction', 'cotten', 'cotton', 'counterparties', 'counterparty', 'cousino', 'cowboy', 'cp', 'cpr', 'creative', 'crosstex', 'crow', 'csikos', 'customerservice', 'cynthia', 'd', 'daily', 'dan', 'daniel', 'daren', 'darren', 'darron', 'dave', 'david', 'day', 'deal', 'dealer', 'deals', 'deliveries', 'demokritos', 'devon', 'dfarmer', 'differ', 'discreet', 'discrepancies', 'div', 'do', 'donald', 'donna', 'dosage', 'drug', 'drugs', 'dth', 'duke', 'dulce', 'dynegy', 'e', 'eastrans', 'easttexas', 'eb', 'ebs', 'ect', 'edmondson', 'edu', 'eel', 'ees', 'egm', 'egmnom', 'ehronline', 'eiben', 'eileen', 'elsa', 'emirates', 'employee', 'ena', 'encina', 'enerfin', 'energy', 'engage', 'enquiries', 'enron', 'enrononline', 'enronxgate', 'enserch', 'entex', 'enw', 'eogi', 'eol', 'epgt', 'epson', 'equistar', 'erections', 'ews', 'exception', 'expedia', 'explode', 'export', 'farmer', 'fat', 'featured', 'feedback', 'female', 'ferc', 'file', 'flag', 'flow', 'flowed', 'flowing', 'following', 'follows', 'font', 'fontfont', 'for', 'foresee', 'forwarded', 'fred', 'from', 'ftworth', 'fuels', 'fyi', 'gains', 'garrick', 'gary', 'gas', 'gco', 'gcs', 'gd', 'gdp', 'george', 'giron', 'gisb', 'glover', 'gmt', 'gold', 'goliad', 'gomes', 'gottlob', 'gpgfin', 'gr', 'gra', 'graphics', 'graves', 'greg', 'greif', 'gtc', 'guadalupe', 'gulf', 'hakemack', 'hall', 'hanks', 'harris', 'has', 'have', 'hawkins', 'header', 'heads', 'heather', 'heidi', 'height', 'henderson', 'hernandez', 'herod', 'herrera', 'hesco', 'hesse', 'hewlett', 'hillary', 'hilliard', 'hol', 'holmes', 'hopefully', 'hotlist', 'hottlist', 'hou', 'houston', 'howard', 'hpl', 'hplc', 'hplnl', 'hplno', 'hplnol', 'hplo', 'hplr', 'hr', 'href', 'hsc', 'htmlimg', 'http', 'hub', 'hughes', 'hull', 'i', 'if', 'iferc', 'iit', 'illustrator', 'imbalance', 'img', 'impacted', 'importance', 'in', 'inherent', 'intel', 'intellinet', 'intercompany', 'interconnect', 'internationa', 'into', 'intraday', 'intrastate', 'invoice', 'invoiced', 'invoices', 'iomega', 'is', 'isc', 'it', 'j', 'jackie', 'jaquet', 'jason', 'jebel', 'jeffrey', 'jennifer', 'jim', 'joanie', 'joanne', 'josey', 'julie', 'july', 'karen', 'katherine', 'kathryn', 'katy', 'kcs', 'kelly', 'ken', 'kevin', 'kimberly', 'kinsey', 'know', 'koch', 'kristen', 'kyle', 'l', 'lamadrid', 'lamphier', 'lannou', 'lateral', 'lauri', 'lee', 'legislation', 'let', 'leth', 'lindley', 'lisa', 'listbot', 'liz', 'lloyd', 'locker', 'logistics', 'logos', 'lone', 'lonestar', 'lots', 'lp', 'lsk', 'lsp', 'luong', 'lyondell', 'm', 'mack', 'macromedia', 'march', 'marlene', 'marlin', 'marta', 'mary', 'materia', 'materially', 'may', 'mazowita', 'mccoy', 'mckay', 'mcmills', 'me', 'measurement', 'meds', 'meetings', 'megan', 'melba', 'melissa', 'memo', 'meredith', 'meter', 'meters', 'methanol', 'meyers', 'midcon', 'mike', 'mills', 'mining', 'mitchell', 'mm', 'mmbtu', 'mmbtus', 'month', 'moopid', 'mops', 'morris', 'msn', 'mtbe', 'mtr', 'muscle', 'mx', 'nbsp', 'neal', 'need', 'nelson', 'neon', 'netherlands', 'neuweiler', 'new', 'ngo', 'nguyen', 'nick', 'nom', 'nomad', 'nominated', 'nominates', 'nomination', 'nominations', 'nommensen', 'noms', 'not', 'oasis', 'october', 'oem', 'of', 'oi', 'olsen', 'on', 'oo', 'ooking', 'opinions', 'opm', 'original', 'origination', 'otcbb', 'out', 'outage', 'outages', 'packard', 'pager', 'pain', 'paliourg', 'panenergy', 'papayoti', 'paso', 'pat', 'path', 'pathed', 'pathing', 'paths', 'patti', 'payback', 'payroll', 'pcx', 'pec', 'pefs', 'pena', 'penis', 'pep', 'pg', 'pgev', 'pharmacy', 'phillip', 'photoshop', 'php', 'pill', 'pills', 'pinion', 'pinnacle', 'pipe', 'pipeline', 'pipelines', 'pipes', 'plant', 'please', 'pm', 'ponton', 'poorman', 'pops', 'prayer', 'predictions', 'prescription', 'pro', 'prod', 'production', 'projections', 'pubiisher', 'questions', 'ranch', 'randall', 'ray', 're', 'readers', 'rebecca', 'recipient', 'redeliveries', 'reflect', 'reflects', 'reinhardt', 'reliant', 'reliantenergy', 'resellers', 'resolve', 'responsibilities', 'resuits', 'rev', 'reveffo', 'reviewed', 'revised', 'revision', 'revisions', 'rick', 'riley', 'rita', 'rivers', 'rnd', 'robert', 'robotics', 'rodriguez', 'rolex', 'romeo', 'rx', 's', 'sabrae', 'sandi', 'sarco', 'sarmiento', 'saturday', 'scheduled', 'scheduler', 'schedulers', 'scheduling', 'schumack', 'se', 'seaman', 'see', 'sent', 'sept', 'settlement', 'settlements', 'sex', 'shawna', 'sheri', 'sherlyn', 'shipper', 'should', 'shut', 'sir', 'sitara', 'smith', 'so', 'sofftwaares', 'soft', 'soma', 'spain', 'spam', 'speckels', 'speculative', 'spot', 'spreadsheet', 'src', 'stacey', 'stack', 'statements', 'stella', 'stephanie', 'steve', 'strangers', 'studio', 'subject', 'subscribers', 'sunsail', 'superty', 'susan', 'suzanne', 'sweeney', 'swing', 'tailgate', 'talked', 'tammy', 'tap', 'targus', 'taylor', 'td', 'teco', 'tejas', 'tenaska', 'terminated', 'termination', 'tess', 'tessie', 'tetco', 'texas', 'texoma', 'th', 'thanks', 'that', 'the', 'there', 'these', 'they', 'this', 'thru', 'ticket', 'tickets', 'tirr', 'tisdale', 'to', 'todd', 'tom', 'tomorrow', 'tongue', 'topica', 'toshiba', 'tr', 'tracked', 'trader', 'transco', 'transport', 'trevino', 'troy', 'tu', 'tufco', 'txu', 'uae', 'unaccounted', 'uncertainties', 'undervalued', 'unify', 'up', 'ur', 'usb', 'utilities', 'valadez', 'valero', 'valign', 'valium', 'valley', 'vance', 'variance', 'vaughn', 'venturatos', 'verdana', 'vi', 'viagra', 'vicente', 'vicodin', 'victor', 'viewsonic', 'villarreal', 'vlt', 'voip', 'vols', 'volume', 'volumes', 'waha', 'walker', 'wallis', 'walters', 'was', 'waste', 'we', 'weight', 'weissman', 'wellhead', 'what', 'whiting', 'wi', 'width', 'wiil', 'will', 'winfree', 'wireless', 'with', 'withers', 'worksheet', 'would', 'wynne', 'xanax', 'xls', 'xp', 'yap', 'you', 'zeroed', 'zivley', 'zonedubai', '|']
def super_generate_feature_vector(text,freq):
    feature = []
    for word in wordlst:
        if any(c.isalpha() for c in word):
            if (len(word)>1):
                frequence = float(freq[word])
            else:
                pass
        else:
            frequence = text.count(word)
        feature.append(frequence)
    return feature
# This method generates a design matrix with a list of filenames
# Each file is a single training example
def generate_design_matrix(filenames):
    design_matrix = []
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            try:
                text = f.read() # Read in text from file
            except Exception as e:
                # skip files we have trouble reading.
                continue
            text = text.replace('\r\n', ' ') # Remove newline character
            words = re.findall(r'\w+', text)
            word_freq = defaultdict(int) # Frequency of all words
            for word in words:
                word_freq[word] += 1

            # Create a feature vector
            feature_vector = super_generate_feature_vector(text, word_freq)
            design_matrix.append(feature_vector)
    return design_matrix

# ************** Script starts here **************
# DO NOT MODIFY ANYTHING BELOW

spam_filenames = glob.glob(BASE_DIR + SPAM_DIR + '*.txt')
spam_design_matrix = generate_design_matrix(spam_filenames)
ham_filenames = glob.glob(BASE_DIR + HAM_DIR + '*.txt')
ham_design_matrix = generate_design_matrix(ham_filenames)
# Important: the test_filenames must be in numerical order as that is the
# order we will be evaluating your classifier
test_filenames = [BASE_DIR + TEST_DIR + str(x) + '.txt' for x in range(NUM_TEST_EXAMPLES)]
test_design_matrix = generate_design_matrix(test_filenames)

X = spam_design_matrix + ham_design_matrix
Y = np.array([1]*len(spam_design_matrix) + [0]*len(ham_design_matrix)).reshape((-1, 1)).squeeze()

np.savez(BASE_DIR + 'spam-data.npz', training_data=X, training_labels=Y, test_data=test_design_matrix)
