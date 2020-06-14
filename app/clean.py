import pandas as pd
import spacy
import gensim

from gensim.parsing.preprocessing import strip_short
from symspellpy import SymSpell, Verbosity
import pkg_resources
import re

nlp = spacy.load('en_core_web_sm')
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

stops = list(
    """
a about above across afterwards again along
already also although am among amongst amount an and another any anyhow
anyone anything anyway anywhere are around as at
be became because become becomes becoming been before beforehand behind
being below beside besides between beyond both bottom but by
can cannot ca could
did do does doing done down due
each eight either eleven else elsewhere empty enough even ever every
everyone everything everywhere except
fifteen fifty first five for former formerly forty four from front full
further
if in is it its itself
latterly
namely neither never next nine no none noone nor not
nothing now nowhere
of off often on once one only onto or other others otherwise our ours ourselves
out over own
rather re really regarding
same say see seem seemed seeming serious several should show side
since six sixty so some something sometime sometimes somewhere
ten than that the their then there thereafter
thereby therefore therein thereupon these they third this those though three
thru thus to twelve twenty
two
various via what whatever when whence whenever where
whereafter whereas whereby wherein whereupon wherever whether which while
whither who whoever whole whom whose why will with within without would
""".split()
) + ['tbsp', 'like', 'PRON', 'unlink', 'urllink', 'link', 'tbsp', 'get', 'blog', 'blogs', 'one', 'get', 'know', 'really', 'time', 'well', 'think', 'got', 'would', 'going', 'day']
# df['tokens'] = df['text'].map(lambda x: nlp.tokenizer(x.lower()))

def clean_jv(doc):
   typo_free = ' '.join([(sym_spell.lookup(i, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)[0].term) for i in doc])
   twol_free = strip_short(typo_free)
   return twol_free
def clean(lst):
    lst = nlp.tokenizer(lst.lower())
#     lst = [token for token in lst if not token.is_stop]
    lst = [token.lemma_ for token in lst if str(token.lemma_) not in stops]
    lst = [re.sub(r'[\W\d\s]', '', string) for string in lst]
    while '' in lst:
        lst.remove('')
    lst = clean_jv(lst)
    lst = [token for token in lst.split() if token not in stops]
    return lst
