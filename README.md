# clients_from_hell


use a numpy where to replace the null values with zeroes frequency

scale the features for some of the classifiers

labels = ['Dunces', 'Criminals', 'Deadbeats', 'Racists', 'Homophobes', 'Sexist', 'Frenemies', 'Cryptic', 'Ingrates', 'Chaotic Good', ]
labels = [
    'Dunces',
    'Criminals',
    'Deadbeats',
    'Racists',
    'Homophobes',
    'Sexist',
    'Frenemies',
    'Cryptic',
    'Ingrates',
    'Chaotic Good',
    ]

corpi = [
    corpus_Dunces, 
    corpus_Criminals, 
    corpus_Deadbeats, 
    corpus_Racists, 
    corpus_Homophobes, 
    corpus_Sexist, 
    corpus_Frenemies, 
    corpus_Cryptic, 
    corpus_Ingrates, 
    corpus_ChaoticGood,
] 

'Dunces' : 'ignoramus',
'Criminals' : 'freelance_felon',
'Deadbeats' : 'whatever this is', 



max_pages = 14

for page_number in range(max_pages + 1):
    dynamic_url = f'https://clientsfromhell.net/tag/freelance-felon/page/{page_number}'