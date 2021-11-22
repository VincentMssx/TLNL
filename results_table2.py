import os
from os import listdir
from subprocess import run
import pandas as pd
from tabulate import tabulate

base_dir = '/Users/victorlebellego/Documents/Dev/Python/TLNL/'
### A Changer ###
os.chdir(base_dir)
###

list_lang = [k.replace('.conllu','').replace('train_','') for k in listdir('data') if '.conllu' in k and 'train' in k]
print('Training '+str(len(list_lang))+' languages')

os.chdir('expe/')
i=1
nbre = len(list_lang)
for lang in list_lang:
    try :
        print('Testing '+lang+' '+str(i)+'/'+str(nbre))
        command = 'make lang=' + lang
        run(command.split(), capture_output=True)
        print('Success !')
        i += 1
    except :
        print("Erreur " + lang)
        pass

print('Reading results...')
os.chdir('out/')
df = pd.DataFrame(columns=["lang", "las", "uas"])
for lang in list_lang:
    try:
        with open(lang+'.res','r') as f:
            lines = f.readlines()
            to_append = lines[-1].split()
            df_length = len(df)
            df.loc[df_length] = to_append
    except :
        print("Erreur " + lang)
        pass

df.set_index('lang', inplace=True)
df.to_csv('out.csv')
print(tabulate(df, headers='keys', tablefmt='psql'))