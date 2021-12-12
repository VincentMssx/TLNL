import os
from os import listdir
import pandas as pd
import numpy as np


base_dir = os.getcwd()

def extract_features(lang, filetype, proj=False):
    if proj == True:
        lang += "_proj"
    os.chdir(base_dir + '/data/')
    taille_phrases = []
    liste_mots = []
    with open(filetype + "_" + lang + ".conllu", 'r', encoding='utf8') as f:
        lines = f.readlines()
        empty_lines = [-1]
        for i,line in enumerate(lines):
            if line == "\n":
                empty_lines.append(i)
        for indice in range(len(empty_lines)-1):
            phrase = lines[empty_lines[indice]+1:empty_lines[indice+1]-1]
            taille_phrase = 0
            for word in phrase:
                if "PUNCT" not in word: #On enlève la ponctuation pour compter
                    taille_phrase += 1
                    if "PROPN" not in word and "NUM" not in word: #On enlève les nombres et les noms propres pour la diversité du vocab
                        word = word.split()
                        if word[2] != "_":
                            liste_mots.append(word[2])
                        else :
                            liste_mots.append(word[1])
            taille_phrases.append(taille_phrase)
        liste_mots = list(set(liste_mots))
        return {"liste_mots" : liste_mots, "taille_phrases":taille_phrases}

def var_exp_df(filetype):
    os.chdir(base_dir)
    list_lang = [k.replace('.conllu', '').replace('train_', '') for k in listdir('data') if
                 '.conllu' in k and 'train' in k]
    tailles_vocab, tailles_phrases, longueur_mots, tx_non_projectivite, nombre_moyen_dep, longueur_moyenne_dep, longueur_moyenne_max_dep  = [],[],[],[],[],[],[]
    for lang in list_lang:
        print('Analysing lang : ',lang)
        my_dict = extract_features(lang, filetype)
        tailles_vocab.append(len(my_dict["liste_mots"]))
        tailles_phrases.append(np.mean(my_dict["taille_phrases"]))
        longueur_mots.append(np.mean([len(x) for x in my_dict["liste_mots"]]))
        tx_non_projectivite.append(1-taux_projectvite(filetype,lang))
        my_dict_dep = long_dependances(filetype,lang)
        nombre_moyen_dep.append(my_dict_dep['nombre_moyen_dep'])
        longueur_moyenne_dep.append(my_dict_dep['longueur_moyenne_dep'])
        longueur_moyenne_max_dep.append(my_dict_dep['longueur_moyenne_max_dep'])
    df = pd.DataFrame({'lang':list_lang,
                       'taille_vocab' : tailles_vocab,
                       'tailles_phrase' : np.round(tailles_phrases,2),
                       'longueur_mots' : np.round(longueur_mots,2),
                       'taux_non_projectivite': np.round(tx_non_projectivite,2),
                       'nombre_moyen_dep':  np.round(nombre_moyen_dep,2),
                       'longueur_moyenne_dep': np.round(longueur_moyenne_dep,2),
                       'longueur_moyenne_max_dep':  np.round(longueur_moyenne_max_dep,2),
                       })
    return df

def make_df(filetype):
    df = var_exp_df(filetype)
    df.set_index('lang', inplace=True)
    df2 = pd.read_csv(base_dir + '/out.csv', index_col='lang')
    df = df.join(df2)
    df2 = pd.read_csv(base_dir + '/complexityScore.csv', index_col='Code')
    df2.drop(['Language', 'Extrapolation'], axis = 1, inplace=True)
    df = df.join(df2)
    os.chdir(base_dir)
    df.to_csv('results_'+filetype+'.csv')
    return df

def taux_projectvite(filetype,lang):
    os.chdir(base_dir + '/data/')
    with open(filetype + "_" + lang + ".conllu", 'r', encoding='utf8') as f:
        lines = f.readlines()
        empty_lines = [-1]
        for i, line in enumerate(lines):
            if line == "\n":
                empty_lines.append(i)
        nbre_phrases = len(empty_lines)
    os.chdir(base_dir + '/expe/out/')
    with open(filetype + "_" + lang + "_proj.conllu", 'r', encoding='utf8') as f:
        lines = f.readlines()
        empty_lines = [-1]
        for i, line in enumerate(lines):
            if line == "\n":
                empty_lines.append(i)
        nbre_phrases_proj = len(empty_lines)
    return nbre_phrases_proj/nbre_phrases

def long_dependances(filetype,lang):
    os.chdir(base_dir + '/expe/out/')
    long, nbre, longmax = [],[],[]
    with open(filetype + "_" + lang + "_pgle.mcf", 'r', encoding='utf8') as f:
        ## construction des indices de phrases ##
        lines = f.readlines()
        sentences = [0]
        for i,line in enumerate(lines):
            line = line.split("\t")
            if int(line[4]) == 1:
                sentences.append(i+1)
        liste = []
        for i in range(len(sentences) - 1):
            liste2 = []
            for line in lines[sentences[i]:sentences[i + 1]]:
                word = line.split("\t")
                if word[3] == 'root':
                    liste2.append(0)
                else :
                    liste2.append(int(word[2]))
            liste.append(liste2)
    liste_abs = rel2abs(liste)
    for liste3 in liste_abs:
        arbre = maketree(liste3)
        nbre.append(len(arbre))
        longueurs = [len(dep) for dep in arbre]
        long.append(np.mean(longueurs))
        longmax.append(np.max(longueurs))
    return {"nombre_moyen_dep" : np.mean(nbre), "longueur_moyenne_dep":np.mean(long), "longueur_moyenne_max_dep" : np.mean(longmax)}

def rel2abs(liste):
    rep = []
    for liste1 in liste:
        rep.append([i+j if j != 0 else -1 for i,j in enumerate(liste1)])
    return rep

def maketree(liste):
    arbre = [[liste.index(-1)]]
    profondeur = 1
    while True:
        nouvel_arbre = []
        liste_recherche = [j[-1] for j in arbre if len(j)==profondeur]
        if liste_recherche == []:
            return arbre
        for l in arbre:
            if l[-1] in liste_recherche and l[-1] in liste:
                indices = [i for i, x in enumerate(liste) if x == l[-1]]
                for k in indices:
                    nouvel_arbre.append(l+[k])
            else:
                nouvel_arbre.append(l)
        arbre = nouvel_arbre
        profondeur += 1