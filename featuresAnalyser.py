import os
from os import listdir
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import featuresAnalyser

base_dir = os.getcwd()

def extract_features(lang, filetype, proj=False):
    if proj == True:
        lang += "_proj"
    os.chdir(base_dir + '/data/')
    taille_phrases = []
    liste_mots = []
    with open(filetype + "_" + lang + ".conllu", 'r', encoding='utf8') as f:
        lines = f.readlines()
        empty_lines = []
        for i,line in enumerate(lines):
            if line == "\n":
                empty_lines.append(i)
        empty_lines = [-1] + empty_lines
        nbre_phrases = len(empty_lines)
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
    tailles_vocab, tailles_phrases, longueur_mots, tx_projectivite = [],[],[],[]
    for lang in list_lang:
        my_dict = extract_features(lang, filetype)
        tailles_vocab.append(len(my_dict["liste_mots"]))
        tailles_phrases.append(np.mean(my_dict["taille_phrases"]))
        longueur_mots.append(np.mean([len(x) for x in my_dict["liste_mots"]]))
        tx_projectivite.append(taux_projectvite(filetype,lang))
    df = pd.DataFrame({'lang':list_lang,
                       'taille_vocab' : tailles_vocab,
                       'tailles_phrase' : tailles_phrases,
                       'longueur_mots' : longueur_mots,
                       'taux_projectivite': tx_projectivite,
                       })
    return df

def make_df(filetype):
    df = var_exp_df(filetype)
    df.set_index('lang', inplace=True)
    df2 = pd.read_csv(base_dir + '/out.csv', index_col='lang')
    df = df.join(df2)
    return df

def taux_projectvite(filetype,lang):
    os.chdir(base_dir + '/data/')
    with open(filetype + "_" + lang + ".conllu", 'r', encoding='utf8') as f:
        lines = f.readlines()
        empty_lines = []
        for i, line in enumerate(lines):
            if line == "\n":
                empty_lines.append(i)
        empty_lines = [-1] + empty_lines
        nbre_phrases = len(empty_lines)
    os.chdir(base_dir + '/expe/out/')
    with open(filetype + "_" + lang + "_proj.conllu", 'r', encoding='utf8') as f:
        lines = f.readlines()
        empty_lines = []
        for i, line in enumerate(lines):
            if line == "\n":
                empty_lines.append(i)
        empty_lines = [-1] + empty_lines
        nbre_phrases_proj = len(empty_lines)
    return nbre_phrases_proj/nbre_phrases


def my_plot():
    df = pd.DataFrame({'mots' : liste_mots})
    df["longueur"]= df['mots'].str.len()

    sns.set(style="ticks")
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True,
                                        gridspec_kw={"height_ratios": (.15, .85)})
    sns.boxplot(df.longueur, ax=ax_box)
    sns.distplot(df.longueur, ax=ax_hist, kde=True)

    ax_box.set(yticks=[])
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)
    plt.show()

    print(df['longueur'].describe())

    print(stats.describe(taille_phrases))

    # sns.set_style('whitegrid')
    # ax = sns.histplot(data=df, x="longueur")
    # ax.set(xlabel='Longueur des mots', ylabel='Nombre')
    # ax.set_title(lang)
    # plt.show()

    #Calculer la profondeur des arbres
    #Calculer la longueur moyenne des mots de la phrase OK
    #POS

    #Stats sur le corpus !!
    #Taux de non projectivité
    #longueur moyenne de dépendances
    #Nature de la dépendance (OBJ)
    #Ordre de dépendance 0 direct, 1 POS, Structure syntaxique
