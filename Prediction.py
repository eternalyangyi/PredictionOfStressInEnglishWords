#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 09:39:13 2017

@author: yy
"""
import pandas as pd
import re
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pickle
import nltk
from sklearn.feature_extraction import DictVectorizer
v_p = ('AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW')
c_p = ('P', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'R', 'S', 'SH', 
       'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH')
vc_p = v_p + c_p
stress_indicater = '[0-2]'
primary_stress = '1'
second_stress = '2'
nomal_stress = '0'
P_V = '[A-Z]{2}[0-2]{1}'
strong_suffixes = ('AL','ANCE','ANCY','ANT','ARD','ARY','AUTO','ENCE','ENCY','ENT','ERY','EST','IAL','IAN','IANA','EN','IC',
                   'IFY','INE','ION','TION','ITY','IVE','ORY','OUS','UAL','URE','WIDE','Y','ADE','E','EE','EEN','EER','ESE',
                   'ESQUE','ETTE','EUR','IER','OON','QUE')
neutral_suffixes = ('ABLE','AGE','AL','ATE','AUTO','ED','EN','ER','EST','FUL','HOOD','IBLE','ING','ILE','ISH','ISM','IST',
                    'IZE','LESS','LIKE','LY','MAN','MENT','MOST','NESS','OID','S','SHIP','SOME','TH','WARD','WISE')

strong_prefixes = ('AD','CO','CON','COUNTER','DE','DI','DIS','E','EN','EX','IN','MID','OB','PARA','PRE','RE','SE','SUB')
neutral_prefixes = ('ANTI','BI','NON','PRO','TRI','DOWN','FORE','MIS','OVER','OUT','UN','UNDER','UP','CONTRA','COUNTER','DE',
                    'EXTRA','INTER','INTRO','MULTI','NON','POST','RETRO','SUPER','TRANS','ULTRA')
tagger = ('CC','CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 
          'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 
          'WRB')

pattern = {'10'  : 1,'01'  : 2,
           '100' : 1,'010' : 2,'001' : 3,
           '1000': 1,'0100': 2,'0010': 3,'0001': 4}
#Create column contain sequnces of phonemes.
def create_column(original_df,max_phonemes_num):
    row_num = len(original_df)
    L = [["None"] * row_num for i in range(max_phonemes_num)]
    df = pd.DataFrame(columns = list(str(i) for i in range(1,max_phonemes_num)))
    for i in range(row_num):
        phonemes = original_df.iloc[i]
        for j in range(len(phonemes)):
            phoneme = phonemes[j]
            v_phoneme = phoneme[:2]
            L[j][i] = v_phoneme
    index = 1
    for i in L:
        df[str(index)] = pd.Series(i).values
        index += 1
    return df
        
def split_words(data,modified_data):
    for line in data:
        line = line.split(":")
        modified_data.append((line[0],line[1].split(" ")))
    return modified_data
def get_Pstress_position(phonemes):
    temp = []
    for phoneme in phonemes:
        if primary_stress in phoneme:
            temp.append(1)
        if nomal_stress in phoneme:
            temp.append(0)
        if second_stress in phoneme:
            temp.append(0)
    return ''.join([str(position) for position in temp])
def get_position(phonemes):
    temp = pattern.get(phonemes)
    if temp == None:
        temp = 0
    return int(temp)
def get_rid_of_stress(phonemes):
    phonemes = ' '.join(phonemes)
    return tuple(re.sub(stress_indicater,'',phonemes).split())

def get_pattern(phonemes):
    temp = 0
    for phoneme in phonemes:
        if phoneme in v_p:
            temp += 1
            temp *= 10
        else:
            temp += 2
            temp *= 10
    temp /= 10
    return int(temp)

def count_l(word):
    return len(word)
def count_p(phonemes):
    return len(phonemes)
def count_v(phonemes):
    v_counter = 0
    for phoneme in phonemes:
        if phoneme in v_p:
            v_counter += 1
    return v_counter
def count_c(phonemes):
    c_counter = 0
    for phoneme in phonemes:
        if phoneme in c_p:
            c_counter += 1
    return c_counter
def adjacent_phoneme(phonemes):
    temp = []
    for i in range(len(phonemes)-1):
        temp.append((phonemes[i],phonemes[i+1]))
    return tuple(temp)
def find_position(primary_stress_list):
    for index in range(len(primary_stress_list)):
        if primary_stress_list[index] == 1:
            return index + 1
def find_pstress_v(phonemes):
    for phoneme in phonemes:
        if primary_stress in phoneme:
            pstress_v = re.sub(primary_stress,'',phoneme)
    for i in range(len(v_p)):
        if pstress_v == v_p[i]:
            return i

def check_prefix(word,prefixes_set):
    for letter_idx in range(len(word) + 1):
        if word[:letter_idx] in prefixes_set:
            return 1
    return 0


def check_suffix(word,suffixes_set):
    word_length = len(word)
    for letter_idx in range(word_length + 1):
        if word[(word_length - letter_idx):] in suffixes_set:
            return 1
    return 0


def get_pos_tag(word):
    return nltk.pos_tag([word])[0][1]
def transfer_tag(tag,index):
    if tag == tagger[index]:
        return 1
    return 0
def train(data, classifier_file):# do not change the heading of the function
    modified_data = []
    split_words(data,modified_data)
    words = pd.DataFrame(data=modified_data, columns=('Spell', 'Pronunc'))
    df = pd.DataFrame()
    Position = pd.DataFrame()
    final_df = pd.DataFrame()
    max_phonemes_num = 15
    words['NoStress'] = words.Pronunc.apply(get_rid_of_stress)
    Position = create_column(words['Pronunc'],max_phonemes_num)
    words['Primary_Stress'] = np.array(words.Pronunc.apply(get_Pstress_position))
    words['C'] = words.Primary_Stress.apply(get_position)
    words['V_counter'] = words.NoStress.apply(count_v)
    df['pattern'] = words.NoStress.apply(get_pattern)
    df['Num_of_letters'] = words.Spell.apply(count_l)
    df['P_counter'] = words.NoStress.apply(count_p)
    df['V_counter'] = words.NoStress.apply(count_v)
    df['C_counter'] = words.NoStress.apply(count_c)
    df['strong_suf'] = words.Spell.apply(check_suffix,args=(strong_suffixes,))
    df['neutral_suf'] = words.Spell.apply(check_suffix,args=(neutral_suffixes,))
    df['strong_pre'] = words.Spell.apply(check_prefix,args=(strong_prefixes,))
    df['neutral_pre'] = words.Spell.apply(check_prefix,args=(neutral_prefixes,))
    words['type_tag'] = words.Spell.apply(get_pos_tag)
    #Create all combination of phonemes sequences
    for i in range(len(tagger)):
        df[tagger[i]] = words.type_tag.apply(transfer_tag,args = (i,))
    final_df = pd.concat([df,Position],axis = 1)
    fake_list = [['None'] * (len(vc_p)+1) for i in range(max_phonemes_num)]
    fake_df = pd.DataFrame(columns = list(str(i) for i in range(1,max_phonemes_num)))
    for i in range(len(vc_p)):
        for j in range(max_phonemes_num):
            fake_list[j][i] = vc_p[i]
    index = 1
    for i in fake_list:
        fake_df[str(index)] = pd.Series(i).values
        index += 1
    f_df = final_df.append(fake_df,ignore_index=True)
    dv = DictVectorizer(sparse=False)
    x = dv.fit_transform(f_df.to_dict(orient = 'records'))
    X = x[:-40]
    #delete fake combination
    y = words['C']
    clf = DecisionTreeClassifier(max_depth = 19)
    clf.fit(X,y)
    pickle.dump(clf,open(classifier_file,"wb"))
################# testing #################

def test(data, classifier_file):# do not change the heading of the function
    modified_data = []
    split_words(data,modified_data)
    words = pd.DataFrame(data=modified_data, columns=('Spell', 'Pronunc'))
    df = pd.DataFrame()
    Position = pd.DataFrame()
    final_df = pd.DataFrame()
    max_phonemes_num = 15
    words['NoStress'] = words.Pronunc.apply(get_rid_of_stress)
    Position = create_column(words['Pronunc'],max_phonemes_num)
    df['pattern'] = words.NoStress.apply(get_pattern)
    df['Num_of_letters'] = words.Spell.apply(count_l)
    df['P_counter'] = words.NoStress.apply(count_p)
    df['V_counter'] = words.NoStress.apply(count_v)
    df['C_counter'] = words.NoStress.apply(count_c)
    df['strong_suf'] = words.Spell.apply(check_suffix,args=(strong_suffixes,))
    df['neutral_suf'] = words.Spell.apply(check_suffix,args=(neutral_suffixes,))
    df['strong_pre'] = words.Spell.apply(check_prefix,args=(strong_prefixes,))
    df['neutral_pre'] = words.Spell.apply(check_prefix,args=(neutral_prefixes,))
    words['type_tag'] = words.Spell.apply(get_pos_tag)
    #Create all combination of phonemes sequences
    for i in range(len(tagger)):
        df[tagger[i]] = words.type_tag.apply(transfer_tag,args = (i,))
    final_df = pd.concat([df,Position],axis = 1)
    fake_list = [['None'] * (len(vc_p)+1) for i in range(max_phonemes_num)]
    fake_df = pd.DataFrame(columns = list(str(i) for i in range(1,max_phonemes_num)))
    for i in range(len(vc_p)):
        for j in range(max_phonemes_num):
            fake_list[j][i] = vc_p[i]
    index = 1
    for i in fake_list:
        fake_df[str(index)] = pd.Series(i).values
        index += 1
    f_df = final_df.append(fake_df,ignore_index=True)
    dv = DictVectorizer(sparse=False)
    x = dv.fit_transform(f_df.to_dict(orient = 'records'))
    #delete fake combination
    X = x[:-40]
    clf = pickle.load(open(classifier_file,"rb")) 
    return clf.predict(X).tolist()
