#!/usr/bin/env python

import json
import pandas as pd
import re
import spacy
import numpy as np

from spacy.matcher import PhraseMatcher
from typing import List, Dict

from lexicalrichness import LexicalRichness
import textstat


# smart single quote
single_quote_reg1 = b'\xe2\x80\x99'.decode('utf-8') # single smart quote
single_quote_reg2 = b'\xe2\x80\x98'.decode('utf-8') # single smart quote

#smart double quotes
left_smart_quote = b'\xe2\x80\x9c'.decode('utf-8')
right_smart_quote = b'\xe2\x80\x9d'.decode('utf-8')

unmatched_lq_reg = r'“[^”]+?(?=[“])'

lpat = re.compile(left_smart_quote)
rpat = re.compile(right_smart_quote)
# quote_pattern = re.compile(f'({left_smart_quote}.+?{right_smart_quote})')
quote_pattern = re.compile(f'({left_smart_quote}[^{left_smart_quote}].+?{right_smart_quote})')

chapter_reg = '([cC]hapter\s+?(7|9|11|12|13|15))'

two_part_date_reg = '((?<=[^\d]))(\d{1,2}/\d{2,4})((?=[^\d]))'
three_part_date_reg = '((?<=[^\d]))(\d{1,2}/\d{1,2}/\d{2,4})((?=[^\d]))'

def print_full(x):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')



class Something:
    pass


def count_left_quotes(x):
    matches = re.findall(lpat, x)
    return len(matches)


def count_right_quotes(x):
    matches = re.findall(rpat, x)
    return len(matches)

def count_quotes(x):
    left_count = count_left_quotes(x)
    right_count = count_right_quotes(x)

    return left_count, right_count


def remove_quotes(x):
    lc, rc = count_quotes(x)

    if lc == rc:
        matches = re.findall(quote_pattern, x)

        for match in matches:
            x = x.replace(match, ' ')

    return x


def remove_temp_quotes(x):
    pat = '"quote"'

    x = re.sub(pat, ' ', x)

    return x


def replace_quotes(x, show_only: bool = False):
    left_count = count_left_quotes(x)
    right_count = count_right_quotes(x)

    if left_count == right_count:
        matches = re.findall(quote_pattern, x)

        for match in matches:
            if show_only:
                print(f'MATCH: {match}')
            else:
                x = x.replace(match, 'QUOTED_TEXT')

    return x


def replace_dates(x, show_only: bool = False):

    x = re.sub(three_part_date_reg, r'\1 DATE_TOKEN \3', x)
    x = re.sub(two_part_date_reg, r'\1 DATE_TOKEN \3', x)

    return x


def chapter_temp(x):

    subs = {'Chapter 7': 'ChapterSEVEN',
            'chapter 7': 'chapterSEVEN',
            'Chapter 9': 'chapterNINE',
            'chapter 9': 'chapterNINE',
            'Chapter 11': 'chapterELEVEN',
            'chapter 11': 'chapterELEVEN',
            'Chapter 12': 'chapterTWELVE',
            'chapter 12': 'chapterTWELVE',
            'Chapter 13': 'chapterTHIRTEEN',
            'chapter 13': 'chapterTHIRTEEN',
            'Chapter 15': 'chapterFIFTEEN',
            'chapter 15': 'chapterFIFTEEN'
            }

    matches = re.findall(chapter_reg, x)

    for match in matches:
        x = x.replace(match[0], subs[match[0]])

    return x


def chapter_replace(x):

    subs = {'Chapter 7': 'ChapterSEVEN',
            'chapter 7': 'chapterSEVEN',
            'Chapter 9': 'chapterNINE',
            'chapter 9': 'chapterNINE',
            'Chapter 11': 'chapterELEVEN',
            'chapter 11': 'chapterELEVEN',
            'Chapter 12': 'chapterTWELVE',
            'chapter 12': 'chapterTWELVE',
            'Chapter 13': 'chapterTHIRTEEN',
            'chapter 13': 'chapterTHIRTEEN',
            'Chapter 15': 'chapterFIFTEEN',
            'chapter 15': 'chapterFIFTEEN'
            }

    for key, value in subs.items():
        x = re.sub(value, key, x)

    return x


def fix_unmatched_lq(txt: str, match_list):

    for match in match_list:
        # print(f'MATCH: {match}')
        count = txt.count(match)
        if count == 1:
            new_txt = f'{match[:-1]}” '
            txt = txt.replace(match, new_txt)

    return txt

def fix_unmatched_rq(txt: str, match_list):
    for match in match_list:
        # print(f'MATCH: {match}')
        count = txt.count(match)
        if count == 1:
            new_txt = f'{match[:-1]}” '
            txt = txt.replace(match, new_txt)

    return txt


def get_sentence_data(txt, nlp):
    active_count: int = 0
    passive_count: int = 0
    sentence_count: int = 0
    sentence_list = list()

    doc = nlp(txt)

    for sent in doc.sents:
        has_verb = False
        has_active = False
        has_passive = False
        for token in sent:
            if token.pos_ == 'VERB':
                has_verb = True
                if 'pass' in token.dep_:
                    has_passive = True

        if has_verb:
            sentence_count += 1
            sentence_list.append(sent.text)
            if has_passive:
                passive_count += 1
            else:
                active_count += 1

    return sentence_list, sentence_count, active_count, passive_count


def display_sentence_tags(sentence, nlp, is_doc: bool = False):
    f_t = 'TOKEN'
    f_p = 'POS'
    f_d = 'DEP'
    print(f'{f_t:{15}}{f_p:{10}}{f_d:{10}}')

    if not is_doc:
        sentence = nlp(sentence)

    for token in sentence:
        print(f'{token.text:{15}}{token.pos_:{10}}{token.dep_:{10}}')

def get_sentence_list(txt: str, nlp) -> List:

    sentence_list: List = list()
    has_verb: bool = False

    doc = nlp(txt)

    for sent in doc.sents:
        has_verb = False
        for token in sent:
            if token.pos_ in ('AUX', 'VERB'):
                has_verb = True
        if has_verb:
            sentence_list.append(sent.text)

    return sentence_list


def sentences_to_string(txt: str, nlp) -> str:
    sentence_list: List = list()
    has_verb: bool = False

    doc = nlp(txt)

    for sent in doc.sents:
        has_verb = False
        for token in sent:
            if token.pos_ in ('AUX', 'VERB'):
                has_verb = True
        if has_verb:
            sentence_list.append(sent.text)

    return ' '.join(sentence_list)


def get_token_count(txt):
    txt = re.sub(r'\s+', ' ', txt)
    return len(txt.split(' '))


def create_matcher(nlp, json_path):

    with open(json_path, 'r') as json_file:
        entities = json.load(json_file)

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(ent) for ent in entities]
    matcher.add('function', patterns)

    return matcher


def feature_engineer(df, nlp, ent_matcher, fw_matcher):
    entity_count_list = list()
    entity_string_list = list()

    function_word_count_list = list()
    function_word_string_list = list()
    sentence_counts_list = list()
    pct_active_list = list()
    pct_passive_list = list()
    avg_qps_list = list()
    quote_count_list = list()
    avg_length_list = list()
    length_std_list = list()
    unique_terms_list = list()
    mtld_list = list()
    hdd_list = list()
    dale_chall_list = list()
    sw_count_list = list()
    sw_list = list()

    for idx, row in df.iterrows():
        ent_match_set = set()
        ent_match_count = 0
        fw_match_count = 0
        fw_match_list = list()
        row_sentence_counts = list()
        quote_count = 0
        token_count = 0

        txt = row['text']
        sentence_list, sentence_count, num_active, num_passive = get_sentence_data(txt, nlp)
        sentence_counts_list.append(sentence_count)

        for sentence in sentence_list:
            if 'QUOTED_TEXT' in sentence:
                quote_count += 1

            token_count = get_token_count(sentence)
            row_sentence_counts.append(token_count)

        quote_count_list.append(quote_count)
        avg_qps_list.append(round((quote_count / sentence_count) * 100, 2))

        sl_mean = np.mean(row_sentence_counts)
        sl_dev = np.std(row_sentence_counts)
        avg_length_list.append(round(sl_mean, 2))
        length_std_list.append(round(sl_dev, 2))

        pct_active_list.append(round(num_active / sentence_count, 2))
        pct_passive_list.append(round(num_passive / sentence_count, 2))

        doc = nlp(txt)

        stop_words = [token.text for token in doc if token.is_stop]
        sw_list.append(stop_words)
        sw_count_list.append(len(stop_words))

        # entity matches
        ent_matches = ent_matcher(doc)
        if len(ent_matches) > 0:
            ent_match_count = len(ent_matches)
            entity_count_list.append(ent_match_count)
            for match_id, start, end in ent_matches:
                span = doc[start:end]
                ent_match_set.add(span.text)
            ent_match_string = ','.join(list(ent_match_set))
            entity_string_list.append(ent_match_string)
        else:
            entity_count_list.append(0)
            # entity_ratio_list.append(0)
            entity_string_list.append('')

        # function_word matcher
        fw_matches = fw_matcher(doc)
        if len(fw_matches) > 0:
            fw_match_count = len(fw_matches)
            function_word_count_list.append(fw_match_count)
            for match_id, start, end in fw_matches:
                span = doc[start:end]
                fw_match_list.append(span.text)
            fw_match_string = ','.join(list(fw_match_list))
            function_word_string_list.append(fw_match_string)
        else:
            function_word_count_list.append(0)
            function_word_string_list.append('')

        # lexical richness features
        lex = LexicalRichness(txt)

        unique_terms_list.append(lex.terms)
        mtld_list.append(round(lex.mtld(threshold=0.72), 2))
        hdd_list.append(round(lex.hdd(draws=42), 2))

        # readability
        dale_chall_list.append(textstat.dale_chall_readability_score(txt))

    df['sentence_count'] = sentence_counts_list
    df['sentence_length_mean'] = avg_length_list
    df['sentence_length_std'] = length_std_list
    df['unique_terms'] = unique_terms_list
    df['sw_pct'] = round((sw_count_list / df['token_count']) * 100, 2)
    df['fw_count'] = function_word_count_list
    df['function_words'] = function_word_string_list
    df['quote_count'] = quote_count_list
    df['avg_qps'] = avg_qps_list
    df['entity_count'] = entity_count_list
    df['entity_ratio'] = round((df['entity_count'] / df['token_count']) * 100, 2)
    df['entity_list'] = entity_string_list
    df['pct_active_voice'] = pct_active_list
    df['pct_passive_voice'] = pct_passive_list
    df['mtld'] = mtld_list
    df['hdd'] = hdd_list
    df['dale_chall'] = dale_chall_list

    return df


def get_average_word_length(txt, nlp):
    num_tokens = 0
    char_count = 0
    doc = nlp(txt)

    for token in doc:
        if token.text in ('QUOTED_TEXT', 'TOKEN', "'s"):
            continue
        if token.pos_ in ('PUNCT', 'SPACE'):
            continue
        num_tokens += 1
        char_count += len(token.text)

    return round(char_count / num_tokens, 3)

