import codecs
import os
from collections import Counter

def rewrite():
    en=[]
    de=[]
    with open('news-commentary-v16.de-en.tsv','r', encoding='utf8') as f:
        for line in f:
            d,e=line.split('\t')
            en.append(e)
            de.append(d)
    with open('news-v16.de','w', encoding='utf8') as f:
        for w in de:
            f.write('{}\n'.format(w))
    with open('news-v16.en','w', encoding='utf8') as f:
        for w in en:
            f.write('{}\n'.format(w))
            
def vocab(ftext,fvocab):
    text = codecs.open(ftext, 'r', 'utf-8').read()
    words = text.split()
    word2cnt = Counter(words)
    if not os.path.exists('preprocessed'): os.mkdir('preprocessed')
    with codecs.open('preprocessed/{}'.format(fvocab), 'w', 'utf-8') as fout:
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))
            
if __name__ == '__main__':
    rewrite()
    print('Separate languages successfully.')
    vocab('news-v16.de','de.vocab.tsv')
    vocab('news-v16.en','en.vocab.tsv')
    print('Build vocabulary successfully')