import os
from re import match
from string import lower
import pickle
from time import time, sleep
from sys import argv

home = '/ifs/tmp/'
present = os.listdir(home)

ngrams = argv[1]
lower = argv[2]
upper = argv[3]

# http://storage.googleapis.com/books/ngrams/books/googlebooks-eng-fiction-all-2gram-20090715-99.csv.zip

def download(url):
	"""Copy the contents of a file from a given URL
	to a local file.
	"""
	import urllib     
	webFile = urllib.urlopen(url)
	localFile = open(home+url.split('/')[-1], 'w')
	localFile.write(webFile.read())
	webFile.close()
	localFile.close()

url = 'http://storage.googleapis.com/books/ngrams/books/'
toget = [ str(i) for i in range(int(lower),int(upper)) ]

for fileno in toget:
    present = os.listdir(home)
    filename = 'googlebooks-eng-all-'+ngrams+'gram-20090715-' + fileno + '.csv.zip'
    if filename[:-4] not in present and filename not in present:
        print filename
        download(url+filename)
    sleep(10)
    com = 'unzip ' + home + filename + ' -d ' + home
    os.system(com)
