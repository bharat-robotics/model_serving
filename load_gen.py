import argparse
import random
import requests
import pandas as pd
import imghdr
import os

def get_urls(req):
    # assume the file can fit in memory this is a bad idea
    df = pd.read_csv('fall11_urls.txt', sep='\t', index_col=False, encoding='latin_1', error_bad_lines=False) 
    url_list = []
    os.system('mkdir -p ./image')
    while len(url_list) < req:
        randnum = random.randint(0,df.shape[0])
        url_id = df.iloc[randnum,0]
        url = df.iloc[randnum,1]
        try:
            r = requests.get(url, allow_redirects=True)
            open('./image/'+url_id, 'wb').write(r.content)
            if imghdr.what('./image/'+url_id) != None:
                url_list.append(url_id)
            else:
                os.system('rm ./image/{}'.format(url_id))
            print('url: {}\nurl_id: {}\n'.format(url, url_id))
        except:
            print('url: {}\nurl_id: {}\n'.format(url, url_id))
            print('unable to retrieve image')
        with open('./url_list', 'w') as out_file:
            for line in url_list:
                out_file.write(line)
            out_file.close()
    return(url_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--requests', type=int, default=10)
    parser.add_argument('--file', type=str, default='none')
    
    args = parser.parse_args()
    if args.file == 'none':
        url_list = get_urls(args.requests)
    '''
    # add support for file input later
    else:
        with open('url_links.txt') as tsv:
            url_list = []
            reader = csv.reader(tsv, delimiter='\t')
            for row in reader:
                url_list.append(row)

        download(url_list)
    '''
