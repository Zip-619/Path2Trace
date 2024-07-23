#!/usr/bin/env python
# coding: utf-8

# In[165]:


import sys

import pandas as pd

sys.path.append('../data/pengpai/')
sys.path.append('../')
from link_prediction.utils import load_file
from urllib.request import urlopen, quote
from tqdm import tqdm
import json

amap_ak = '' # amap tooken
def query_poi_type(poi, city):
    url = "https://restapi.amap.com/v3/place/text?"
    query_poi = quote(poi)
    query_city = quote(city)
    url = url + "key=" + amap_ak + "&keywords=" + query_poi + "&city=" + query_city
    # print(url)
    req = urlopen(url, timeout=10)
    res = req.read().decode()
    query_res = json.loads(res)
    if query_res['status'] == '1' and int(query_res['count']) > 0:
        return query_res['pois'][0]['type']
    else:
        return None

if __name__=="__main__":

    case_paths = load_file('../data/pengpai/labeled_data/case_path.pickle')
    addr2type = pd.DataFrame(columns=['case_id','full','poi','village','township','county','city','province','type'])
    count = 0
    for case_id, path in tqdm(case_paths.items()):
        count += 1
        for addr in path:
            full = addr.get_addr('full')
            poi = addr.get_addr('poi')
            village = addr.get_addr('village')
            township = addr.get_addr('township')
            county = addr.get_addr('county')
            city = addr.get_addr('city')
            province = addr.get_addr('province')
            type = query_poi_type(poi,city)
            addr2type = addr2type.append({'case_id':case_id,'full':full,'poi':poi,'village':village,'township':township,'county':county,'city':city,'province':province,'type':type},ignore_index= True)
        if count%100 == 0:
            addr2type.to_csv('data/addr2type.csv',index=False)
    addr2type.to_csv('data/addr2type.csv',index=False)

