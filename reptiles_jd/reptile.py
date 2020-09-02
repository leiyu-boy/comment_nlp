# -*- coding: utf-8 -*-
# @Time     : 2020/8/19 4:48 下午
# @Author   : yu.lei
import hashlib
import json
import sys
import time

import requests
from lxml import etree
import pandas as pd

from reptiles_jd import CURRENT_PATH
from setting import ORDERID, SECRET

headers = {
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36'
}


# 搜所商品，取评论数目top15的商品id
def wares_search(keyword):
    params = {
        'keyword': keyword,
        'enc': 'utf-8',
        'wq': '鞋子',
        'psort': 4
    }
    url = 'https://search.jd.com/Search'
    resp = requests.get(url, params=params, headers=headers)
    html = etree.HTML(resp.text)
    url_list = html.xpath("//li[@class='gl-item']/div[@class='gl-i-wrap']/div[@class='p-img']/a/@href")[:15]
    for i in url_list:
        print(keyword, i)
        id = i.replace('//item.jd.com/', '').replace('.html', '')
        get_comment_url(id)


# 用的是流冠代理，可以免费试用
def get_proxy():
    _version = sys.version_info
    is_python3 = (_version[0] == 3)
    orderId = ORDERID
    secret = SECRET
    host = "flow.hailiangip.com"
    port = "14223"
    user = "proxy"
    timestamp = str(int(time.time()))  # 计算时间戳
    txt = "orderId=" + orderId + "&" + "secret=" + secret + "&" + "time=" + timestamp
    if is_python3:
        txt = txt.encode()
    sign = hashlib.md5(txt).hexdigest()  # 计算sign
    password = 'orderId=' + orderId + '&time=' + timestamp + '&sign=' + sign + "&pid=-1" + "&cid=-1" + "&uid=" + "&sip=0" + "&nd=1"
    proxyUrl = "http://" + user + ":" + password + "@" + host + ":" + port
    proxy = {"http": proxyUrl, "https": proxyUrl}
    return proxy


def get_comment_url(id):
    # 3表示好评，1表示差评
    for comment_type in [3, 1]:
        # 分别爬取10页，即100条评论
        for page in range(0, 10):
            url = 'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId={}&score={}&sortType=5&page={}&pageSize=10&isShadowSku=0&fold=1'.format(
                id, comment_type, page)
            get_comment_data(comment_type, url)


def get_comment_data(comment_type, url):
    data_list = []
    try:
        resp = requests.get(url, headers=headers, proxies=get_proxy())
    except Exception as e:
        print(e)
        get_comment_data(comment_type, url)
        return
    text = resp.text
    try:
        text = text.replace('fetchJSON_comment98(', '')
        text = text[:-2]
    except Exception as e:
        print(e)
        get_comment_data(comment_type, url)
        return
    try:
        json_data = json.loads(text)
    except:
        return
    comments = json_data['comments']
    for i in comments:
        data_list.append(i['content'])
    data = pd.DataFrame(data_list, columns=['comment'])
    if comment_type == 3:
        data.to_csv('{}/../Data/great_comment.csv'.format(CURRENT_PATH), index=False, encoding='utf-8', mode='a',
                    header=False)
    else:
        data.to_csv('{}/../Data/bad_comment.csv'.format(CURRENT_PATH), index=False, encoding='utf-8', mode='a',
                    header=False)
    print('success')
