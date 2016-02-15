#!/usr/bin/env python

import requests
import time


def downloadimg_login():
    pic_file = int(time.time())
    pic_url = "https://kyfw.12306.cn/otn/passcodeNew/getPassCodeNew?module=login&rand=sjrand"
    print '[+] Download Picture: {}'.format(pic_url)
    try:
        resp = requests.get(pic_url, verify=False, timeout=5)
    except:
        resp = requests.get(pic_url, verify=False, timeout=3)
    with open("./login/%s.jpg"%pic_file, 'wb') as fp:
        fp.write(resp.content)
    return pic_file

def downloadimg_passenger():
    pic_file = int(time.time())
    pic_url = 'https://kyfw.12306.cn/otn/passcodeNew/getPassCodeNew?module=passenger&rand=randp'
    print '[+] Download Picture: {}'.format(pic_url)
    try:
        resp = requests.get(pic_url, verify=False, timeout=5)
    except:
        resp = requests.get(pic_url, verify=False, timeout=3)
    with open("./passenger/%s.jpg"%pic_file, 'wb') as fp:
        fp.write(resp.content)
    return pic_file

requests.packages.urllib3.disable_warnings()

for i in range(1000):
    downloadimg_login()
    time.sleep(2)
