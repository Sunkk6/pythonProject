# -*- coding: utf-8 -*-
import re
import requests
from urllib import error
from bs4 import BeautifulSoup
import os

num = 0
numPicture1 = 5000
file1 = 'C:\\Users\\28972\\Desktop\\垃圾数据集\\'
List = []


def Find(url, A):
    global List
    print('正在检测图片总数，请稍等.....')
    t = 0
    i = 1
    s = 0
    while t < 5000:
        Url = url + str(t)
        try:
            # 这里搞了下
            Result = A.get(Url, timeout=7, allow_redirects=False)
        except BaseException:
            t = t + 60
            continue
        else:
            result = Result.text
            pic_url = re.findall('"objURL":"(.*?)",', result, re.S)  # 先利用正则表达式找到图片url
            # thumbURL objURL
            s += len(pic_url)
            if len(pic_url) == 0:
                break
            else:
                List.append(pic_url)
                t = t + 60
    return s


def recommend(url):
    Re = []
    try:
        html = requests.get(url, allow_redirects=False)
    except error.HTTPError as e:
        return
    else:
        html.encoding = 'utf-8'
        bsObj = BeautifulSoup(html.text, 'html.parser')
        div = bsObj.find('div', id='topRS')
        if div is not None:
            listA = div.findAll('a')
            for i in listA:
                if i is not None:
                    Re.append(i.get_text())
        return Re


def dowmloadPicture(html, keyword):
    global num
    # t =0
    pic_url = re.findall('"objURL":"(.*?)",', html, re.S)  # 先利用正则表达式找到图片url
    print('找到关键词:' + keyword + '的图片，即将开始下载图片...')
    for each in pic_url:
        print('正在下载第' + str(num + 1) + '张图片，图片地址:' + str(each))
        try:
            if each is not None:
                pic = requests.get(each, timeout=7)
            else:
                continue
        except BaseException:
            print('错误，当前图片无法下载')
            continue
        else:
            string = file + r'\\' + keyword + '_' + str(num) + '.jpg'
            # string = r'img' + '_' + keyword + '_' + str(num) + '.jpg'
            fp = open(string, 'wb')
            fp.write(pic.content)
            fp.close()
            num += 1
        if num >= 5000:
            return


if __name__ == '__main__':  # 主函数入口

    headers = {
        'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
        'Connection': 'keep-alive',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0',
        'Upgrade-Insecure-Requests': '1'
    }

    A = requests.Session()
    A.headers = headers

    ###############################

    tm = numPicture1
    numPicture = tm
    line_list = []
    with open(r'C:\Users\28972\Desktop\name.txt', encoding='utf-8') as file:
        line_list = [k.strip() for k in file.readlines()]  # 用 strip()移除末尾的空格

    for word in line_list:
        url = 'https://image.baidu.com/search/flip?tn=baiduimage&istype=2&ie=utf-8&gsm=5a&z=7&word=' + word + '&pn='
        # url = 'https://cn.bing.com/images/async?q=%E8%80%81%E5%B8%88&first='+word+'&count=35&relp=35&scenario' \
        # '=ImageBasicHover&datsrc=N_I&layout' \ '=RowBased&mmasync=1 '
        tot = Find(url, A)
        Recommend = recommend(url)  # 记录相关推荐
        print('经过检测%s类图片共有%d张' % (word, tot))
        file = file1 + word
        y = os.path.exists(file)
        if y == 1:
            print('该文件已存在，请重新输入')
            file = word + '文件夹2'
            os.mkdir(file)
        else:
            os.mkdir(file)
        t = 0
        tmp = url
        while t < numPicture1:
            try:
                url = tmp + str(t)
                # result = requests.get(url, timeout=10)
                # 这里搞了下
                result = A.get(url, timeout=10, allow_redirects=False)
                print(url)
            except error.HTTPError as e:
                print('网络错误，请调整网络后重试')
                t = t + 60
            else:
                dowmloadPicture(result.text, word)
                t = t + 60
        numPicture = numPicture + tm
        num = 0

    print('杰杰杰杰杰杰结束了')
