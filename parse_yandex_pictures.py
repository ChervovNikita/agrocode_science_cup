import requests
from bs4 import BeautifulSoup
import random
import ua_parser as ua
from tqdm import tqdm
import pandas as pd

def get_link_img(text):
    url = 'https://yandex.ru/images/search?text=' + text
    response=requests.get(url,headers={'user_agent':f'{ua}'})
    soap= BeautifulSoup(response.content,"html.parser")
    print(soap)
    links=soap.find_all("img",class_="serp-item__thumb justifier__thumb")
    names = []
    for link in links:
        link = link.get("src")
        linked = "https:"+str(link)
        #writing to a file
        name=hex(random.randint(0, int(1e20)))
        names.append(name)
        p = requests.get(linked)
        out = open(f"extra_data/{name}.jpg", "wb")
        out.write(p.content)
        out.close()
    return names


def main():
    labels_to_seart = open('labels.txt', 'r').read().split('\n\n')

    prev_data = pd.read_csv('extra_data.csv')
    idxs = list(prev_data.idx)
    item_nms = list(prev_data.item_nm)

    start = 1
    for label in tqdm(labels_to_seart):
        # if start == 0:
        #     if label == 'кронштейн':
        #         start = 1
        #     print(label)
        #     continue
        print(label)
        label = 'строительный ' + label
        paths = get_link_img(label)
        print(len(paths))
        for path in paths:
            idxs.append(path)
            item_nms.append(label)
        df = pd.DataFrame({'idx': idxs, 'item_nm': item_nms})
        df.to_csv('extra_data.csv')
        break


if __name__ == '__main__':
    main()


