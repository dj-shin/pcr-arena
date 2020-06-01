import glob
import requests
from bs4 import BeautifulSoup as bs
from PIL import Image
from io import BytesIO


def crawl(idx):
    if len(glob.glob('./gall_screenshots/{}_*.*'.format(idx))) > 0:
        return
    open('./gall_screenshots/{}_flag.ckpt'.format(idx), 'wt')

    url = 'https://gall.dcinside.com/mgallery/board/view/?id=iriya1&no={}'.format(idx)
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/\
                *;q=0.8,application/signed-exchange;v=b3',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        'Connection': 'keep-alive',
        'Host': 'gall.dcinside.com',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36\
                (KHTML, like Gecko) Chrome/76.0.3809.100 Safari/537.36',
    }
    sess = requests.Session()
    sess.headers.update(headers)
    sess.proxies['http'] = 'socks5h://localhost:9050'
    sess.proxies['https'] = 'socks5h://localhost:9050'

    resp = sess.get(url)
    if not resp.status_code == requests.codes.ok:
        return
    soup = bs(resp.text, features='html.parser')
    print(idx, resp)
    images = ['https://image.dcinside.com/' + img['src'].split('/')[-1] for img in soup.find(
        class_='gallview_contents').find(class_='writing_view_box').find_all('img')]
    for img_idx, img_url in enumerate(images):
        resp = sess.get(img_url)
        if resp.status_code == requests.codes.ok:
            Image.open(BytesIO(resp.content)).save('./gall_screenshots/{}_{}.png'.format(idx, img_idx))


def main():
    for idx in range(2163):
        crawl(idx)


if __name__ == '__main__':
    main()
