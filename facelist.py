import cv2
import imagehash
import requests
from PIL import Image
from io import BytesIO


class FaceList:
    def __init__(self, download=False):
        self._faceid = dict()
        self.face_hashes = dict()
        with open('char_faceid.txt') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                name, char_id = line.split()
                self._faceid[name] = [
                    int('1{:03d}11'.format(int(char_id))),
                    int('1{:03d}31'.format(int(char_id)))
                ]

        if download:
            self.download()

        self.methods = [
            imagehash.average_hash,
            imagehash.phash,
            imagehash.dhash,
            imagehash.whash,
            lambda img: imagehash.whash(img, mode='db4'),
        ]

        for char in self._faceid:
            self.face_hashes[char] = list()
            for uid in self._faceid[char]:
                face = cv2.imread('./thumbnail/{}_{}.png'.format(char, str(uid)[4]))
                face_hashes = [method(Image.fromarray(self.preprocess(face))) for method in self.methods]
                self.face_hashes[char].append(face_hashes)

    def preprocess(self, img):
        img[:24, :56, :] = 0
        img[102:, :92, :] = 0
        return img[12:128 - 12, 12:128 - 12, :]

    def download(self):
        for char in self._faceid:
            for uid in self._faceid[char]:
                r = requests.get('https://redive.estertion.win/icon/unit/{}.webp'.format(uid))
                if r.status_code == requests.codes.ok:
                    img = Image.open(BytesIO(r.content))
                    print(img)
                    img.save('./thumbnail/{}_{}.png'.format(char, str(uid)[4]))

    def hash_images(self, methods):
        pass


if __name__ == '__main__':
    fl = FaceList(download=True)
