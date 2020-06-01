import colored
import csv
import cv2
import glob
import numpy as np
import imutils
# import time
import pytesseract

from collections import Counter
from facelist import FaceList
from PIL import Image


def find_all_templates(img, template):
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    (_, max_val, _, _) = cv2.minMaxLoc(res)

    threshold = max_val * 0.95
    loc = np.where(res >= threshold)
    w, h = template.shape[:2][::-1]

    return loc


def transparent_png(img):
    alpha_channel = img[:, :, 3]
    rgb_channels = img[:, :, :3]

    # White Background Image
    white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255

    # Alpha factor
    alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)

    # Transparent Image Rendered on White Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    white = white_background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = base + white
    return final_image.astype(np.uint8)


def multiscale_template_matching(img, template):
    scale_factor = 0.3
    img = imutils.resize(img, width=int(img.shape[1] * scale_factor))
    template = imutils.resize(template, width=int(template.shape[1] * scale_factor))

    w, h = template.shape[:2][::-1]

    found = None

    min_scale = template.shape[1] * 10 / img.shape[1]
    max_scale = min_scale * 4

    for scale in np.linspace(min_scale, max_scale, 10)[::-1]:
        resized = imutils.resize(img, width=int(img.shape[1] * scale))
        r = img.shape[1] / resized.shape[1]

        if resized.shape[0] < h or resized.shape[1] < w:
            break

        res = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF)
        (_, max_val, _, max_loc) = cv2.minMaxLoc(res)

        if found is None or found[0] < max_val:
            found = max_val, max_loc, r

    _, max_loc, r = found
    return max_loc, r


def group_locs(loc):
    result = dict()
    for pt in zip(*loc[::-1]):
        flag = False
        for k in result:
            for v in result[k]:
                if abs(v[0] - pt[0]) < 10 and abs(v[1] - pt[1]) < 10:
                    result[k].append(pt)
                    flag = True
                    break
        if not flag:
            result[pt] = [pt]
    return sorted(list(result.keys()), key=lambda x: x[1])


def locate_frame(img, sx, sy, ex, ey):
    # img = img[sy:ey, sx:ex, :]
    lum = (img[:, :, 0] * 0.2126 + img[:, :, 1] * 0.7152 + img[:, :, 2] * 0.0722) / 255

    mid_x = (sx + ex) // 2
    mid_y = (sy + ey) // 2

    sx = (sx + mid_x) // 2
    ex = (ex + mid_x) // 2
    sy = (sy + mid_y) // 2
    ey = (ey + mid_y) // 2

    # expand sx to left
    # while np.mean(lum[sy:ey, sx]) < 0.9:
    while np.mean((lum[sy:ey, sx] > 0.9).astype(np.float)) < 0.9:
        sx -= 1
    # expand ex to right
    while np.mean((lum[sy:ey, ex] > 0.9).astype(np.float)) < 0.9:
        ex += 1
    # expand sy to top
    while np.mean((lum[sy, sx:ex] > 0.9).astype(np.float)) < 0.9:
        sy -= 1
    # expand ey to bottom
    while np.mean((lum[ey, sx:ex] > 0.9).astype(np.float)) < 0.9:
        ey += 1

    return sy + 1, ey - 1, sx + 1, ex - 1


def find_star(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if img.shape[0] > img.shape[1]:
        resized = imutils.resize(img, width=128)
    else:
        resized = imutils.resize(img, height=128)

    if resized.shape[0] > 128:
        resized = resized[:128, :, :]
    if resized.shape[1] > 128:
        resized = resized[:, :128, :]

    sx, ex = 8, 91
    sy, ey = 102, 122
    star_area = resized[sy:ey, sx:ex, :]

    star_list = []
    for i in range(5):
        subarea = star_area[:, int((ex - sx) * i / 5):int((ex - sx) * (i + 1) / 5) - 1, :]
        is_star = np.mean(subarea[:, :, 1]) > 80
        star_list.append(is_star)
    assert sum(star_list[:sum(star_list)]) == sum(star_list)
    assert sum(star_list) >= 1
    return sum(star_list)


def find_unit(img, fl, name=None):
    if img.shape[0] > img.shape[1]:
        resized = imutils.resize(img, width=128)
    else:
        resized = imutils.resize(img, height=128)

    if resized.shape[0] > 128:
        resized = resized[:128, :, :]
    if resized.shape[1] > 128:
        resized = resized[:, :128, :]

    # Image.fromarray(resized).save(name)

    methods = fl.methods
    results = [[] for _ in methods]

    img_hashes = [method(Image.fromarray(fl.preprocess(resized))) for method in methods]

    for unit in fl.face_hashes:
        for face_hashes in fl.face_hashes[unit]:
            for idx, (img_hash, face_hash) in enumerate(zip(img_hashes, face_hashes)):
                results[idx].append((unit, img_hash - face_hash))

    top = Counter(
            sum([[x[0] for x in result if x[1] == sorted(result, key=lambda x: x[1])[0][1]] for result in results], []))
    # top = Counter([sorted(result, key=lambda x: x[1])[0][0] for result in results])

    if top.most_common(1)[0][1] >= len(methods) - 1:
        return top.most_common(1)[0][0]
    elif top.most_common(1)[0][1] >= (len(methods) + 1) // 2:
        # print('Warning')
        for result in results:
            # print([(x[0], x[1]) for x in sorted(result, key=lambda x: x[1])[:5]])
            pass
        return top.most_common(1)[0][0]
    else:
        for result in results:
            # print([(x[0], x[1]) for x in sorted(result, key=lambda x: x[1])[:10]])
            pass
        raise Exception('Ambiguous image')


def find_level(img):
    y, x = int(img.shape[0] * 30 / 128), int(img.shape[1] * 65 / 128)
    patch = img[:y, :x, :]

    # patch = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    # patch = np.logical_and(patch[:, :, 1] < 120, (patch[:, :, 2] < 120)).astype(np.uint8) * 255

    Image.fromarray(patch).save('level.png')
    level = pytesseract.image_to_string(patch)
    # print(level)
    return level


def parse_screenshot(path, fl, win, lose):
    # print(path)
    # start_t = time.time()

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    _, r = multiscale_template_matching(img, win)
    win = imutils.resize(win, width=int(win.shape[1] * r))

    # _, r = multiscale_template_matching(img, lose)
    lose = imutils.resize(lose, height=int(win.shape[0]))

    # print('Multi-scale detection : {:.3f} sec'.format(time.time() - start_t))
    # start_t = time.time()

    win_loc = find_all_templates(img, win)
    lose_loc = find_all_templates(img, lose)

    win_loc = group_locs(win_loc)
    lose_loc = group_locs(lose_loc)

    # print('Location : {:.3f} sec'.format(time.time() - start_t))

    out_img = img.copy()

    for loc in win_loc:
        cv2.rectangle(out_img, loc, (loc[0] + win.shape[1], loc[1] + win.shape[0]), (255, 0, 0), 2)
    for loc in lose_loc:
        cv2.rectangle(out_img, loc, (loc[0] + lose.shape[1], loc[1] + lose.shape[0]), (0, 0, 255), 2)
    # Image.fromarray(out_img).save('rect.png')

    assert len(win_loc) == len(lose_loc), 'Win : {} / Lose : {}'.format(win_loc, lose_loc)

    # start_t = time.time()
    result = []
    failed = False
    for wl, ll in zip(win_loc, lose_loc):
        if wl[0] < ll[0]:
            status = '공덱승'
        else:
            status = '방덱승'

        x1, x2 = min(wl[0], ll[0]), max(wl[0], ll[0])
        face_size = int((x2 - x1) / 5.95)
        y = wl[1] + int(face_size / 1.8)

        atk_deck = []
        for i in range(5):
            sx, sy = x1 + face_size * i, y
            ex, ey = x1 + face_size * (i + 1), y + face_size
            try:
                sy, ey, sx, ex = locate_frame(img, sx, sy, ex, ey)
                frame = img[sy:ey, sx:ex, :]
                unit = find_unit(frame, fl, name='atk_{}.png'.format(i))
                star = find_star(frame)
                # lv = find_level(frame)
                lv = None
                print((unit, star, lv), end='\t')
                atk_deck.append((unit, star, lv))
            except Exception as exc:
                # print(exc)
                failed = True
                print('{}fail{}'.format(colored.bg(9), colored.attr('reset')), end='\t')
            cv2.rectangle(out_img, (sx, sy), (ex, ey), (255, 0, 0), 2)
        print('vs', end='\t')

        def_deck = []
        for i in range(5):
            sx, sy = x2 + face_size * i, y
            ex, ey = x2 + face_size * (i + 1), y + face_size
            try:
                sy, ey, sx, ex = locate_frame(img, sx, sy, ex, ey)
                frame = img[sy:ey, sx:ex, :]
                unit = find_unit(frame, fl, name='def_{}.png'.format(i))
                star = find_star(frame)
                # lv = find_level(frame)
                lv = None
                print((unit, star, lv), end='\t')
                def_deck.append((unit, star, lv))
            except Exception as exc:
                print(exc)
                failed = True
                print('{}fail{}'.format(colored.bg(9), colored.attr('reset')), end='\t')
            cv2.rectangle(out_img, (sx, sy), (ex, ey), (0, 0, 255), 2)
        print(status)

        # TODO: if failed, find candidates based on other units and try again

        result.append((tuple(atk_deck), tuple(def_deck), status))
        # Image.fromarray(out_img).save('rect.png')
        # assert not failed

    # print('Unit Detection : {:.3f} sec'.format(time.time() - start_t))

    return result, failed


def flatten(d):
    if isinstance(d, list) or isinstance(d, tuple):
        return sum([flatten(x) for x in d], [])
    else:
        return [d]


def main():
    fl = FaceList()

    win = cv2.imread('./resources/win.png', cv2.IMREAD_UNCHANGED)
    win = transparent_png(win)
    win = cv2.cvtColor(win, cv2.COLOR_BGR2RGB)

    lose = cv2.imread('./resources/lose.png', cv2.IMREAD_UNCHANGED)
    lose = transparent_png(lose)
    lose = cv2.cvtColor(lose, cv2.COLOR_BGR2RGB)

    # for path in ['./arena_screenshots/deck{}.jpg'.format(i) for i in range(3, 4)]:
    total_cnt = 0
    failed_cnt = 0

    with open('arena.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for path in glob.glob('./arena_screenshots/*.*'):
            try:
                results, failed = parse_screenshot(path, fl, win, lose)
                if not failed:
                    for result in results:
                        writer.writerow(flatten(result))
            except Exception as exc:
                print(path, exc)
                failed = True
            total_cnt += 1
            if failed:
                failed_cnt += 1
    print('Success : {} / {} ({:.2f}%)'.format(
        total_cnt - failed_cnt, total_cnt, (total_cnt - failed_cnt) / total_cnt * 100))


if __name__ == '__main__':
    main()
