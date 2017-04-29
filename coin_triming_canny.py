#coding:utf-8

import cv2, matplotlib
import numpy as np
import matplotlib.pyplot as plt

#画像の読み込み
coin_name = 'img/8.jpg'
coins = cv2.imread(coin_name)

#グレースケール化
coins_gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)

#平滑化(GaussianBlur)
coins_preprocessed = cv2.GaussianBlur(coins_gray, (5, 5), 0)

#閾値処理 → キャニー法
#_, coins_binary = cv2.threshold(coins_preprocessed, 160, 255, cv2.THRESH_BINARY)
coins_binary = cv2.Canny(coins_preprocessed, 5, 50)

#マスク処理
#coins_binary1 = cv2.bitwise_not(coins_binary)

#輪郭検出
coins_contours = cv2.findContours(coins_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


#コピー作成
coins_and_contours = np.copy(coins)

# 60以上の面積のみ抽出
min_coin_area = 300
large_contours = [cnt for cnt in coins_contours[1] if cv2.contourArea(cnt) > min_coin_area]

# 輪郭エリアを得る
cv2.drawContours(coins_and_contours, large_contours, -1, (255,0,0), 3)

# 画像内のコイン数を出力
print('number of coins: %d' % len(large_contours))

# コピー作成
bounding_img = np.copy(coins)

# それぞれのコインの輪郭を描写し、保存
i = 0
for contour in large_contours:
    x, y, w, h = cv2.boundingRect(contour)
    #cv2.rectangle(bounding_img, (x -5, y -5), (x + w +5, y + h +5), (255, 255, 255))
    cv2.imwrite(coin_name +str(i)+'.jpg', bounding_img[y -5:y+h +5, x -5:x+w +5])
    i += 1
