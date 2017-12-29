# coding:utf-8

import cv2
import numpy as np


class Triming:
    def __init__(self, img_path):
        self.img_path = img_path
        self.coins = cv2.imread("./static/upload_file/" + self.img_path)

    def get_gray_slace(self):
        return cv2.cvtColor(self.coins, cv2.COLOR_BGR2GRAY)

    def get_gaussian(self):
        coins_gray = self.get_gray_slace()
        return cv2.GaussianBlur(coins_gray, (5, 5), 0)

    def run_canny(self, min_coin_area=300):
        coins_preprocessed = self.get_gaussian()
        coins_binary = cv2.Canny(coins_preprocessed, 5, 50)
        coins_contours = cv2.findContours(coins_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        coins_and_contours = np.copy(self.coins)
        large_contours = [cnt for cnt in coins_contours[1] if cv2.contourArea(cnt) > min_coin_area]
        bounding_img = np.copy(self.coins)
        for contour in large_contours:
            x, y, w, h = cv2.boundingRect(contour)
            # cv2.rectangle(bounding_img, (x -5, y -5), (x + w +5, y + h +5), (255, 255, 255))
            cv2.imwrite('./static/triminged/' + self.img_path, bounding_img[y - 5:y + h + 5, x - 5:x + w + 5])


class Triming2:
    def __init__(self, img_path):
        self.img_path = img_path
        self.origin = cv2.imread("./static/upload_file/" + self.img_path, 1)
        self.coins = cv2.imread("./static/upload_file/" + self.img_path, 0)
        self.height = self.coins.shape[0]
        self.width = self.coins.shape[1]

    @classmethod
    def _display_hst(cls, coins, ave=False):
        """
        画像のヒストグラムを表示します。
        veは、ヒストグラムの平均を出力するかしないかのフラグ。デフォルトではFalse
        """

        # ヒストグラム表示用
        img_hst = np.zeros([100, 256]).astype('uint8')
        rows, cols = img_hst.shape[:2]

        # 度数分布を求める
        hdims = [256]
        hranges = [0, 256]
        hist = cv2.calcHist([coins], [0], None, hdims, hranges)
        # 　度数の最大値を取得
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(hist)
        for i in range(0, 256):
            v = hist[i]
            cv2.line(img_hst, (i, rows), (i, rows - rows * (v / max_val)), (255, 255, 255))
        if ave:
            return img_hst, np.sum(np.multiply(hist, [[i] for i in range(hdims[0])])) / (
                coins.shape[0] * coins.shape[1])  # aveをTrueにすれば、ヒストグラムの平均を返す
        else:
            return img_hst

    def filter(self):
        """
        :return: 平滑化された画像オブジェクトを返すまでを担当
        """
        img_hst, mean_hst = self._display_hst(self.coins, ave=True)
        bckg = 0
        if mean_hst >= 255 / 2:  # 背景が黒っぽいならば
            bckg = 255
        N = 4
        Bayer = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]
        Bayer_matrix = [[Bayer[i][j] * len(Bayer) * len(Bayer) for j in range(len(Bayer[i]))] for i in
                        range(len(Bayer))]
        img_tmp = np.zeros(self.coins.shape, dtype='uint8')
        for y in range(self.height):
            for x in range(self.width):
                v = self.coins[y][x]
                if v < Bayer_matrix[y % N][x % N]:
                    img_tmp[y][x] = bckg
                else:
                    img_tmp[y][x] = 255 - bckg
        img_tmp2 = cv2.medianBlur(img_tmp, 5)
        return img_tmp2

    def expand(self):
        """
        :return:膨張処理された画像オブジェクトを返すまでを担当
        """
        n = 8
        element = np.array([[1 for _ in range(n)] for __ in range(n)])
        img_tmp2 = self.filter()
        img_tmp3 = cv2.dilate(img_tmp2, element, iterations=4)
        return img_tmp3

    def extract_blob(self, img_tmp3):
        """
        :return: ブロブを取り出す処理
        """
        nlabel, img_lab = cv2.connectedComponents(img_tmp3)
        return nlabel, img_lab

    def find_max_blob(self, blobs):
        """
        :param blobs:
        :return: 面積が最大のblobを求めるかんせうう
        """
        i = 1
        b = cv2.compare(blobs, i, cv2.CMP_EQ)
        x, y, w, h = cv2.boundingRect(b)
        max_area = w * h
        for j in range(2, len(blobs)):
            b = cv2.compare(blobs, j, cv2.CMP_EQ)
            x, y, w, h = cv2.boundingRect(b)
            if max_area < w * h:
                i = j
                max_area = w * h
        return cv2.compare(blobs, i, cv2.CMP_EQ)

    def get_trimed_img(self):
        """
        :return: トリミング後の画像オブジェクトを得る
        """
        img_tmp3 = self.expand()
        n_label, img_lab = self.extract_blob(img_tmp3)
        img_tmp = self.find_max_blob(img_lab)
        x, y, w, h = cv2.boundingRect(img_tmp)
        if w == self.width and h == self.height:
            img_tmp2 = self.filter()
            nlabel, img_lab = cv2.connectedComponents(img_tmp2)
            img_tmp = self.find_max_blob(img_lab)
            x, y, w, h = cv2.boundingRect(img_tmp)
        img_dst = self.origin[max(0, y - 10): y + h + 10, max(0, x - 10): x + w + 10]
        return img_dst

    def run(self):
        img_dst = self.get_trimed_img()
        cv2.imwrite('./static/triminged/' + self.img_path, img_dst)


if __name__ == "__main__":
    print("try")
    trimming_class2 = Triming2("static/img/img832.jpg")
    print(trimming_class2.width, trimming_class2.height)
    img_dst = trimming_class2.get_trimed_img()
    cv2.imwrite("static/img/a.jpg", img_dst)

# #画像の読み込み
# coin_name = 'img/8.jpg'
# coins = cv2.imread(coin_name)
#
# #グレースケール化
# coins_gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
#
# #平滑化(GaussianBlur)
# coins_preprocessed = cv2.GaussianBlur(coins_gray, (5, 5), 0)
#
# #閾値処理 → キャニー法
# #_, coins_binary = cv2.threshold(coins_preprocessed, 160, 255, cv2.THRESH_BINARY)
# coins_binary = cv2.Canny(coins_preprocessed, 5, 50)
#
# #マスク処理
# #coins_binary1 = cv2.bitwise_not(coins_binary)
#
# #輪郭検出
# coins_contours = cv2.findContours(coins_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#
# #コピー作成
# coins_and_contours = np.copy(coins)
#
# # 60以上の面積のみ抽出
# min_coin_area = 300
# large_contours = [cnt for cnt in coins_contours[1] if cv2.contourArea(cnt) > min_coin_area]
#
# # 輪郭エリアを得る
# cv2.drawContours(coins_and_contours, large_contours, -1, (255,0,0), 3)
#
# # 画像内のコイン数を出力
# print('number of coins: %d' % len(large_contours))
#
# # コピー作成
# bounding_img = np.copy(coins)
#
# # それぞれのコインの輪郭を描写し、保存
# i = 0
# for contour in large_contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     #cv2.rectangle(bounding_img, (x -5, y -5), (x + w +5, y + h +5), (255, 255, 255))
#     cv2.imwrite(coin_name +str(i)+'.jpg', bounding_img[y -5:y+h +5, x -5:x+w +5])
#     i += 1
