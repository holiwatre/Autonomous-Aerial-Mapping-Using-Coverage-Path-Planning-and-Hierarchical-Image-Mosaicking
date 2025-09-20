import cv2
import numpy as np
import sys

class Image_Stitching():
    def __init__(self):
        # 매칭 파라미터
        self.ratio = 0.85
        self.min_match = 10
        # SIFT 생성: OpenCV ≥4.4에서는 cv2.SIFT_create() 사용
        try:
            self.sift = cv2.SIFT_create()
        except AttributeError:
            # 구버전(OpenCV <4.4) 호환을 위해 contrib 모듈 폴백
            self.sift = cv2.xfeatures2d.SIFT_create()
        # 블렌딩을 위한 스무딩 윈도우 크기
        self.smoothing_window_size = 800

    def registration(self, img1, img2):
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good_points = []
        good_matches = []
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        # 매칭 결과 저장
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
        cv2.imwrite('matching.jpg', img3)

        H = None
        if len(good_points) > self.min_match:
            image1_kp = np.float32([kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32([kp2[i].pt for (i, _) in good_points])
            H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)
        return H

    def create_mask(self, img1, img2, version):
        h1, w1 = img1.shape[:2]
        w2 = img2.shape[1]
        h_panorama = h1
        w_panorama = w1 + w2
        offset = int(self.smoothing_window_size / 2)
        barrier = w1 - offset
        mask = np.zeros((h_panorama, w_panorama), dtype=np.float32)

        if version == 'left_image':
            mask[:, barrier - offset:barrier + offset] = np.tile(
                np.linspace(1, 0, 2 * offset).T, (h_panorama, 1)
            )
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset:barrier + offset] = np.tile(
                np.linspace(0, 1, 2 * offset).T, (h_panorama, 1)
            )
            mask[:, barrier + offset:] = 1

        return cv2.merge([mask, mask, mask])

    def blending(self, img1, img2):
        H = self.registration(img1, img2)
        if H is None:
            raise RuntimeError('충분한 매칭 포인트가 없어 호모그래피를 계산할 수 없습니다.')

        h1, w1 = img1.shape[:2]
        w2 = img2.shape[1]
        h_panorama = h1
        w_panorama = w1 + w2

        # 왼쪽 이미지 블렌딩
        panorama1 = np.zeros((h_panorama, w_panorama, 3), dtype=np.float32)
        mask1 = self.create_mask(img1, img2, version='left_image')
        panorama1[0:h1, 0:w1] = img1
        panorama1 *= mask1

        # 오른쪽 이미지 워핑 및 블렌딩
        mask2 = self.create_mask(img1, img2, version='right_image')
        panorama2 = cv2.warpPerspective(img2, H, (w_panorama, h_panorama)).astype(np.float32)
        panorama2 *= mask2

        result = panorama1 + panorama2
        # 결과 크롭
        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col].astype(np.uint8)
        return final_result


def main(img_path1, img_path2):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    if img1 is None or img2 is None:
        print('이미지를 불러올 수 없습니다. 경로를 확인하세요.')
        return

    stitcher = Image_Stitching()
    panorama = stitcher.blending(img1, img2)
    cv2.imwrite('panorama.jpg', panorama)
    print('파노라마 이미지가 panorama.jpg로 저장되었습니다.')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('사용법: python3 linrl3_image_stitching.py <이미지1> <이미지2>')
    else:
        main(sys.argv[1], sys.argv[2])