"""
Kamera kalibrasyonu (Zhang yöntemi) - satranç tahtası ile.

Üretilen dosya:
  config/camera_calib.npz  -> K (iç matris), dist (distorsiyon katsayıları)

Odak uzaklığı (piksel) = K[0,0]; config.yaml -> camera.focal_length_px alanına yazın.

Kullanım:
  1) 9x6 iç köşeli satranç tahtası yazdırın (kare boyutu önemli değil, düzlem yeter).
  2) python tools/calibrate_camera.py --index 0
  3) Tahtayı farklı açı/mesafelerde gösterin; köşeler bulununca SPACE ile örnek alın.
     En az 15 örnek sonrası 'c' ile kalibre edin, 'q' ile çıkın.
"""
import argparse
import os

import cv2
import numpy as np

PATTERN = (9, 6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", default="config/camera_calib.npz")
    args = ap.parse_args()

    objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2)

    obj_points, img_points = [], []
    cap = cv2.VideoCapture(args.index)
    shape = None

    print("SPACE: örnek al | c: kalibre et | q: çık")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shape = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, PATTERN, None)
        disp = frame.copy()
        if found:
            cv2.drawChessboardCorners(disp, PATTERN, corners, found)
        cv2.putText(disp, f"ornek: {len(obj_points)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("kalibrasyon", disp)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(' ') and found:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            obj_points.append(objp)
            img_points.append(corners)
        elif k == ord('c') and len(obj_points) >= 10:
            rms, K, dist, _, _ = cv2.calibrateCamera(
                obj_points, img_points, shape, None, None)
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            np.savez(args.out, K=K, dist=dist)
            print(f"RMS hata: {rms:.3f}")
            print(f"Odak uzaklığı fx={K[0,0]:.1f} px, fy={K[1,1]:.1f} px")
            print(f"Kaydedildi: {args.out}")
            print("config.yaml -> camera.focal_length_px değerini fx ile güncelleyin.")
        elif k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
