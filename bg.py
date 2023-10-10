"""
python bg.py --video_source=/dev/video2 | ffmpeg -i - -f v4l2 -vf format=yuv420p /dev/video4

Para reproducir el stream:
ffplay -f v4l2 /dev/video4

Si recibimos el error `ioctl(VIDIOC_G_FMT)`:
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback devices=1 video_nr=4 exclusive_caps=1

References
----------
https://superuser.com/a/1331422
"""
import cv2
import argparse
import sys
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import numpy as np
import matplotlib.pyplot as plt
from cvzone.FaceMeshModule import FaceMeshDetector
import mediapipe
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_FACE_OVAL
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Realizar eliminación de fondo en un flujo de video"
    )
    parser.add_argument("--video_source", default="/dev/video2", help="Fuente de video")
    parser.add_argument("--background", default="fire.png", help="Imagen de fondo")
    return parser.parse_args()


def hist_match(source, template):
    # Calculate histogram of source and template
    source_hist, _ = np.histogram(source.ravel(), 256, [0, 256])
    template_hist, _ = np.histogram(template.ravel(), 256, [0, 256])

    # Normalize cumulative histogram
    source_cumsum = np.cumsum(source_hist) / source.size
    template_cumsum = np.cumsum(template_hist) / template.size

    # Create a mapping to match histograms
    hist_map = np.interp(source_cumsum, template_cumsum, np.arange(256))

    # Apply mapping
    return hist_map[source].reshape(source.shape).astype(np.uint8)


def convertScale(img, alpha, beta):
    """Add bias and gain to an image with saturation arithmetics. Unlike
    cv2.convertScaleAbs, it does not take an absolute value, which would lead to
    nonsensical results (e.g., a pixel at 44 with alpha = 3 and beta = -210
    becomes 78 with OpenCV, when in fact it should become 0).
    """

    new_img = img * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img.astype(np.uint8)


def automatic_brightness_and_contrast_based_on_background(img, background, influence: float = 0.5):
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_background = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)

    # Calculate the mean and standard deviation of the background
    mean_bg, stddev_bg = cv2.meanStdDev(gray_background)

    # Calculate the mean and standard deviation of the image
    mean_img, stddev_img = cv2.meanStdDev(gray_img)

    # Compute alpha (contrast) and beta (brightness)
    alpha = stddev_bg / stddev_img
    beta = mean_bg - alpha * mean_img

    # Apply influence
    alpha = 1 + (alpha - 1) * influence
    beta *= influence

    # Apply adjustments to each channel
    channels = cv2.split(img)
    adjusted_channels = []
    for ch in channels:
        adjusted_channels.append(convertScale(ch, alpha=alpha, beta=beta))

    # Merge channels back
    adjusted_img = cv2.merge(adjusted_channels)
    # plt.imshow(adjusted_img)

    return adjusted_img


def scale_points_around_origin(routes, scale):
    x0 = routes[:, 0].min()
    x1 = routes[:, 0].max()
    x_half = x0 + (x1 - x0) / 2

    y0 = routes[:, 1].min()
    y1 = routes[:, 1].max()
    y_half = y0 + (y1 - y0) / 2

    C = np.array(
        [
            [1, 0, -x_half],
            [0, 1, -y_half],
            [0, 0, 1],
        ]
    )

    # Crear matriz de escala
    S = np.array(
        [
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, 1],
        ]
    )

    # # Matriz de traslación
    # T = np.array([
    #     [1, 0, -routes[:, 0].min()],
    #     [0, 1, -routes[:, 1].min()],
    #     [0, 0, 1],
    # ])
    
    # Matriz original con una dimensión extra
    X = np.concatenate([routes, np.ones((routes.shape[0], 1))], axis=1)

    # Debug
    # plt.clf(); plt.plot(routes[:, 0], routes[:, 1]); plt.grid(); plt.xlim([0, 640]); plt.ylim([480, 0])
    
    # Concatenar transformaciones
    X_ = X @ C.T @ S.T @ np.linalg.inv(C.T)
    # Debug
    # plt.clf(); plt.plot(X_[:, 0], X_[:, 1]); plt.grid(); plt.xlim([0, 640]); plt.ylim([480, 0])

    return np.round(X_[:, :2]).astype(int)


def face_extractor(img, detector: FaceMeshDetector, routes_idx: list, scale: float = 1):
    chroma = np.ones_like(img) * [[255, 0, 255]]
    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    imgOut, faces = detector.findFaceMesh(scaled_img, draw=False)

    if faces:
        # Finding the coordinates of points
        routes = []
        face = faces[0]
        for source_idx, target_idx in routes_idx:
            routes.append(face[source_idx])
            routes.append(face[target_idx])
        routes = np.array(routes)

        # plt.clf(); plt.plot(routes[:, 0], routes[:, 1]); plt.grid()# ; plt.xlim([0, 640]); plt.ylim([480, 0])
        # plt.imshow(scaled_img)

        # ---------------------------------------------------------------------------------------- #
        scaled_image_bbox = scale_points_around_origin(
            np.array([[0, 0], [img.shape[0], 0], [img.shape[0], img.shape[1]], [0, img.shape[1]]]),
            scale,
        )
        # plt.plot(scaled_image_bbox[:, 1], scaled_image_bbox[:, 0], "x")
        
        # # Fill square with original image
        frame = np.ones_like(img) * [[255, 0, 255]]
        y0, x0 = scaled_image_bbox.min(axis=0)
        y1, x1 = scaled_image_bbox.max(axis=0)

        scaled_routes = routes.copy()
        scaled_routes[:, 0] += x0
        scaled_routes[:, 1] += y0
        # plt.plot(scaled_routes[:, 0], scaled_routes[:, 1]);

        if y0 < 0:
            scaled_img = scaled_img[-y0:-y0+img.shape[0]]
            y0 = 0
            y1 = img.shape[0]
        if x0 < 0:
            scaled_img = scaled_img[:, -x0:-x0+img.shape[1]]
            x0 = 0
            x1 = img.shape[1]
        frame[y0:y1, x0:x1] = scaled_img
        # ---------------------------------------------------------------------------------------- #

        mask = np.zeros((img.shape[0], img.shape[1]))
        mask = cv2.fillConvexPoly(mask, scaled_routes, 1)

        # Expand dims and stack 3 times to make it RGB
        mask = mask.astype(bool)
        mask = np.expand_dims(mask, axis=2)
        mask = np.concatenate([mask, mask, mask], axis=2)

        imgOut = np.where(mask, frame, chroma)
    else:
        imgOut = chroma

    # fig = plt.figure(figsize = (15, 15))
    # plt.axis('off')
    # plt.imshow(imgOut)

    imgOut = np.float32(imgOut)

    return imgOut


def main(video_source: str, background_file: str):
    cap = cv2.VideoCapture(video_source)

    w = 640
    h = 480

    # Set the frame width to 640 pixels
    cap.set(3, w)
    # Set the frame height to 480 pixels
    cap.set(4, h)

    bakground = cv2.imread(background_file)
    # bakground = cv2.cvtColor(bakground, cv2.COLOR_BGR2RGB)
    background = cv2.resize(bakground, (w, h))

    # Initialize the SelfiSegmentation class. It will be used for background removal.
    # model is 0 or 1 - 0 is general 1 is landscape(faster)
    segmentor = SelfiSegmentation(model=0)

    # Initialize FaceMeshDetector object
    # staticMode: If True, the detection happens only once, else every frame
    # maxFaces: Maximum number of faces to detect
    # minDetectionCon: Minimum detection confidence threshold
    # minTrackCon: Minimum tracking confidence threshold
    detector = FaceMeshDetector(staticMode=False, maxFaces=1, minDetectionCon=0.5, minTrackCon=0.5)
    df = pd.DataFrame(list(FACEMESH_FACE_OVAL), columns=["p1", "p2"])

    # order face oval lines
    routes_idx = []
    p1 = df.iloc[0]["p1"]
    p2 = df.iloc[0]["p2"]

    for i in range(len(df)):
        obj = df[df["p1"] == p2]
        p1 = obj["p1"].values[0]
        p2 = obj["p2"].values[0]

        current_route = []
        current_route.append(p1)
        current_route.append(p2)
        routes_idx.append(current_route)

    influence = 0.2

    if not cap.isOpened():
        print("Error: no se pudo abrir {}".format(video_source))
        exit()

    while True:
        # Capture a single frame
        success, img = cap.read()
        img = cv2.resize(img, (w, h))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Use the SelfiSegmentation class to remove the background
        # Replace it with a magenta background (0, 255, 0)
        # imgBG can be a color or an image as well. must be same size as the original if image
        # 'cutThreshold' is the sensitivity of the segmentation.
        # imgOut = segmentor.removeBG(img, imgBg=(255, 0, 255), cutThreshold=0.95)
        # ---------------------------------------------------------------------------------------- #
        imgOut = face_extractor(img, detector, routes_idx)

        u_chroma = np.array([255, 0, 255])
        l_chroma = np.array([255, 0, 255])

        mask = cv2.inRange(imgOut, l_chroma, u_chroma)

        # Reemplazamos la máscara con la imagen de fondo
        f = automatic_brightness_and_contrast_based_on_background(
            imgOut.copy(), background, influence=influence
        )
        # f = imgOut.copy()
        f[mask != 0] = background[mask != 0]

        # Stack the original image and the image with background removed side by side
        # imgStacked = cvzone.stackImages([img, imgOut, f], cols=3, scale=1)

        # ------------------------------- SACAR LA SALIDA A STDOUT ------------------------------- #
        # # Codificar el resultado en formato JPEG
        # is_success, im_buf_arr = cv2.imencode(".jpg", f)
        # if is_success:
        #     # Escribir en stdout
        #     sys.stdout.buffer.write(im_buf_arr.tobytes())
        #     sys.stdout.flush()
        # ---------------------------------------------------------------------------------------- #

        # Display the stacked images
        cv2.imshow("Image", f)

        # Wait for the user to press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_arguments()
    main(args.video_source, args.background)
