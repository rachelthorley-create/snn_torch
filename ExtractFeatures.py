import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from collections import deque
import csv
from skimage.feature import graycomatrix, graycoprops

Output_Image = 20

# Run the experiment for each photo

class PlantState:

    def __init__(self, centre):

        self.centre = centre

        self.area_history = []

        self.mask = None



def process_experiment(folder, tray_corners):

    images = sorted(
        glob.glob(
            os.path.join(
                folder,
                "*.jpg"
            )
        )
    )

    results = {}

    final_image = None
    final_mask = None
    final_labels = None

    # Use first few images to calibrate plant centres
    calibration_images = images[:10]

    plant_centres = calibrate_plant_centres(
        calibration_images,
        tray_corners
    )

    for image_number, image_path in enumerate(images, start=1):

        print("Processing:", image_path)

        tray, mask = process_image(
            image_path,
            tray_corners
        )

        areas, features, visual = measure_plants(
            mask,
            plant_centres, 
            tray
        )

        results[image_number] = {

            "areas": areas,

            "features": features

        }

        if image_number == Output_Image :
            final_image = tray
            final_mask = mask
            final_labels = visual

    return (
        results,
        final_image,
        final_mask,
        final_labels
    )


# procesing an image pipeline


def process_image(image_path, tray_corners):

    image = cv2.imread(image_path)

    if image is None:
        raise Exception("Image not found")

    padding = 50

    large_width = 2048 + 2*padding
    large_height = 1276 + 2*padding


    matrix = get_tray_transform(
        tray_corners,
        output_size=(large_width, large_height),
        padding=padding
    )


    tray_large = rectify_tray(
        image,
        matrix,
        size=(large_width, large_height)
    )

    mask_large = segment_green(
        tray_large
    )

    mask_large = clean_mask(
        mask_large
    )

    mask = cv2.resize(
        mask_large,
        (1024, 638),
        interpolation=cv2.INTER_AREA
    )

    mask = (
        mask > 127
    ).astype(np.uint8) * 255

    tray = cv2.resize(
        tray_large,
        (1024, 638),
        interpolation=cv2.INTER_AREA
    )

    return tray, mask


# detect centers of each plant

def detect_initial_centres(mask):

    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(mask)


    centres=[]


    for label in range(1,num_labels):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]


        # remove noise
        if area < 20:
            continue


        cx,cy = centroids[label]


        centres.append(
            (
                int(cx),
                int(cy)
            )
        )


    return centres



# calibrate plant centres for each image


def calibrate_plant_centres(
        images,
        tray_corners):


    all_centres=[]


    for image_path in images:


        tray, mask = process_image(
            image_path,
            tray_corners
        )


        centres = detect_initial_centres(
            mask
        )


        all_centres.extend(
            centres
        )


    # Convert to numpy

    points=np.array(
        all_centres
    )


    # Cluster exactly 24 plants

    from sklearn.cluster import KMeans


    kmeans = KMeans(
        n_clusters=24,
        random_state=0
    )


    kmeans.fit(
        points
    )


    centres=[]

    for c in kmeans.cluster_centers_:

        centres.append(
            (
                int(c[0]),
                int(c[1])
            )
        )

    # sort by y first
    centres = sorted(
        centres,
        key=lambda p: p[1]
    )


    # split into 4 rows
    rows = []

    for i in range(4):

        row = centres[i*6:(i+1)*6]

        row = sorted(
            row,
            key=lambda p: p[0]
        )

        rows.append(row)


    # flatten back
    centres = [
        point
        for row in rows
        for point in row
    ]

    return centres


# Pot regions generation


def get_tray_transform(
        corners,
        output_size=(2048,1276),
        padding=0):


    width, height = output_size


    src = np.float32(corners)


    dst = np.float32([

        [padding, padding],
        [width-padding, padding],
        [width-padding, height-padding],
        [padding, height-padding]
    ])


    return cv2.getPerspectiveTransform(
        src,
        dst
    )


# Rectify the tray

def rectify_tray(
        image,
        matrix,
        size=(1024,638)):

    return cv2.warpPerspective(
        image,
        matrix,
        size
    )


#  HSV segmentation


def segment_green(tray):

    hsv = cv2.cvtColor(
        tray,
        cv2.COLOR_BGR2HSV
    )

    lower_green = np.array(
        [27,80,30]
    )

    upper_green = np.array(
        [95,255,220]
    )

    mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    return mask




def clean_mask(mask):

    kernel = np.ones(
        (5,5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    return mask


# Pot centers


def generate_pot_centres(
        width=1024,
        height=638,
        rows=4,
        cols=6):

    centres=[]

    cell_width=width/cols
    cell_height=height/rows

    for r in range(rows):

        for c in range(cols):

            centres.append(

                (
                    int((c+0.5)*cell_width),
                    int((r+0.5)*cell_height)
                )

            )

    return centres

# Assign conected components to plants


def measure_plants(mask, plant_centres, image):

    height, width = mask.shape


    # Slightly expand plant regions


    kernel = np.ones(
        (5,5),
        np.uint8
    )

    assignment_mask = cv2.dilate(
        mask,
        kernel,
        iterations=1
    )


    # labels:
    # 0 = unassigned
    # -1 = boundary

    labels = np.zeros(
        (height,width),
        np.int32
    )


    queue = deque()


    # initialise seeds

    for i,(cx,cy) in enumerate(plant_centres):

        # find nearest plant pixel
        # if centre is not exactly on plant

        found=False

        for r in range(0,30):

            for dy in range(-r,r+1):
                for dx in range(-r,r+1):

                    x=cx+dx
                    y=cy+dy

                    if (
                        0<=x<width and
                        0<=y<height and
                        mask[y,x]>0
                    ):

                        labels[y,x]=i+1

                        queue.append(
                            (x,y)
                        )

                        found=True
                        break

                if found:
                    break

            if found:
                break



    # 8-connected graph growth

    neighbours=[
        (-1,-1),(0,-1),(1,-1),
        (-1,0),          (1,0),
        (-1,1),(0,1),(1,1)
    ]


    while queue:


        x,y=queue.popleft()

        current=labels[y,x]


        for dx,dy in neighbours:

            nx=x+dx
            ny=y+dy


            if (
                nx<0 or
                ny<0 or
                nx>=width or
                ny>=height
            ):
                continue


            # only grow into plant pixels

            if assignment_mask[ny,nx]==0:
                continue



            # empty pixel

            if labels[ny,nx]==0:

                labels[ny,nx]=current

                queue.append(
                    (nx,ny)
                )


            # another plant reached this pixel

            elif labels[ny,nx]!=current:

                labels[ny,nx]=-1


    # Attach disconnected leaves based on proximity
    labels = attach_nearby_components(
        mask,
        labels,
        plant_centres,
        max_distance=40
    )

    # calculate areas

    areas=[]
    features=[]


    visual=np.zeros(
        (height,width,3),
        np.uint8
    )


    rng=np.random.default_rng(42)



    for i in range(
        1,
        len(plant_centres)+1
    ):

        # Use original mask for final plant measurement

        plant = (labels == i) & (mask > 0)


        area = np.sum(
            plant
        )


        areas.append(
            area
        )


        shape_features = calculate_shape_features(
            plant
        )


        
        colour_features = calculate_colour_features(
            plant,
            image
        )


        plant_features = {
            **shape_features,
            **colour_features
        }


        features.append(
            plant_features
        )


        colour=rng.integers(
            50,
            255,
            3,
            dtype=np.uint8
        )

        visual[plant]=colour



    return areas, features, visual





def attach_nearby_components(mask, labels, plant_centres, max_distance=40):

    # Find all disconnected green regions
    num_labels, component_labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask
    )

    height, width = mask.shape


    for component in range(1, num_labels):

        component_mask = component_labels == component

        area = stats[component, cv2.CC_STAT_AREA]


        # ignore tiny noise
        if area < 5:
            continue


        cx, cy = centroids[component]


        # Check if this component already belongs to a plant
        existing_labels = labels[component_mask]

        if np.any(existing_labels > 0):
            continue


        # Find closest plant centre
        distances = []

        for px, py in plant_centres:

            d = np.sqrt(
                (cx-px)**2 +
                (cy-py)**2
            )

            distances.append(d)


        closest = np.argmin(distances)

        distance = distances[closest]


        # Attach leaf if close enough
        if distance <= max_distance:

            labels[component_mask] = closest + 1


    return labels

# analysing and exposting shape features


def calculate_shape_features(plant_mask):

    # Pixel area (used for plant growth measurement)
    pixel_area = np.sum(plant_mask)


    # Very small objects have unreliable shape
    if pixel_area < 30:
        return {
            "compactness": 0,
            "solidity": 0
        }


    # Find contours
    contours, _ = cv2.findContours(
        plant_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    if len(contours) == 0:
        return {
            "compactness": 0,
            "solidity": 0
        }


    # Use largest contour
    contour = max(
        contours,
        key=cv2.contourArea
    )


    # Geometric area only for shape calculations
    contour_area = cv2.contourArea(
        contour
    )


    perimeter = cv2.arcLength(
        contour,
        True
    )


    # Compactness
    if perimeter == 0:

        compactness = 0

    else:

        compactness = (
            4 * np.pi * contour_area
        ) / (
            perimeter ** 2
        )


    # Convex hull
    hull = cv2.convexHull(
        contour
    )


    hull_area = cv2.contourArea(
        hull
    )


    # Solidity
    if hull_area == 0:

        solidity = 0

    else:

        solidity = (
            contour_area / hull_area
        )


    return {

        "compactness": compactness,
        "solidity": solidity
    }



# color features identification
def calculate_colour_features(plant_mask, image):

    pixels = image[plant_mask]

    if len(pixels) == 0:
        return {
            "mean_hue": 0,
            "mean_saturation": 0,
            "mean_value": 0,
            "hue_std": 0,
            "saturation_std": 0,
            "value_std": 0,
            "exg_std": 0,
            "glcm_contrast": 0,
        }

    # Convert BGR -> HSV

    hsv_pixels = cv2.cvtColor(
        pixels.reshape(-1, 1, 3),
        cv2.COLOR_BGR2HSV
    ).reshape(-1, 3)

    H = hsv_pixels[:, 0].astype(np.float32)
    S = hsv_pixels[:, 1].astype(np.float32)
    V = hsv_pixels[:, 2].astype(np.float32)

    # Ignore nearly grey pixels
    valid = S > 30

    if np.sum(valid) == 0:
        return {
            "mean_hue": 0,
            "mean_saturation": 0,
            "mean_value": 0,
            "hue_std": 0,
            "saturation_std": 0,
            "value_std": 0,
            "exg_std": 0,
            "glcm_contrast": 0,
        }

    H = H[valid]
    S = S[valid]
    V = V[valid]

    # Continuous HSV statistics

    mean_hue = np.mean(H)
    mean_saturation = np.mean(S)
    mean_value = np.mean(V)

    hue_std = np.std(H)
    saturation_std = np.std(S)
    value_std = np.std(V)

    # Excess Green statistics

    B = pixels[:, 0].astype(np.float32)[valid]
    G = pixels[:, 1].astype(np.float32)[valid]
    R = pixels[:, 2].astype(np.float32)[valid]

    exg = 2 * G - R - B
    exg_std = np.std(exg)

    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    texture = np.zeros_like(gray)
    texture[plant_mask] = gray[plant_mask]

    ys, xs = np.where(plant_mask)

    if len(xs) >= 100:

        x1 = xs.min()
        x2 = xs.max() + 1
        y1 = ys.min()
        y2 = ys.max() + 1

        roi = texture[y1:y2, x1:x2]

        # Quantise to 16 levels
        roi = (roi / 16).astype(np.uint8)

        glcm = graycomatrix(
            roi,
            distances=[1],
            angles=[0],
            levels=16,
            symmetric=True,
            normed=True
        )

        glcm_contrast = graycoprops(
            glcm,
            "contrast"
        )[0, 0]

    else:

        glcm_contrast = 0

    return {

        "mean_hue": mean_hue,

        "mean_saturation": mean_saturation,

        "mean_value": mean_value,

        "hue_std": hue_std,

        "saturation_std": saturation_std,

        "value_std": value_std,

        "exg_std": exg_std,

        "glcm_contrast": glcm_contrast,

    }

# Handling output to the CSV file


def export_feature_table(results, filename="feature_table.csv"):

    with open(filename, "w", newline="") as file:

        writer = csv.writer(
            file,
            delimiter=";"
        )


        # =========================================================
        # HEADER
        # =========================================================

        writer.writerow([

            "Image",
            "Plant",

            "Area_pixels",
            "Compactness",
            "Solidity",

            "Mean_hue",
            "Mean_Saturation",
            "Mean_Value",

            "Hue_STD",
            "Saturation_STD",
            "Value_STD",

            "ExG_STD",
            "GLCM_Contrast",

        ])


        # =========================================================
        # Write absolute feature values
        # =========================================================

        for image_name, data in results.items():

            areas = data["areas"]
            features = data["features"]


            for plant_id, area in enumerate(
                areas,
                start=1
            ):

                plant_features = features[plant_id - 1]


                writer.writerow([

                    image_name,

                    plant_id,


                    # Shape
                    int(area),

                    f"{plant_features['compactness']:.4f}".replace(".", ","),

                    f"{plant_features['solidity']:.4f}".replace(".", ","),


                    # Colour

                    f"{plant_features['mean_hue']:.2f}".replace(".", ","),

                    f"{plant_features['mean_saturation']:.2f}".replace(".", ","),

                    f"{plant_features['mean_value']:.2f}".replace(".", ","),


                    f"{plant_features['hue_std']:.4f}".replace(".", ","),

                    f"{plant_features['saturation_std']:.4f}".replace(".", ","),

                    f"{plant_features['value_std']:.4f}".replace(".", ","),


                    f"{plant_features['exg_std']:.4f}".replace(".", ","),

                    f"{plant_features['glcm_contrast']:.4f}".replace(".", ","),

                ])


    print(f"Saved {filename}")


if __name__ == "__main__":

    tray_corners = [

        (190,900),
        (2222,900),
        (2222,2166),
        (190,2166)

    ]

    folder = "dataset/Experiment1"

    results, final_image, final_mask, visual = process_experiment(

        folder,
        tray_corners

    )

    export_feature_table(results)

    plt.figure(
        figsize=(18,6)
    )

    plt.subplot(1,3,1)
    plt.imshow(
        cv2.cvtColor(
            final_image,
            cv2.COLOR_BGR2RGB
        )
    )
    plt.title("Rectified tray")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(
        final_mask,
        cmap="gray"
    )
    plt.title("Green mask")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(
        visual
    )
    plt.title("Detected plants")
    plt.axis("off")

    plt.show()
