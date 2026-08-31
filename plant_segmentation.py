import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from skimage.morphology import skeletonize


# ==================================================
# GLOBAL RESOLUTION
# ==================================================

MASK_WIDTH = 2048
MASK_HEIGHT = 1280



# ==================================================
# LEAF OBJECT
# ==================================================

class LeafState:


    def __init__(self, leaf_id):

        self.id = leaf_id

        self.area = 0

        self.centroid = None

        self.direction = None

        self.mask = None

        self.colour = None

        self.age = 0



# ==================================================
# PLANT OBJECT
# ==================================================

class PlantState:


    def __init__(self, centre):


        self.centre = centre


        self.leaves = []


        self.mask = np.zeros(
            (MASK_HEIGHT,MASK_WIDTH),
            np.uint8
        )


        self.area = 0




# ==================================================
# BACKGROUND MODEL
# ==================================================

def create_background_model(images, tray_corners):


    frames=[]


    for path in images[:5]:


        tray=load_and_rectify(
            path,
            tray_corners
        )


        frames.append(
            tray
        )



    frames=np.array(frames)



    background=np.median(
        frames,
        axis=0
    )



    return background.astype(
        np.uint8
    )




# ==================================================
# IMAGE LOADING
# ==================================================

def load_and_rectify(
        path,
        corners):


    image=cv2.imread(path)


    if image is None:

        raise Exception(
            "Image not found"
        )



    matrix=get_tray_transform(
        corners
    )


    return rectify_tray(
        image,
        matrix
    )





# ==================================================
# PERSPECTIVE CORRECTION
# ==================================================

def get_tray_transform(
        corners,
        output_size=(MASK_WIDTH,MASK_HEIGHT)):


    width,height=output_size


    src=np.float32(
        corners
    )


    dst=np.float32([

        [0,0],
        [width,0],
        [width,height],
        [0,height]

    ])



    return cv2.getPerspectiveTransform(
        src,
        dst
    )




def rectify_tray(
        image,
        matrix,
        size=(MASK_WIDTH,MASK_HEIGHT)):


    return cv2.warpPerspective(
        image,
        matrix,
        size
    )





# ==================================================
# LEAF PROBABILITY MAP
# ==================================================

def create_leaf_probability(
        image,
        background):


    # -----------------------------
    # Background difference
    # -----------------------------


    diff=cv2.absdiff(
        image,
        background
    )


    diff=np.max(
        diff,
        axis=2
    )


    diff=cv2.normalize(
        diff,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )




    # -----------------------------
    # Colour difference
    # -----------------------------


    hsv=cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )


    h,s,v=cv2.split(
        hsv
    )



    colour_score=s.astype(
        np.float32
    )



    probability=(

        0.6*diff
        +
        0.4*colour_score

    )



    return probability.astype(
        np.uint8
    )





def probability_to_mask(
        probability):


    _,mask=cv2.threshold(

        probability,

        60,

        255,

        cv2.THRESH_BINARY

    )


    return mask.astype(
        np.uint8
    )






# ==================================================
# MASK CLEANING
# ==================================================

def clean_mask(mask):


    kernel=cv2.getStructuringElement(

        cv2.MORPH_ELLIPSE,

        (3,3)

    )


    mask=cv2.morphologyEx(

        mask,

        cv2.MORPH_OPEN,

        kernel,

        iterations=1

    )



    mask=cv2.morphologyEx(

        mask,

        cv2.MORPH_CLOSE,

        kernel,

        iterations=1

    )


    return mask






# ==================================================
# GRID REMOVAL
# ==================================================

def remove_grid_lines(mask,tray):


    result=mask.copy()



    gray=cv2.cvtColor(
        tray,
        cv2.COLOR_BGR2GRAY
    )


    edges=cv2.Canny(
        gray,
        80,
        160
    )



    lines=cv2.HoughLinesP(

        edges,

        rho=1,

        theta=np.pi/180,

        threshold=120,

        minLineLength=200,

        maxLineGap=20

    )



    if lines is None:

        return result



    line_mask=np.zeros_like(
        mask
    )



    for line in lines:


        # OpenCV normally returns [[x1,y1,x2,y2]]
        # but sometimes returns [x1,y1,x2,y2]

        coords=line.flatten()


        if len(coords)!=4:

            continue



        x1,y1,x2,y2=coords



        # remove only strong vertical/horizontal grid lines

        if abs(x1-x2)<5 or abs(y1-y2)<5:


            cv2.line(

                line_mask,

                (int(x1),int(y1)),

                (int(x2),int(y2)),

                255,

                5

            )



    result[line_mask>0]=0


    return result

# ==================================================
# POT CENTRES
# ==================================================

def generate_pot_centres(
        width=MASK_WIDTH,
        height=MASK_HEIGHT,
        rows=4,
        cols=6):


    centres=[]


    cw=width/cols
    ch=height/rows



    for r in range(rows):

        for c in range(cols):

            centres.append(

                (
                    int((c+0.5)*cw),
                    int((r+0.5)*ch)
                )

            )


    return centres





# ==================================================
# COLOUR FEATURES
# ==================================================

def extract_colour_features(
        image,
        mask):


    hsv=cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )


    pixels=hsv[mask>0]


    if len(pixels)==0:

        return None



    mean_h=np.mean(
        pixels[:,0]
    )


    mean_s=np.mean(
        pixels[:,1]
    )


    mean_v=np.mean(
        pixels[:,2]
    )



    return {

        "hue":mean_h,

        "saturation":mean_s,

        "brightness":mean_v

    }





# ==================================================
# LEAF EXTRACTION
# ==================================================

def extract_leaf_candidates(
        mask,
        image):


    contours,_=cv2.findContours(

        mask,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE

    )



    leaves=[]


    leaf_id=0



    for contour in contours:


        area=cv2.contourArea(
            contour
        )


        # remove tiny objects
        # mainly centre leaves/noise

        if area < 100:

            continue



        if len(contour)<5:

            continue



        ellipse=cv2.fitEllipse(
            contour
        )


        (cx,cy),(major,minor),angle=ellipse



        elongation=major/(minor+1)



        # leaves are elongated
        # circles are removed

        if elongation < 1.4:

            continue




        leaf_mask=np.zeros_like(
            mask
        )


        cv2.drawContours(

            leaf_mask,

            [contour],

            -1,

            255,

            -1

        )



        leaf=LeafState(
            leaf_id
        )



        leaf.area=area


        leaf.centroid=(

            float(cx),

            float(cy)

        )


        leaf.direction=angle


        leaf.mask=leaf_mask



        leaf.colour=extract_colour_features(

            image,

            leaf_mask

        )



        leaves.append(
            leaf
        )


        leaf_id+=1



    return leaves





# ==================================================
# ASSIGN LEAVES TO PLANTS
# ==================================================

def assign_leaves_to_plants(
        leaves,
        plants):


    for plant in plants:

        plant.leaves=[]

        plant.mask=np.zeros(

            (MASK_HEIGHT,MASK_WIDTH),

            np.uint8

        )


    for leaf in leaves:


        lx,ly=leaf.centroid



        distances=[]


        for plant in plants:


            px,py=plant.centre


            d=np.sqrt(

                (lx-px)**2+

                (ly-py)**2

            )


            distances.append(
                d
            )



        index=np.argmin(
            distances
        )


        plants[index].leaves.append(
            leaf
        )


        plants[index].mask=cv2.bitwise_or(

            plants[index].mask,

            leaf.mask

        )



    for plant in plants:

        plant.area=cv2.countNonZero(
            plant.mask
        )


    return plants





# ==================================================
# AREA CALCULATION
# ==================================================

def calculate_leaf_features(plants):


    output=[]


    for plant in plants:


        data={


            "plant_area":
                plant.area,


            "leaf_count":
                len(plant.leaves),


            "leaf_areas":
                [

                leaf.area

                for leaf in plant.leaves

                ],



            "directions":
                [

                leaf.direction

                for leaf in plant.leaves

                ]

        }


        output.append(
            data
        )


    return output





# ==================================================
# MAIN PROCESSING LOOP
# ==================================================

def process_experiment(
        folder,
        tray_corners):


    images=sorted(

        glob.glob(

            os.path.join(
                folder,
                "*.jpg"
            )

        )

    )



    centres=generate_pot_centres()



    plants=[

        PlantState(c)

        for c in centres

    ]



    background=None


    first=True


    results={}



    final_image=None

    final_mask=None



    for image_path in images:


        print(
            "Processing:",
            image_path
        )


        tray=load_and_rectify(

            image_path,

            tray_corners

        )



        if first:


            background=create_background_model(

                images,

                tray_corners

            )


            first=False

            continue




        probability=create_leaf_probability(

            tray,

            background

        )



        mask=probability_to_mask(
            probability
        )



        mask=clean_mask(
            mask
        )



        mask=remove_grid_lines(

            mask,

            tray

        )



        leaves=extract_leaf_candidates(

            mask,

            tray

        )



        plants=assign_leaves_to_plants(

            leaves,

            plants

        )



        results[

            os.path.basename(
                image_path
            )

        ]=calculate_leaf_features(
            plants
        )



        final_image=tray.copy()


        final_mask=np.zeros_like(
            mask
        )



        for plant in plants:


            final_mask=cv2.bitwise_or(

                final_mask,

                plant.mask

            )



    return (

        results,

        final_image,

        final_mask,

        plants

    )





# ==================================================
# VISUALISATION
# ==================================================

def create_assignment_image(
        plants):


    output=np.zeros(

        (
        MASK_HEIGHT,
        MASK_WIDTH,
        3
        ),

        np.uint8

    )


    rng=np.random.default_rng(
        42
    )


    for i,plant in enumerate(plants):


        colour=rng.integers(

            60,

            255,

            3,

            dtype=np.uint8

        )



        output[
            plant.mask>0
        ]=colour



        cv2.putText(

            output,

            str(i+1),

            plant.centre,

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (255,255,255),

            3

        )



    return output





# ==================================================
# RUN
# ==================================================

if __name__=="__main__":


    tray_corners=[

        (190,900),

        (2222,900),

        (2222,2166),

        (190,2166)

    ]



    folder="dataset/Experiment1"



    results,image,mask,plants=process_experiment(

        folder,

        tray_corners

    )



    assignment=create_assignment_image(
        plants
    )



    plt.figure(
        figsize=(18,6)
    )



    plt.subplot(1,3,1)

    plt.title(
        "Rectified image"
    )

    plt.imshow(

        cv2.cvtColor(

            image,

            cv2.COLOR_BGR2RGB

        )

    )

    plt.axis("off")




    plt.subplot(1,3,2)

    plt.title(
        "Leaf segmentation"
    )

    plt.imshow(
        mask,
        cmap="gray"
    )

    plt.axis("off")




    plt.subplot(1,3,3)

    plt.title(
        "Plant leaf assignment"
    )

    plt.imshow(
        assignment
    )

    plt.axis("off")



    plt.show()