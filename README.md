# EraGenes-Prediction

## Introduction
EraGenes is basically the combination of two latin words <b>Era</b> means <i>age</i> and <b>Genes</b> means <i>gender</i>.
<br>
This project is basically about <b>Extracting</b> the basic key features from Human faces in <b><i>Real time</i></b>.

Use this link to download this project directly from google drive which consist of 5 minutes of demo vedio in high resolution:

        https://drive.google.com/open?id=16TT7O2Kb20PA_Fw46uqiIIFpt1L7GNtC
# Project Architecture

    1.Introduction
    2.Project Specification
    3.Installing Dependencies
    4.Strategy
    5.Python files
    6.How to run
    7.Output Snapshot
    
# Project Specification:
## Built on
    
      1.python3
      2.Linux based Platform (works on windows too).
      3.Requires high level of computational processor.
      4.must have updated gnu-g++ compiler.
      5.Cmake (latest version)
      
## Requirements 

      1.scikit-learn
      2.dlib
      3.matplotlib
      4.opencv(cv2)
      5.keras
      6.math
      7.h5py
      
# How to install Dependencies
Before installing the dependencies i would suggest to update and uprade to your system by the following command:

    sudo apt-get update && dist-ugrade;
## Method 1
use of pip command

    1.pip install package_name --upgrade
    
## Method 2
use the below command

    pip install -r spec.txt
The above command will install all the required dependencies using spec.txt file.

# Project Strategy
## Face Detection
The first step is to detect the face in the given Image or a frame.<br>The best way is to Use Haarcascade classifier algorithm .

![alt_tag](http://www.jamesshorten.com/images/FindMouthCentreFlowChart.png)

The algorithm itself can be hard to understand but actually it is following the basic strategy i.e finding a feature pixel by pixel.<br>
wheh it finds the first features it goes for onother one with more confidence .the algorithm runs in a loop until frame size is finshed.if all the facial features are found then it returns the coordinates <b>TOP LEFT AND BOOTOM RIGHT</b>.

The working of the above algorithm looks somewhat like this:<br>
![alt_tag](https://memememememememe.me/assets/posts/training-haar-cascades/haarFace.jpg)<br>
Now the face is been detected its time to move the next strategy.

## Extract Facial Features 
To extract the facial features we are using convolution neural network.It is a advanced machine learning algorithm that is used in the feild of computer vision.<br>
![alt_tag](https://image.slidesharecdn.com/paper-presentation3-140114164750-phpapp02/95/neuroevolution-and-deep-learing-4-638.jpg?cb=1389718191)

This features are used to define the age and gender classification.

## Facial Landmark Detection
What is facial key points and landmarks?

    ans
 ![alt_tag](https://ars.els-cdn.com/content/image/1-s2.0-S0957417415004170-gr1.jpg)
# Python Files
This section is about the different python files used for building the project.
        $ipython3 dataset.py
this python file is used to build a raw dataset form bunch of image dataset.
<b><h2>CONFUSED?</h2></b><br>
 Well let me explain you .<br>
 If you take a image you will find out that the each image is set of pixels and each pixels are in the form of <B> [ R, G, B] </B>
 so, it is hard to do the analysis.which brings us to the file.<br>
 this python code follows certain steps:
 
    1.face cropping
    using below python file:
    $ipython3 cropfunc.py
 The image is been cropped and resised to (64 x 64) dimension of image.<br>
![alt_tag](https://github.com/vshantam/Age-Prediction/blob/master/Analysis/Capture2.PNG)
    
    2.grayscale conversion
![alt-tag](https://github.com/vshantam/Age-Prediction/blob/master/Analysis/Capture3.PNG)

    3.wirting the pixels to the dataset
![alt_tag](https://github.com/vshantam/Age-Prediction/blob/master/Analysis/Screenshot%20(20).png)

Now that we have created our dataset it is the time for machne learnng analysis for that we are using the follwing python code:

        $clf.py
        
## Analysis
dataset variation with pixels<br>

![alt_tag](https://github.com/vshantam/Age-Prediction/blob/master/Analysis/Figure_3.png)

![alt_tag](https://github.com/vshantam/Age-Prediction/blob/master/Analysis/Figure_2.png)


Pie chart<br>

![alt_tag](https://github.com/vshantam/Age-Prediction/blob/master/Analysis/Figure_1.png)


Machine learning (Regression)<br>
    
        X axis : Learning Acuuracy
        Y axis : Learning Loss
![alt_tag](https://github.com/vshantam/Age-Prediction/blob/master/Gender/gplot1.png)

# How to run
Before running the program make sure you have provided the corect path of haarcascade and the other classifiers.<br>
Here are the lis of classifier used in the project :

    1.haarcascade_eye.xml
    2.haarcascade_frontalface_default.xml
    # 1 and two used in demo.py file and main.py file
    3.gclf.pkl 
    # used in main.py file
    4.shape_predictor_68_face_landmarks.dat used in main and demo.py file
    5 pretrained model file used in demo.py file.\
<b> Note :</b>do not need to change the path of file except the demo.py file if you want to use the pretrained files.

finally to run the file make sure you have updated system with installed all the dependencies with webcam connected.<br>
run the below command:

    $ ipython3 demo.py
if all went well you will see the desired responses.<br>
<b>NOTE:</b> I am using ipython3 version to run the program .if you want you can use others too such as : python,python3 etc.

# Output snapshot

If all went well you will see the output like below.

![alt_tag](https://github.com/vshantam/EraGenes-Prediction/blob/master/output/output1.png)

![alt_tag](https://github.com/vshantam/EraGenes-Prediction/blob/master/output/output2.png)

As you can see the output is working with slightly variance, this can be happening because of some reasons such as :

        1.Bad Training.
        2.Bad Feature Selection.
        3.Noise.
        
But as you can see the facial key points are very much accurate.when i kept my mouth closed , fitted properly but when i tried to open it,the facial key points moved accordingly .<br>
I guess that is the beauty of Machine intelligence .
