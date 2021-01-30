This project aims to make the Video Conference meetings more automative, this software:

    1. automatically removes the background of a person(user) for user's privacy. 
    2.detects when a person raises hand to intimate that this person wants to talk something in a discussion. 
    3.End User can also use a virtual pen option. By using option , user can write the text on the screen that other persons in that video call can see, just with finger         movements. So the users don't need to buy any touch screen laptops or any other third party hardware tools to write the text on the screen.

This project is divided into three modules.
  1. Background Removing
  2. Palm Detection
  3. Virtual Pen and Eraser

1.Background Removing:

    i) This Module is built up on Mask R_CNN Model.
    ii) i used the Keras API Embedded with TensorFlow Library.
 
 2.Palm Detection:
    
    i) I used the HOG + SVM algorithm to build this Model.
    ii) I have taken the Dlib Library to implement this algorithm.
    iii) I prepared the dataset for this Model using sliding Window Concept.
    
  3.Virtual Pen and Eraser:
  
      i) I used the OpenCV Library to build this Model.
      ii) I have taken the "Blue Pen Cap" as a reference to draw the lines(writing text on the screen).
      
Note: I am planning to add few more modules that would be helpful to this project to make it evenmore innovative.
