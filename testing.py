import numpy as np
import object_size
import cv2
from sklearn.externals import joblib 
class Classify:
    @staticmethod
    def classifier(originalimg,resizedimg):
            width=[]
            height=[]
            cv2.imshow('img',originalimg)
            cv2.waitKey(0)
            test_image = np.expand_dims(resizedimg, axis = 0)
            prediction=joblib.load('colgatelays.pkl')
            result = prediction.predict(test_image)
            finalvalue=result[0]
            answer = np.argmax(finalvalue)
            if answer==0:
                print('colgate')
                #output_image=cv2.imread(img)
                object_size.getDimensions(originalimg)
                width.extend(object_size.getWidth())
                height.extend(object_size.getHeight())
                print (max(width))
                print (max(height))
            elif answer==1:
                print('Hamam')
                object_size.getDimensions(originalimg)
                width.extend(object_size.getWidth())
                height.extend(object_size.getHeight())
                print (max(width))
                print (max(height))
                
            elif answer==2:
                print('lays')
                #output_image=cv2.imread(img)
                object_size.getDimensions(originalimg)
                width.extend(object_size.getWidth())
                height.extend(object_size.getHeight())
                print (width)
                print (height)
    def imgConverter(imagedata):
        import base64
        from PIL import Image
        from io import BytesIO
        import numpy as np
        def stringToRGB(base64_string):
            imgdata = base64.b64decode(str(base64_string))
            image1 = Image.open(BytesIO(imgdata))
            return cv2.cvtColor(np.array(image1), cv2.COLOR_BGR2RGB)        
        originalimg=stringToRGB(imagedata)
        resizedimg=cv2.resize(originalimg,(150,150))
        Classify.classifier(originalimg,resizedimg)
   
originalimg=cv2.imread('demo.jpg')
resizedimg=cv2.resize(originalimg,(150,150))
Classify.classifier(originalimg,resizedimg)
