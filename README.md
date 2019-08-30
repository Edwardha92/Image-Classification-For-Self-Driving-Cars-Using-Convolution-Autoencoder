# Image-Classification-For-Self-Driving-Cars-Through-Convolution-Autoencoder
Master Thesis
<div>
  <img src="http://benchmark.ini.rub.de/Images/11.png" width = 10%/><img src="http://benchmark.ini.rub.de/Images/3.png" width = 10%/><img src="http://benchmark.ini.rub.de/Images/7.png" width = 10%/><img src="http://benchmark.ini.rub.de/Images/00032_00005.jpg" width = 10%/><img src="http://benchmark.ini.rub.de/Images/6.png" width = 10%/><img src="http://benchmark.ini.rub.de/Images/00029_00018.jpg" width = 10%/><img src="http://benchmark.ini.rub.de/Images/9.png" width = 10%/><img src="http://benchmark.ini.rub.de/Images/12.png" width = 10%/><img src="http://benchmark.ini.rub.de/Images/00027_00022.jpg" width = 10%/>
<h3>Overview about the Dataset:</h3>
<ul>
  <li>Single-image, multi-class classification problem</li>
  <li>More than 40 classes</li>
  <li>More than 50,000 images in total</li>
  <li>Large, lifelike database</li>
  <li>Reliable ground-truth data due to semi-automatic annotation</li>
  <li>Physical traffic sign instances are unique within the dataset	(i.e., each real-world traffic sign only occurs once) </li>
  </ul>
 </div>
 <div>
<h3>Image format:</h3>
  <ul>
    <li>The images contain one traffic sign each</li>
    <li>Images contain a border of 10 % around the actual traffic sign (at least 5 pixels) to allow for edge-based approaches</li>
    <li>Images are stored in PPM format (Portable Pixmap, P6)</li>
    <li>Image sizes vary between 15x15 to 250x250 pixels</li>
    <li>Images are not necessarily squared</li>
    <li>The actual traffic sign is not necessarily centered within the image. This is true for images that were close to the image border in the full camera image</li>
  </ul>
  </div>
  
<a href="http://benchmark.ini.rub.de/Dataset/GTSRB_Python_code.zip">Link </a> to download <b>the GTSRB Dataset (German Traffic Sign Recognition Benchmark)</b>

I use in my work images 32x32 pixels and just 10 classes (the most frequent 10 classes). Therefor i should firstly the dataset preprocess, then load the dataset.
<div>
<ol>
  <li>Load the training und test Dataset using <a href="https://github.com/Edwardha92/Image-Classification-For-Self-Driving-Cars-Through-Convolution-Autoencoder/blob/master/Source%20Code/readTrafficSigns.py"> readTrafficSigns.py</a> and convert it into 32x32x3 size, then split the training data in training und validation data. At the end save it into three arrays (train, valid, test)</li>
  <li>Using <a href = "https://github.com/Edwardha92/Image-Classification-For-Self-Driving-Cars-Through-Convolution-Autoencoder/blob/master/Source%20Code/Classes/utils.py">utils.py</a> to explor the Dataset and see the Histogram of Data in train und valid Data</li>
  <li>Reshape the image data into rows and apply the normalization on the data by division all data on 255 </li>
  <li>Train the filters using the <a href= "https://github.com/Edwardha92/Image-Classification-For-Self-Driving-Cars-Through-Convolution-Autoencoder/blob/master/Source%20Code/Classes/ConvoultionAutoEncoder.py">Convolution_Autoencoder.py</a> then plot the evolution of the mean squre error</li>
  <li>Make a visualization to the results of Convolution Autoencoder using the <a herf= "https://github.com/Edwardha92/Image-Classification-For-Self-Driving-Cars-Through-Convolution-Autoencoder/blob/master/Source%20Code/Classes/Visualization.py">Visualization</a> class</li>
  <li>Befor training the classifier the features will extracted</li>
  <li>Then the Classifier will trained. In this work it'll used SVM (Support Vector Machine). The modell tested with tow type of SVM. One of it is <a href = "https://github.com/Edwardha92/Image-Classification-For-Self-Driving-Cars-Through-Convolution-Autoencoder/blob/master/Source%20Code/Classes/OVA_SVM.py">OVA_SVM</a> and the other is SVC form scikit learn.</li> 
  <li>After Training the classifier it'll tested using the validation and the test Dataset to make prediction to the test Label. </li>
  <li>At the end the results displayed using the confiusion matrix</>
  </ol>
</div>
  <h2>Flow Chart</h2>
  <img src="https://user-images.githubusercontent.com/44674760/64015431-d7207300-cb24-11e9-852a-83af927d0381.PNG" width=35% height=35%>
<h2>Die Architecture of the Modell</h2>
  <img src="https://user-images.githubusercontent.com/44674760/64016764-2025f680-cb28-11e9-9320-2aaa5f9052a0.PNG">
<h2>Result of colored images dataset</h2>
  <h3> First layer of convolutional Autoencoder</h3>
  <img src="https://user-images.githubusercontent.com/44674760/64015949-f8359380-cb25-11e9-9d8c-4c350dd1500b.jpg">
  <h3> Second layer of convolutional Autoencoder</h3>
  <img src="https://user-images.githubusercontent.com/44674760/64016030-33d05d80-cb26-11e9-81e8-a30248e80404.jpg">
  <h3> Third layer of convolutional Autoencoder</h3>
  <img src="https://user-images.githubusercontent.com/44674760/64016046-3cc12f00-cb26-11e9-9c57-1d3bf9d14079.jpg">
  
  <h3>OVA_SVM</h3>
  The best accuracy is 79.79% using 1K Batch size 
  <img src="https://user-images.githubusercontent.com/44674760/64016374-223b8580-cb27-11e9-867f-321f4e7ada5c.jpg" width=70% height=70%>
  <h3>SVC</h3>
  The best accuracy is 94.56% using 10K Batch size 
  <img src="https://user-images.githubusercontent.com/44674760/64016388-2b2c5700-cb27-11e9-8c13-e1782819fd70.jpg" width=70% height=70%>
  
  
  <h2>Result of grayscale images dataset</h2>
  <h3> First layer of convolutional Autoencoder</h3>
  <img src="https://user-images.githubusercontent.com/44674760/64016129-77c36280-cb26-11e9-9d51-e4ee26763827.jpg">
  <h3> Second layer of convolutional Autoencoder</h3>
  <img src="https://user-images.githubusercontent.com/44674760/64016151-8578e800-cb26-11e9-929e-44ee3fbd942e.jpg">
  <h3> Third layer of convolutional Autoencoder</h3>
  <img src="https://user-images.githubusercontent.com/44674760/64016176-932e6d80-cb26-11e9-8fd9-a45cef4cf781.jpg">
  
  <h3>OVA_SVM</h3>
  The best accuracy is 65.71% using 1K Batch size 
  <img src="https://user-images.githubusercontent.com/44674760/64016463-64fd5d80-cb27-11e9-9b63-6e85fed9a16d.jpg" width=70% height=70%>
  <h3>SVC</h3>
  The best accuracy is 92.13% using 10K Batch size 
  <img src="https://user-images.githubusercontent.com/44674760/64016497-734b7980-cb27-11e9-8e2c-ae41ae6fd4eb.jpg" width=70% height=70%>
