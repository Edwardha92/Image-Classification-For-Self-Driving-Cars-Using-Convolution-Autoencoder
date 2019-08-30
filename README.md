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
