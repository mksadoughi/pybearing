# pybearing
This code has been written for fault detection of rolling element bearings using a physics based deep learning approach.
Similar to other data-driven approaches, the supervised fault detection model needs to be trained at the offline stage using the training data and then, at the online stage, the trained model can be used for fault detection. The input training data is time-series vibration signal. See Ref [1] for more description of the implemented approach.

The required toolkits:
```python
NumPy SciPy TensorFlow Scikit-learn
```

## Getting started

Import the ```pybearing``` class as pybearing.
```python
from pybearing import pybearing 
```
Build the object named as ```"model"``` to load all the attributes defined in pybearing.
```python
model = pybearing()
```
Before processing data and training a deep learning model, we need to load or prepare training data. 

1. if you have the data ready you can use the ```"load data"``` attribute, which is explained as follow: 

Define the directory of the data (note that data should be in numpy array format).

```python
dir_data = 'D:\Dropboxfiles\Dropbox\Github_codes\Bearing fault detection\data.npy'
```
Data should have the standard shape of (n by m), where n is the number of samples, the first m-2 columns are the 
actual vibration signal, the (m-1)th column is the average speed of the bearing in Hz and the last column is the label of data. label 1 is for the faulty bearing and label 0 is for the healthy bearing condition.

```python
import numpy as np
fs = 20000                             # Sampling rate of the training data (Hz)
```
Use the ```"load_data"``` attribute to load the data. 

```python
data = model.load_data(fs, dir_data)
```
 
2. Assuming that we do not have any real data, we can use ```"signal_gen"``` attribute to generate a simulated dataset for both healthy and faulty conditions. Please see Ref [2] that explains the process of generating sample vibration signal.

Define the input parameters for generating the sample vibration data.

```python
time_tot = 3         # signal time duration 
fs = 20000           # signal sampling rate                   
oc = 3               # order characteristic frequency (fault characteristic frequency/bearing frequency)
fn = 3000            # natural frequency of the system 
decay_f = 1          # decay factor        
n_samples = 50       # total number of samples to generate (for both healthy and faulty classes)
speed_range = np.array([5,10])        # the range of bearing speed. the bearing speeds will e generated randomly in this range
```

Call the ```"signal_gen"``` attribute to generate the sample data.

```python
data = model.signal_gen(time_tot, fs, oc, fn, decay_f, n_samples, speed_range)
```

After generating or loading the vibration data, we need to pre-process the signals in order to reduce the noise-to-signal level. To this end, we can use the ```"signal_analyser"``` attribute. In many real-world condition, the number of available samples are not enough. Therefore, this function also implements a data augmentation technique in order to increase the size of data.   

Define the input parameters for pre-processing the vibration data.

```python
saveing_size =1000         # the size of each data after processing
samples = 20000            # the length of each sub signal in the data augmentation process
stride = 10000             # the stride size between each to sequential sub signals. by decreasing this size, the number of sub signals will increase. 
upfactor = 5               # upfactor for the computed order tracking process 
```

```python
processed_data = model.signal_analyser(data, saveing_size, samples, stride, upfactor)
```
After pre-processing the vibration data, we can fit deep learning model. 

Defines the parameters for the CNN model.

```python
validation_ratio = 0.25      # the ratio for dividing the data into training and validation
learning_rate = 0.0001       # the learning rate for the optimization process
num_classes = 2              # number of classes. in our case we have two classes (healthy and faulty)
training_epochs = 50         # number of training epochs
batch_size = 128             # batch size
display_step = 1             # number of epoch steps for showing the model results
model_dir = r"C:\Users\sadoughi\Downloads\save_net.ckpt"   # the location for saving the model
```
Now, we can use the ```"fit"``` attribute to fit the model.

```python
model.fit(processed_data, validation_ratio, learning_rate, num_classes, training_epochs, batch_size, display_step, model_dir)
```
By running the above code the trained model will be save as ```ckpt``` in the directory defined as model_dir. 


The trained model can be used for fault detecion using a new set of vibration data. 

Similar to explained before, we can  load the testing data or generate new vibration data using the following lines of codes.

```python
data = model.signal_gen(time_tot, fs, oc, fn, decay_f, n_samples, speed_range)
processed_data = model.signal_analyser(data, saveing_size, samples, stride, upfactor)
```
Now, we can perform the fault detection on the new vibration dataset based on the trained model saved in ```model_dir``` directory. 

```python
model_dir = r"C:\Users\sadoughi\Downloads\save_net.ckpt"   # the location for saving the model
model.predict(processed_data, num_classes, model_dir)
```
## Reference
1. Sadoughi, Mohammakazem, Austin Downey, Garrett Bunge, Aditya Ranawat, Chao Hu, and Simon Laflamme.
"A Deep Learning-based Approach for Fault Diagnosis of Roller Element Bearings." (2018). 
2. McFadden, P. D., and J. D. Smith. "Model for the vibration produced by a single point defect in a rolling element bearing." 
Journal of sound and vibration 96, no. 1 (1984): 69-82.
