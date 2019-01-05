#########################################################################      
##   this code performs a fault detection strategy using deep learning         
##   on the rolling element bearing vibration data.                            
##   Form more information on our apraoch and the algorithm please see         
##   our published article:                                                    
##   Sadoughi, Mohammakazem, Austin Downey, Garrett Bunge, Aditya Ranawat,     
##   Chao Hu, and Simon Laflamme. "A Deep Learning-based Approach for Fault    
##   Diagnosis of Roller Element Bearings." (2018).                            
########################################################################


from scipy.signal import hilbert
import numpy as np
from scipy import interpolate
import scipy.signal as sig
from scipy.fftpack import fft
from scipy import integrate
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import random
tf.reset_default_graph()


## the computed order tracking prgram. this code transfers the signal from the time domain to the phase domain. 
def COT(data, speed, upfactor, fs):
    
            samples = np.shape(data)[0]     # extract the number of samples   
            t = np.arange(0.0, samples/fs, 1/fs)    # generate a time step data
            
            # resample the data to the higher sampling rate. This can increase the accuracy of interpolation which 
            # will be used in the next steps. 
            dataUp = sig.resample(data, samples * upfactor)      
            timeUp = np.arange(0.0, samples/fs, 1/(fs*upfactor));
            
            f = interpolate.interp1d(t, speed, fill_value="extrapolate")       # interpolation function for the speed data
            speedUp = f(timeUp)                                                # result of interpolation
            
            phase_int = integrate.cumtrapz(speedUp, timeUp, initial=0)         # build another interpolation function for the phase
            phase = np.arange(0.0, phase_int[-1], phase_int[-1]/(samples*upfactor))  # result  of interpolation function
            
            f = interpolate.interp1d(phase_int, dataUp) 
            data_ordered = f(phase) 
            
            return data_ordered, phase                                         # store the data and its phase 



## This function has been written for generating the random batch of samples from the data for stochastic optimization. 
            # it will be used in the training process of deep learning model
def next_batch(x, y, batch_size):    # x is the input, y is the output and batch_size is the size of batch (number of samples)
    num_elements = x.shape[0]
    new_index = np.arange(num_elements)                                        # generate random index
    np.random.shuffle(new_index)                                               # shuffle data
    new_index = new_index [:batch_size]                                        # select the first batch_size number of samples
    
    
    # the output will be the batch of input and output which have been chosen randomly
    x = x[new_index,:]
    y = y[new_index]
    return x, y


## the function for building a covolutional layer
def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, keepratio, name):
    # shape of input for convolution
    conv_filt_shape = [filter_shape, num_input_channels, num_filters]
    
    # wights and biasses
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')
    out_layer = tf.nn.conv1d(input_data, weights, 1, padding='SAME')
    
    # adding the bias term
    out_layer += bias
    
    # adding the activation layer
    out_layer = tf.nn.relu(out_layer)
    
    # adding a pooling laer
    out_layer = tf.nn.pool(out_layer, [5], 'MAX', 'SAME', strides = [5])
    
    # adding a dropout layer
    out_layer = tf.nn.dropout(out_layer, keepratio)
    
    return out_layer
        
        
        
class pybearing:
    
    def __init__(self):
        return None
    
    ## loading the data with the numpy format
    def load_data(self, fs, dir_data):
        self.fs = fs         
        data = np.load(dir_data)
        self.n_samples = len(data)
        return data

    ## generating the simulated vibration data from the beating with both helthy and faulty condition
    def signal_gen(self, time_tot, fs = 20000, oc = 3, fn = 3000, decay_f = 2, n_samples = 200, speed_range = np.array([10,20])):
        
        
        self.time_tot = time_tot
        self.fs = fs
        self.oc = oc
        self.fn = fn
        self.decay_f = decay_f
        self.n_samples = n_samples
        self.speed_range = speed_range
        
        # generate the time steps
        t = np.linspace(0, self.time_tot, num = self.fs*self.time_tot)
        
        # initialize the data: data.shape = number of samples * Len(t) + 2. the last column of data represents the data label 
        # (e.g. label = 1 for faulty signal and label = 0 for healthy signal). the one before the last column shows the speed data
        data = np.zeros([self.n_samples,len(t)+2])
        
        for k in range(self.n_samples):
            
            # Randomly chose the fault conidtion. label = 1: faulty signal and label = 0: healthy signal
            label = random.randint(0, 1) 
            fr = random.randint(self.speed_range[0], self.speed_range[1])   # randomly select the speed
            
            if label == 1:     # faulty signal
                signal = np.zeros(len(t))    # initialize the signal value
                t_fault = np.linspace(0, self.time_tot, num = self.oc*fr*self.time_tot)
                
                for i in t_fault:
                    signal += np.heaviside(t-i, 1.0)*np.exp(-1*self.decay_f*self.fs*(t-i)**2)   
                    
                signal = signal * np.sin(2*np.pi*self.fn*t) * np.sin(2*np.pi*fr*t)
                signal += 0.2*np.random.normal(0,1,len(t))
                data[k,0:len(t)] = signal
                data[k,len(t)] = fr
                data[k,len(t)+1] = label
            else:   
                signal = np.sin(2*np.pi*self.fn*t) * np.sin(2*np.pi*fr*t)
                signal += 0.2*np.random.normal(0,1,len(t))
                data[k,0:len(t)] = signal
                data[k,len(t)] = fr
                data[k,len(t)+1] = label
                
                
            if 10 * k % self.n_samples == 0: 
                print ("Progress: %03d out of %03d samples have been generated" % (k, self.n_samples))
        print ("All samples have been generated succesfully") 
        return data
    
    ## performing several signal processing techniques to denoise and filter the signal and transfer it to the order domain.
    def signal_analyser(self, data, saveing_size =1000, samples = 2000, stride = 3000, upfactor = 5):
        
        self.saveing_size = saveing_size
        self.samples = samples
        self.stride = stride
        self.upfactor = upfactor  
        
        reference_order = np.linspace(0.0, 10, self.saveing_size)
        
        processed_data = np.zeros([0, self.saveing_size+1])
        signal_len = data.shape[1]-2
        
        for i in range(self.n_samples):        
            # adding the spead data
            speed = data[i,-2]*np.ones(signal_len)
            signal = data[i,0:-2]
            # slicing the signal
            first_index = 0
            last_index = first_index + self.samples
            
            while last_index < signal_len: 
            
                subsignal = signal[first_index:last_index]
                subspeed = speed[first_index:last_index]        
                first_index = first_index + self.stride
                last_index = first_index + self.samples
                # COT analysis 
                signal_ordered = np.empty([subsignal.shape[0],2])
                signal_ordered[:,1], signal_ordered[:,0] = COT(subsignal, subspeed, 1, self.fs)   
                
                # envelope analsis 
                amplitude_envelope = np.empty(signal_ordered.shape)
                amplitude_envelope[:,1] = np.abs(hilbert(signal_ordered[:,1]))
                amplitude_envelope[:,0] = signal_ordered[:,0]   
                         
                #FFT analysis
                orer_s = len(amplitude_envelope[:,0])/(amplitude_envelope[-1,0]-amplitude_envelope[0,0])
                ps_signal_ordered = np.empty([len(amplitude_envelope[:,0])//2, amplitude_envelope.shape[1]])
                ps_signal_ordered [:,0] = np.linspace(0.0, orer_s/2, len(amplitude_envelope[:,0])//2)
                ps_signal = fft(amplitude_envelope[:,1])
                ps_signal_ordered [:,1] = np.abs(ps_signal[0:self.samples//2])  
                
                ps_signal_final = np.empty([self.saveing_size, amplitude_envelope.shape[1]])
                ps_signal_final [:,0] = reference_order
                f = interpolate.interp1d(ps_signal_ordered[:,0], ps_signal_ordered[:,1])
                ps_signal_final[:,1] = f(ps_signal_final [:,0]) 
                new_data = np.concatenate((np.expand_dims( ps_signal_final [:,1], axis=0) , np.expand_dims(np.expand_dims(data[i,-1], axis=0), axis=0)), axis=1)    
                processed_data  = np.concatenate((processed_data, new_data), axis=0) 
            if 10 * i % self.n_samples == 0: 
                print ("Progress: %03d out of %03d samples have been processed" % (i, self.n_samples))   
        print ("All samples have been processed succesfully") 
        return processed_data
                 
             
    ## fit a deep learning model on the pre-processed dataset using CNN.         
    def fit(self, processed_data, validation_ratio, learning_rate, num_classes, training_epochs, batch_size, display_step, model_dir):
        
        self.validation_ratio = validation_ratio
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.display_step = display_step 
        
        ############# Leaving one speed out as testing ###############       
        index_testing = len(processed_data)*self.validation_ratio-1
        index_testing = int(index_testing)
        x_testing = processed_data[:index_testing,0:-1]
        y_testing = processed_data[:index_testing,-1]
        x_training = processed_data[index_testing:,0:-1]
        y_training = processed_data[index_testing:,-1]
        
        onehot_encoder = OneHotEncoder(sparse=False)
        y_training = onehot_encoder.fit_transform(np.expand_dims(y_training,1))
        y_testing = onehot_encoder.fit_transform(np.expand_dims(y_testing,1))
        
        ##############################################################
        num_channels = 1
        input_length = x_training.shape[1]
        
        # declare the training data placeholders
        x = tf.placeholder(tf.float32, [None, input_length, num_channels])
        y = tf.placeholder(tf.float32, [None, self.num_classes])
        keepratio = tf.placeholder(tf.float32)
                
        layer1 = create_new_conv_layer(x, num_channels, 5, 8, 3, keepratio, name='layer1')
        flattened = tf.reshape(layer1, [-1, tf.shape(layer1)[1] * tf.shape(layer1)[2] ])
        
        wd1 = tf.Variable(tf.truncated_normal([1000, 2], stddev=0.03), name='wd1')
        bd1 = tf.Variable(tf.truncated_normal([2], stddev=0.01), name='bd1')
        dense_layer1 = tf.matmul(flattened, wd1) + bd1
        dense_layer1 = tf.nn.relu(dense_layer1)
        
        y_ = tf.nn.softmax(dense_layer1)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = dense_layer1, labels = y))
        optimiser = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cross_entropy)
        
        # define an accuracy assessment operation
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        
        with tf.Session() as sess:            
            sess.run(init_op)
            for epoch in range(self.training_epochs):
                
                avg_cost = 0.
                total_batch = int(len(y_training)/self.batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_x, batch_y = next_batch(x_training, y_training, self.batch_size)
                    batch_x = np.expand_dims(batch_x, axis=2)
                    _, cost = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y, keepratio:1.})
                    avg_cost += cost /total_batch
            
                # Display logs per epoch step
                if epoch % self.display_step == 0: 
                    print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
                    test_acc = sess.run(accuracy, feed_dict={x:np.expand_dims(x_testing, axis=2), y: y_testing, keepratio:1.})
                    print (" Test accuracy: %.3f" % (test_acc))
            saver.save(sess, model_dir)                      
        
    ## predicting the fault detection accuracy on a new data set        
    def predict(self, processed_data, num_classes, model_dir):
        
        self.num_classes = num_classes
        
        ############# Leaving one speed out as testing ###############       
        x_testing = processed_data[:,0:-1]
        y_testing = processed_data[:,-1]
        
        onehot_encoder = OneHotEncoder(sparse=False)
        y_testing = onehot_encoder.fit_transform(np.expand_dims(y_testing,1))
        
        ##############################################################
        
        num_channels = 1
        input_length = x_testing.shape[1]
        # declare the training data placeholders
        x = tf.placeholder(tf.float32, [None, input_length, num_channels])
        y = tf.placeholder(tf.float32, [None, self.num_classes])
        keepratio = tf.placeholder(tf.float32)
        
        layer1 = create_new_conv_layer(x, num_channels, 5, 8, 3, keepratio, name='layer1')
        flattened = tf.reshape(layer1, [-1, tf.shape(layer1)[1] * tf.shape(layer1)[2] ])
        
        wd1 = tf.Variable(tf.truncated_normal([1000, 2], stddev=0.03), name='wd1')
        bd1 = tf.Variable(tf.truncated_normal([2], stddev=0.01), name='bd1')
        dense_layer1 = tf.matmul(flattened, wd1) + bd1
        dense_layer1 = tf.nn.relu(dense_layer1)
        
        y_ = tf.nn.softmax(dense_layer1)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        saver = tf.train.Saver()
        with tf.Session() as sess:   
            saver.restore(sess, model_dir)
            test_acc = sess.run(accuracy, feed_dict={x:np.expand_dims(x_testing, axis=2), y: y_testing, keepratio:1.})
            print (" Test accuracy: %.3f" % (test_acc))          
        
     
        



            
