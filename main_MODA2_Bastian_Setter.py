import numpy as np
import h5py
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.ticker as mticker
import random
from scipy.signal import find_peaks
import itertools
import os
import pickle
from tensorflow import keras
from keras import layers,activations
import tensorflow as tf


def analyse_model(build_model, path, retrain = True, optimizer=keras.optimizers.Adam(3e-05), loss=keras.losses.Huber(),
                  metrics=[keras.metrics.MeanSquaredError()], include_pure_noise=False, digitize_signal=True, epochs=30, batchsize=50):
    '''
    Handels the whole process of 
    creating datasets, 
    training the provided model, 
    training to its optimal epoch 
    and analysing the optimally trained model.
    '''
    #creating paths to save files later
    paths = [path+'/plots/', path+'/30_model/', path+'/adjusted_model/']
    for single_path in paths:
        if not os.path.isdir(single_path):
            os.makedirs(single_path)
    #creating empty array to save estimators
    savearray = np.zeros((5,))

    #load data, quntize simulations and create train, test and validtion set
    noise, signal = load_signal_noise()
    noise, signal, s_dig = quantize_signal(noise, signal)
    x_train, y_train = create_set(noise, s_dig, signal, 0, 25600 *0.7, 70000, digitize_signal, include_pure_noise)
    x_test, y_test = create_set(noise, s_dig, signal, 25600*0.7,25600*0.88, 12000, digitize_signal, include_pure_noise)
    x_validate, y_validate = create_set(noise, s_dig, signal, 25600*0.88, 25600, 8000, digitize_signal, include_pure_noise)

    
    build_model.summary()
    temp_weights = build_model.get_weights()#saves starting weights, to allow for retraining later
    #compile and train model according to inputs
    build_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    history = build_model.fit(x_train, y_train, batch_size=batchsize,epochs=epochs, validation_data=(x_validate, y_validate), shuffle=True)

    # save model and history
    build_model.save(paths[1])
    with open(paths[1]+'/history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    # determine best epoch and save the estimator calucalted in the process
    optimal_epoch,savearray[0] = determine_best_epoch(history.history)
    savearray[3] = optimal_epoch
    
    #generate plot of the first training
    plot_history(history.history,optimal_epoch,save_path=f'{paths[0]}Long_')

    
    if optimal_epoch <0.90*epochs and retrain: # retrain to best epoch if overfitting can be expected
        
        build_model.set_weights(temp_weights)#load weights saved earlier
        #comile and train again with same parameters, but optimal_epoch
        build_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        history = build_model.fit(x_train, y_train, batch_size=batchsize,epochs=optimal_epoch, validation_data=(x_validate, y_validate), shuffle=True)
    
    #save history of retrain. If no retrain has taken place save old history twice
    with open(paths[2]+'/history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)   
    

    #save best model and note mean squared errors of trained networks
    #last state of retrain
    retrain_evaluation = [history.history['val_mean_squared_error'][-1]]
    build_model.save(paths[2])

    for i in range(5):#train for 5 more epochs
        hist = build_model.fit(x_train, y_train, batch_size=batchsize,epochs=1, validation_data=(x_validate, y_validate), shuffle=True)
        hist = hist.history
        retrain_evaluation.append(hist['val_mean_squared_error'][-1])#add mean squared error to list for estimator calculation later
        if hist['val_mean_squared_error'][-1]<np.amin(retrain_evaluation):#save model over previous, if better performance is acchieve
            build_model.save(paths[2])
        
    
    savearray[1] = middle_mean(retrain_evaluation)#save estimater calculate on the base of the mean squared error saved for the last 6 epochs
    
    
    build_model = keras.models.load_model(paths[2])#load the best model trained for analysis

    if optimal_epoch <0.90*epochs and retrain:#plot history of retrain if retrain has occured
        plot_history(history.history, save_path=f'{paths[0]}Retrain_',only_mse=False)

    for i in range(3):#plot three signals along with their reconstructions
        plot_random_predict(x_test, y_test, build_model, i, verbose=False, save_path=paths[0])

    analyse_network(x_test, y_test, build_model, save_path=paths[0])#generate extensive graphical analysis

    savearray[2] = build_model.evaluate(x = x_test,y = y_test)[-1]#evaluate the optimal model on the test data set and save the estimator

    np.savetxt(f'{path}/eval_numers.txt',savearray)#first_val_opt,sec_val_end, sec_test_end,opt_epoch


    # plot history


def load_signal_noise(noisefile = "./noise.hdf5", simfile = "./simulation.hdf5"):
    '''
    Loads the noise and simulation files provided and returns their contents
    '''
    f_noise = h5py.File(noisefile)
    f_signal = h5py.File(simfile)

    noise = f_noise["traces [mV]"][:]
    time_noise = f_noise["times [ns]"][:]

    signal = f_signal["traces [mV]"][:]
    time_signal = f_signal["times [ns]"][:]
    return noise, signal

def quantize_signal(noise, signal, digstep = 0.59171598,seed = 1236):
    '''
    Quantizes signal in steps given by digistep and 
    shuffles noise, signal and quantized signal with a given seed 
    to ensure random but matching datasets
    '''
    s_new = signal.copy()
    s_dig = np.round(s_new/digstep)*digstep

    np.random.seed(seed)
    np.random.shuffle(noise)
    np.random.seed(seed)
    np.random.shuffle(signal)
    np.random.seed(seed)
    np.random.shuffle(s_dig)
    return noise, signal, s_dig

def create_set(noise, dig, sig, start, end, mininum_samples=100000, digitize_signal=True, include_pure_noise=False):
    '''
    Creates specified number input and output samples.

    '''
    #slice noise, signal and quntized signal
    start = int(start)
    end = int(end)
    noise = noise[start:end]
    sig = sig[start:end]
    dig = dig[start:end]

    #calculate often much the signals will be used to provided the requested set size
    #and allocate space for data set
    if include_pure_noise == 'equal':#'equal' requestes the data set to contain pure noise and events to even parts
        repeats = int(0.5*mininum_samples/(end-start))+1
        xx = np.empty((repeats*(end-start), len(noise[0])))

        for counter in range(repeats):#add pure noise to second half of the array
            xx[(counter+repeats)*(end-start):(counter+1+repeats)*(end-start)] = noise

    else:
        repeats = int(mininum_samples/(end-start))+1
        xx = np.empty((repeats*(end-start), len(noise[0])))

        if include_pure_noise == 'once':#add one set of pure noise traces to the data set
            xx[(repeats)*(end-start):] = noise
            repeats -= 1

    for counter in range(repeats):# add input samples to the data set 
        if digitize_signal:#digitized or not as requested
            xx[counter*(end-start):(counter+1)*(end-start)] = dig + np.roll(noise, counter)
        else:
            xx[counter*(end-start):(counter+1)*(end-start)] = sig + np.roll(noise, counter)

    # add 0 array as true reconstructions to output array en pair to the input samples
    if include_pure_noise == 'once':
        yy_noise = list(np.zeros_like(noise))
    elif include_pure_noise == 'equal':
        yy_noise = list(np.zeros_like(noise))*repeats
    else:
        yy_noise = []

    yy = list(sig)*repeats+yy_noise #add signals to output array

    return xx, np.array(yy)

def determine_best_epoch(history, times = 3, required_progress = 0.1):
    '''
    Determine the best epoch in the provided train
    and calculate the middle mean at the optimal epoch for the validation history'''
    #load processed histories
    processed_history = smooth_and_seperate_history(history, targets=['val_mean_squared_error','adjusted'],times = times,required_progress = required_progress)
    optimal_epoch = np.argmin(processed_history['adjusted_val_mean_squared_error'])#determine optimal epoch

    smooth_val_mean_squared_error = processed_history['smooth_val_mean_squared_error']
    if optimal_epoch < 2:#calculate the middle mean, taking into account the edges of the training
        output = middle_mean(smooth_val_mean_squared_error[:5])
    elif optimal_epoch >len(smooth_val_mean_squared_error)-3:
        output = middle_mean(smooth_val_mean_squared_error[-5:])
    else:
        output = middle_mean(smooth_val_mean_squared_error[optimal_epoch-2:optimal_epoch+3])

    return optimal_epoch, output

def smooth_and_seperate_history(history, times=3,targets = ['mean_squared_error','val_mean_squared_error','adjusted'],required_progress = 0.1):
    '''
    extracted wanted parts of history
    and smooth them
    and adjust by epoch penalty if necessary
    '''
    processed_history = {}
    smooth_width=int(len(history[targets[0]])/20)#calculate kernel size to smooth data with based on trace length

    for target in targets:#extract every history trace required
        if target == 'adjusted':# adjusts the previously loaded trace
            processed_history[f'adjusted_{save_target}'] = adjust_history_trace(processed_history[f'smooth_{save_target}'],required_progress = required_progress)
        else:
            processed_history[target] = history[target]#save true history trace
            unnormalised_smooth = smooth_curve(history[target],smooth_width,times = times)#smooth true history trace
            processed_history[f'smooth_{target}'] = unnormalised_smooth/unnormalised_smooth[-1]*history[target][-1]#normalise smoothed trace
            save_target = target
    return processed_history

def smooth_curve(data,pad_loss=3,type = 'flat',times = 1):
    '''
    Smooth provided array through convolution with a kernel
    '''

    data = np.array(data).copy()
    #generete kernel
    if type == 'pyramid': 
        kernel = np.concatenate([np.arange(1,pad_loss+2),np.arange(start = pad_loss, stop= 0,step=-1)])
    elif type == 'flat':
        kernel = np.ones(pad_loss*2+1)
    else:
        print('Not a valid kernel type. Use \'pyramid\' or \'flat\'.')
        raise Exception

    # repeat smoothing for given amount of times
    for i in range(times):
        data = np.pad(data, pad_loss, mode='edge')#pad with respective edge values to keep shape in convolution
        data = np.convolve(data, kernel, mode='valid')
    return data

def adjust_history_trace(history_trace,required_progress = 0.1):
    '''
        punish history for given amount
    '''
    hist_len = len(history_trace)
    progress_penalty_base = np.full((hist_len,),np.power(required_progress+1,1/hist_len))#break down total penalty for 1 epoch
    progress_penalty = np.power(progress_penalty_base,np.arange(hist_len))#scale penalty to any epoch
    return history_trace*progress_penalty

def middle_mean(numbers):
    '''
        Method used to calculate mean of array, while reducing impact of outliers
    '''
    if np.isscalar(numbers):
        return numbers
    else:
        #calculate mean of array, without the best and worst value
        num = np.sort(np.array(numbers))
        return np.mean(num[1:-1])

def plot_history(history, best_epoch = None,only_mse=True, plot=False, save_path=False):
    '''
    Plot training history mean squared error.
    Best epoch and loss can be added to the plot
    '''
    if only_mse:#only plot mse
        processed_history = smooth_and_seperate_history(history)
        plt.figure(figsize=(5,3))
        plt.plot(processed_history['val_mean_squared_error'],label='Validation')
        plt.plot(processed_history['mean_squared_error'],label='Train')
        plt.plot(processed_history['smooth_val_mean_squared_error'],label='Smooth validation',lw =2)
        plt.plot(processed_history['smooth_mean_squared_error'],label='Smooth train',lw =2)
        if best_epoch:
            plt.plot(processed_history['adjusted_val_mean_squared_error'],label='Adjusted smooth validation',lw =2)
            plt.axvline(best_epoch,c = 'k',lw =3,label='best epoch')
        plt.yscale('log')
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel('Mean squared error')
        plt.legend(loc='upper right')

    else:#plot mse and loss
        processed_history = smooth_and_seperate_history(history,targets=['mean_squared_error','val_mean_squared_error','adjusted','loss','val_loss'])
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 3))
        ax0.plot(processed_history['val_mean_squared_error'],label='Validation')
        ax0.plot(processed_history['mean_squared_error'],label='Train')
        ax0.plot(processed_history['smooth_val_mean_squared_error'],label='Smooth validation',lw =2)
        ax0.plot(processed_history['smooth_mean_squared_error'],label='Smooth train',lw =2)
        if best_epoch:
            ax0.plot(processed_history['adjusted_smooth_val_mean_squared_error'],label='Adjusted smooth validation',lw =2)
            ax0.axvline(best_epoch,c = 'k',lw =3,label='best epoch')
        ax0.set_ylabel('Mean squared error')
    
        ax1.plot(processed_history['val_loss'],label='Validation')
        ax1.plot(processed_history['loss'],label='Train')
        ax1.plot(processed_history['smooth_val_loss'],label='Smooth validation',lw =2)
        ax1.plot(processed_history['smooth_loss'],label='Smooth train',lw =2)
        ax1.set_ylabel('Loss')

        for ax in (ax0,ax1):
            ax.set_xlabel('Epoch')
            ax.legend(loc='upper right')
            ax.set_yscale('log')
            ax.grid()

    #save and clean up figure in memory
    if save_path:
        plt.savefig(f'{save_path}Metrics.pdf', bbox_inches='tight')
    if plot:
        plt.show()
    else:
        plt.close()

def plot_random_predict(x_test, y_test, model, seed=0, verbose=True, save_path=False):
    '''
    Plot the input, output and reconstruction of a random test sample based on seed
    '''
    #determine sample
    times = np.linspace(0, 640, 2048)
    random.seed(seed)
    pos = random.randint(0, x_test.shape[0])

    #evaluate sample
    x = x_test[pos:pos+1]
    y_real = y_test[pos]
    y_pred = model.predict(x).reshape(x_test.shape[1])

    if verbose:
        print(pos)

    #plot sample
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].set_ylabel('Amplitude [mV]')
    
    axes[0].plot(times, x.reshape(x_test.shape[1]), label='Input')
    axes[1].plot(times, y_real, label='Real signal')
    axes[2].plot(times, y_pred, label='Prediction')
    for ax in axes:
        ax.set_xlabel('Time [ns]')
        ax.legend()
    if save_path:
        plt.savefig(f'{save_path}/CompareSignal_{pos}.pdf',bbox_inches='tight')
    if verbose:
        plt.show()
    else:
        plt.close()

def analyse_network(x_test, y_test, trained_model, threshold=1/np.e, plot=False, save_path=False, split_value = 17.82):
    #construct the graphical analysis for a model 
    
    #calculate the 5 metrics amplitude, center, begin, duration and peak amount for test data set
    stats = calculate_stats(x_test, y_test, trained_model, threshold, split_value)


    fig, axes = plt.subplots(4, 5, figsize=(15, 11), sharex='col')
    fig = tight_pairs(5,fig)# adjust positions of subplots
    fig.text(0.065,0.313,'Metric histograms [1000]',rotation='vertical',va='center',ha='right',size=12)
    fig.text(0.065,0.714,'Correlation plot',rotation='vertical',va='center',ha='right',size=12)
    fig.text(0.065,0.714,'Reconstructed over true signal',rotation='vertical',va='center',ha='left',size=12)
    
    names = ['Amplitude [mV]', 'Center [ns]','Begin [ns]', 'Duration [ns]', 'Peak amount']
    for column,name in enumerate(names):
        #generate histogram bins
        min_bin = np.amin([np.amin(stats['all_signal'][column]),np.amin(stats['all_signal'][column])])
        max_bin = np.amax([np.amax(stats['all_signal'][column]),np.amax(stats['all_signal'][column])])
        if column == 0:
            bins = np.logspace(np.log10(min_bin), np.log10(max_bin), 30)
        elif column == 4:
            bins = np.arange(min_bin-0.5, max_bin+1.5)
        else:
            bins = np.linspace(min_bin, max_bin, 30)

        #plot correlation plots
        axes[0, column].plot(stats['all_signal'][column], stats['all_recon'][column], 'o', ms=1 ,color ='C0')
        axes[0, column].plot(stats['low_signal'][column], stats['low_recon'][column], 'o', ms=1 , color ='C2')

        axes[1, column].plot(stats['all_signal'][column], stats['all_recon'][column], 'o', ms=1 , color ='C0')
        axes[1, column].plot(stats['high_signal'][column], stats['high_recon'][column], 'o', ms=1 ,color ='C1')
        axes[1, column].xaxis.set_tick_params(labelbottom=True)

        #plot histograms
        axes[2, column].hist(stats['low_signal'][column], bins=bins,histtype='step',color ='C4')
        axes[2, column].hist(stats['low_recon'][column], bins=bins,histtype='step',color ='C3')
        y_vals = uncolide_plots(axes[2, column])
        axes[2, column].yaxis.set_major_locator(mticker.FixedLocator(y_vals))
        axes[2, column].set_yticklabels(['{:1.1f}'.format(x /1000) for x in y_vals])

        axes[3, column].hist(stats['high_signal'][column], bins=bins,histtype='step',color ='C4')
        axes[3, column].hist(stats['high_recon'][column], bins=bins,histtype='step',color ='C3')
        y_vals = uncolide_plots(axes[3, column])
        axes[3, column].yaxis.set_major_locator(mticker.FixedLocator(y_vals))
        axes[3, column].set_yticklabels(['{:1.1f}'.format(x /1000) for x in y_vals])
        axes[3, column].set_xlabel(name)

    axes[0, 0].set_ylabel('Low amplitude signals')
    axes[1, 0].set_ylabel('High amplitude signals')
    axes[2, 0].set_ylabel('Low amplitude signals')
    axes[3, 0].set_ylabel('High amplitude signals')

    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_xscale('log')
    #draw combined legend
    recs = [Rectangle((0,0),1,1,color=c,ec="k") for c in ['C0','C1','C2','C4','C3']]
    labels = ['All signals','Low amplitude signals','High amplitude signals','True signals','Reconstructed signals']
    fig.legend(recs,labels,ncol=5,loc='center',bbox_to_anchor=(0.5,0.5))
    for ax in axes.flatten():
        ax.grid()

    if save_path:
        plt.savefig(f'{save_path}/Statanalysis.pdf',bbox_inches='tight')
    if plot:
        plt.show()
    else:
        plt.close()

def calculate_stats(x_test, y_test, trained_model, threshold, split_value):
    '''
    Calculates the stats for all 5 metrics for the provided test data set and model
    '''
    times = np.linspace(0, 640, 2048)
    stats = {}
    #dived test data set based on true amplitude
    all_data = (x_test, y_test)
    low_data = filter_dataset_amplitude(x_test, y_test, end=split_value)
    high_data = filter_dataset_amplitude(x_test, y_test, start=split_value)

    #calculate stats for every divided data set
    for set,setstring in zip([all_data,low_data,high_data],['all','low','high']):
        (x_test ,y_test) = set
        recon_signals = trained_model.predict(x_test)
        stats[f'{setstring}_recon'] = stats_on_single_set(x_test, threshold, times, recon_signals)
        stats[f'{setstring}_signal'] = stats_on_single_set(x_test, threshold, times, y_test)
    return stats

def filter_dataset_amplitude(x, y, start=0., end=np.inf):
    '''
        Filter data set based on amplitude range
    '''

    xx = x.copy()
    yy = y.copy()
    mask = (np.max(yy, axis=1) > start)*(np.max(yy, axis=1) < end)

    return xx[mask], yy[mask]

def stats_on_single_set(x_test, threshold, times, signals):
    '''
    calculate stats for single data set
    '''

    recon_stats = np.empty((x_test.shape[0], 5))
    for counter, cur_sig in enumerate(signals):#for every sample calculate the metrics
        s_max, s_max_i = np.max(cur_sig), np.argmax(cur_sig)
        peak_i, _ = find_peaks(np.abs(cur_sig), height=threshold * s_max)
        peak_times = times[peak_i]
        recon_stats[counter] = s_max, times[s_max_i], peak_times[0], peak_times[-1] - peak_times[0], len(peak_i)
    return recon_stats.T # strength(max),center,beginning,duration,peak_count

def tight_pairs(n_cols, fig=None):
    """
    Stitch vertical pairs together.

    Input:
    - n_cols: number of columns in the figure
    - fig: figure to be modified. If None, the current figure is used.

    Assumptions: 
    - fig.axes should be ordered top to bottom (ascending row number). 
      So make sure the subplots have been added in this order. 
    - The upper-half's first subplot (column 0) should always be present

    Effect:
    - The spacing between vertical pairs is reduced to zero by moving all lower-half subplots up.

    Returns:
    - Modified fig
    """
    if fig is None:
        fig = plt.gcf()
    for ax in fig.axes:
        if hasattr(ax, 'get_subplotspec'):
            ss = ax.get_subplotspec()
            row, col = ss.num1 // n_cols, ss.num1 % n_cols
            if (row % 2 == 0) and (col == 0): # upper-half row (first subplot)
                y0_upper = ss.get_position(fig).y0
            elif (row % 2 == 1): # lower-half row (all subplots)
                x0_low, _ , width_low, height_low = ss.get_position(fig).bounds
                ax.set_position(pos=[x0_low, y0_upper - height_low, width_low, height_low])
    return fig

def uncolide_plots(ax):
    '''
        lower top y label if it overlaps with the plot above
    '''
    y_vals = ax.get_yticks()
    x,y = ax.get_ylim()
    if y <y_vals[-2]*1.1:
        y = y_vals[-2]*1.1
        ax.set_ylim((x,y))
    return y_vals



if __name__ == '__main__':
    def model(sizes,kernel_size = [0,256]):
        '''
            generate model architecture
        '''
        #initialise model
        model = keras.Sequential()
        model.add(layers.Input((2048, 1)))

        #add one or two convolutional layer of defined kernel size
        #default is a single layer with kernel size 256
        if kernel_size[0] ==0:
            model.add(layers.Conv1D(kernel_size=(kernel_size[1]), filters=1, padding="same"))
            model.add(layers.LeakyReLU(alpha=0.3))
        else:
            model.add(layers.Input((2048, 1)))
            model.add(layers.Conv1D(kernel_size=(kernel_size[0]), filters=1, padding="same"))
            model.add(layers.LeakyReLU(alpha=0.3))
            model.add(layers.Conv1D(kernel_size=(kernel_size[1]), filters=1, padding="same"))
            model.add(layers.LeakyReLU(alpha=0.3))

        model.add(layers.Reshape((2048,)))#reshape convolutional output to add dense autoencoder
        #add dense layer of defined size and activation function based on list with consequtive desired sizes
        for size in sizes:
            model.add(layers.Dense(size))
            model.add(layers.LeakyReLU(alpha=0.3))

        #add output layer and build model
        model.add(layers.Dense(2048))
        model.build()
        return model

    def generate_sizes(depth=48,skip=4):
        '''
            generate list of dense layer sizes
        '''
        sizes = [2048]
        while sizes[-1]>depth: #if target depth is not reached add more layers
            sizes.append(sizes[-1]/skip)
        sizes[-1] = depth#after target deoth is reached or surpassed replace last layer with desired layer
        sizes = sizes+sizes[-2::-1]#add inverted list to increase layer size
        return list(map(int, sizes))


    
    #This line builds, trains and analyses the optimal determined model for denoising of antenna data
    #any optimal discussed parameters are set as default parameters
    analyse_model(model(generate_sizes()), f'models/ideal', epochs=600, batchsize=50,retrain=False)
    
       
    print('Finished')
