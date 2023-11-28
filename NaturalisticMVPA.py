# -*- coding: utf-8 -*-
"""
Created on Mon May 29 15:21:10 2023

@author: intoumanis
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 26 08:54:14 2023

@author: intoumanis
"""

import tkinter as tk
from tkinter import messagebox
import customtkinter
import pandas as pd
import os
from mne import read_epochs
from mne.io import read_raw_edf, read_raw_fif
#from mne.datasets import eegbci
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import scipy.io
from mne.decoding import (SlidingEstimator, cross_val_multiscore, LinearModel, get_coef)

plt.ioff()

### Define the main function that performs the MVPA

def eeg_MVPA(input_dir, output_dir, fs, window_size, Npermutations, scores, type_of_files, k):
    
    try:
                
        input_exist = os.path.exists(input_dir)
        if not input_exist:
            pb.destroy()
            progress_text.destroy()
            messagebox.showerror(message='There is no directory ' + input_dir)
            
        pb.configure(fg_color = ('gray80', 'gray30'))
        pb.start()
        
        # Move to the folder with the EEG data
        os.chdir(input_dir)
        path_to_data = Path('.')
        progress_text.configure(text = 'Loading the files...')
        root.update()
    
       
        # Read a sample dataset in the directory to figure out the dimensions
        # i.e., how many subjects, channels and time points exist
        if type_of_files == '.mat':
            
            files = [file.name for file in path_to_data.iterdir() if '.mat' in str(file)]
            if len(files) == 0:
                pb.destroy()
                progress_text.destroy()
                return
                messagebox.showerror(message='There are no ' + type_of_files + ' files in the specified input directory.')
            
            Nsubjects = len(files)
        
            with open(files[0], "rb") as f:
                test = scipy.io.loadmat(f)['X']
            Nchannels = test.shape[0]
            Ntimes = test.shape[1]
        
            # Create an array with the data of all subjects
            X = np.zeros((Nsubjects, Nchannels, Ntimes))
            for i in range(len(files)):
                with open(files[i], "rb") as f:
                    X[i,:,:] = scipy.io.loadmat(f)['X']
        
        elif type_of_files == '.edf':
            
            files = [file.name for file in path_to_data.iterdir() if '.edf' in str(file)]
            if len(files) == 0:
                pb.destroy()
                progress_text.destroy()
                return
                messagebox.showerror(message='There are no ' + type_of_files + ' files in the specified input directory.')
            
            Nsubjects = len(files)
            
            test_file = read_raw_edf(files[0], preload=True)
            test = test_file._data
            Nchannels = test.shape[0]
            Ntimes = test.shape[1]
            
            # Create an array with the data of all subjects
            X = np.zeros((Nsubjects, Nchannels, Ntimes))
            for i in range(len(files)):
                this_file = read_raw_edf(files[i], preload=True)
                X[i,:,:] = this_file._data
        
        elif type_of_files == '.fif':
            
            files = [file.name for file in path_to_data.iterdir() if '.fif' in str(file)]
            if len(files) == 0:
                pb.destroy()
                progress_text.destroy()
                return
                messagebox.showerror(message='There are no ' + type_of_files + ' files in the specified input directory.')
            
            Nsubjects = len(files)
            
            #test_file = read_raw_fif(files[0], preload=True)
            #test = test_file._data
            test_file = read_epochs(files[0], preload=True)
            test = test_file.get_data()[0,:,:]
            Nchannels = test.shape[0]
            Ntimes = test.shape[1]
            
            # Create an array with the data of all subjects
            X = np.zeros((Nsubjects, Nchannels, Ntimes))
            for i in range(len(files)):
                this_file = read_epochs(files[i], preload=True)
                X[i,:,:] = this_file.get_data()[0,:,:]
    
    
        # Make time windows and average the activity of each channel within each window
        progress_text.configure(text = 'Creating the time windows...')
        pb.set(0.14)
        root.update()
        
        # Change some variables from strings to integers
        fs = int(fs.split(' ')[0])
        Npermutations = int(Npermutations)    
        
        if window_size == "-" :
            window_size = Ntimes/fs
        else:
            window_size = float(window_size.split(' ')[0])
    
        window_length = int(window_size*fs)
        onsets = np.arange(0, X.shape[2]-window_length, window_length).tolist()
        X_windows = np.zeros((Nsubjects, Nchannels, len(onsets)))
        for i in range(len(onsets)):
            thisWindow = X[:,:,onsets[i]:(onsets[i]+window_length)]
            X_windows[:,:,i] = np.mean(thisWindow, axis=2)
    
        # Load the labels to be predicted (e.g., High vs. Low delta of WTP for sugar-containing)
        scores_exist = os.path.isfile(scores)
        if not scores_exist:
            pb.destroy()
            progress_text.destroy()
            messagebox.showerror(message='There is no such file ' + input_dir)
        
        target = pd.read_excel(scores, header=None).to_numpy()
        target = np.concatenate(target)
    
        # Develop the logistic regression classifier, with 5-fold cross-validation
        progress_text.configure(text = 'Estimating the classifier...')
        pb.set(0.29)
        root.update()
    
        clf = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear'))
        time_decod = SlidingEstimator(clf,scoring='roc_auc', verbose=True)
    
        # Get the accuracy of the classifier. The result will be an accuracy value for
        # each time window, which represents the average accuracy in this window across
        # all the 5 cross-validation runs
        scores = cross_val_multiscore(time_decod, X_windows, target, cv=5)
        scores = np.mean(scores, axis=0)
    
        # Plot the accuracy across time windows
        output_exist = os.path.exists(output_dir)
        if not output_exist:
            os.makedirs(output_dir)
        
        times = [x/fs for x in onsets]
        fig, ax = plt.subplots()
        ax.plot(times, scores*100, label='score')
        ax.axhline(50, color='k', linestyle='--', label='chance')
        ax.set_ylim(bottom=0, top=100)
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.axvline(.0, color='k', linestyle='-')
        fig.set_size_inches(12, 6)
        file_to_be_saved = os.path.join(output_dir, "Accuracy_across_time.png")
        plt.savefig(file_to_be_saved, bbox_inches='tight', dpi=300)
        plt.close()
    
        # Do permutations to examine the statistical significance of the accuracy
        # (This is time consuming, so by default we do 1000, but the user may change it to 10000)
        # At each permutation step/iteration, the target variable is shuffled and the classification
        # procedure is repeated.
        progress_text.configure(text = 'Running ' + str(Npermutations) + ' permutations...')
        pb.set(0.43)
        root.update()
    
        random_scores = np.zeros((len(onsets),Npermutations))
        for i in range(Npermutations):
            np.random.shuffle(target)
            theseScores = cross_val_multiscore(time_decod, X_windows, target, cv=5)
            random_scores[:,i] = np.mean(theseScores, axis=0)
    
        # Calculate p-values for each time window, as the percentage of random accuracies
        # that are higher than the actual accuracy.
        progress_text.configure(text = 'Calculating p-values...')
        pb.set(0.57)
        root.update()
    
        p_values = []
        for i in range(len(onsets)):
            Number_greater = sum(1 for x in random_scores[i,:] if x > scores[i])
            thisPvalue = Number_greater/Npermutations
            p_values.append(thisPvalue)
    
        # Save an excel file with the uncorrected p-values
        window_timestamps = [str(x/fs)+'-'+str(x/fs+window_size) for x in onsets]
        p = {'WindowID': range(len(onsets)), 'Time interval (in seconds)': window_timestamps, 'p-value': p_values}
        df_p = pd.DataFrame(p)
        file_to_be_saved = os.path.join(output_dir, "Uncorrected_pvalues.xlsx")
        df_p.to_excel(excel_writer = file_to_be_saved, index=False)
    
        # Find the p-values that are < 0.05
        progress_text.configure(text = 'Finding the largest cluster(s)...')
        pb.set(0.71)
        root.update()
    
        sig_p = [x for x in range(len(p_values)) if p_values[x] < 0.05]
        
        if len(sig_p)==0:
            pb.destroy()
            progress_text.destroy()
            return
            messagebox.showinfo(message='There are no time windows with statistically significant accuracy.')
    
        # Find clusters of significant p-values
    
        # First, we define a function that calculates the size of the largest cluster
        # i.e., the maximum number of consecutive time windows with significant accuracy
        
        def max_cluster(nums):
            gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
            edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
            new_edges = list(zip(edges, edges))
            lengths = []
            for i in range(len(new_edges)):
                lengths.append(new_edges[i][1]-new_edges[i][0]+1)
            return max(lengths)
    
        clusters = max_cluster(sig_p)
        
        def clusters_greater_than_k(data, k):
            splitted = np.split(data, np.where(np.diff(data) != 1)[0]+1)
            x = []
            for i in range(len(splitted)):
                if len(splitted[i]) >= k:
                    x.append(np.ndarray.tolist(splitted[i]))
            return [item for sublist in x for item in sublist]
        
        
        # Then, we find all clusters that have this maximum size (they might be more than one)
        sig_p_cluster = []
        for i in range(len(sig_p)-clusters+1):
            checks = []
            for j in range(clusters):
                thisCheck = sig_p[i] + j == sig_p[i+j]
                checks.append(thisCheck)
            
            if all(checks):
                for j in range(clusters):
                    sig_p_cluster.append(sig_p[i+j])
        
        sig_p_cluster = clusters_greater_than_k(sig_p_cluster, int(k))
    
        # Visualize again, but this time, apart from plotting the accuracy across time,
        # we also indicate the windows where the accuracy was significant after the cluster-based correction
        shaded_area = []
        for i in range(len(times)):
            shaded_area.append(np.partition(random_scores[i,:].flatten(), -round(Npermutations*0.05))[-round(Npermutations*0.05)])
        shaded_area = np.array(shaded_area)
    
        fig, ax = plt.subplots()
        ax.plot(times, 100*scores, label='Accuracy')
        ax.fill_between(times, 0, 100*shaded_area, color='lightgray', label = '95% quantile of ' + str(Npermutations) + ' permutations')
        for i in range(len(sig_p_cluster)):
            ax.axvspan(window_size*sig_p_cluster[i]-window_size*0.5, window_size*sig_p_cluster[i]+window_size*0.5, color='salmon', alpha=0.5, lw=0, label =  "_"*i + "Statistically significant after cluster-based correction")
        ax.set_ylim(bottom=0, top=100)
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Accuracy (%)')  # Area Under the Curve
        ax.legend()
        ax.axvline(.0, color='k', linestyle='-')
        fig.set_size_inches(24, 12)
        file_to_be_saved = os.path.join(output_dir, "Accuracy_across_time_after_correction.png")
        plt.savefig(file_to_be_saved, bbox_inches='tight', dpi=300)
        plt.close()
    
        # Visualize features
        progress_text.configure(text = 'Saving the spatial patterns of the classifier...')
        pb.set(0.86)
        root.update()
    
        clf = make_pipeline(StandardScaler(), LinearModel(LogisticRegression(solver='liblinear')))
        time_decod = SlidingEstimator(clf,scoring='roc_auc', verbose=True)
        time_decod.fit(X_windows, target)
        coef = get_coef(time_decod, 'patterns_', inverse_transform=True)
        
        df = pd.DataFrame(coef).T
        file_to_be_saved = os.path.join(output_dir, "Spatial_filters.xlsx")
        df.to_excel(excel_writer = file_to_be_saved)
    
        #if type_of_files == '.mat':
        #    
        #    raw_file = mne.io.read_raw_brainvision(example_raw_file, preload=True)
        #    raw_file.drop_channels(['HEOG', 'VEOG', 'BIP1'])
        
        #    montage = mne.channels.make_standard_montage("standard_1020")
        #    raw_file.set_montage(montage)
        #
        #elif type_of_files == '.edf':
        #    
        #    raw_file = read_raw_edf(example_raw_file, preload=True)
        #    eegbci.standardize(raw_file)
        #    montage = mne.channels.make_standard_montage("standard_1005")
        #    raw_file.set_montage(montage)
        #
        #
        #for i in range(len(sig_p_cluster)):
        #    fig, ax = plt.subplots(figsize=(5, 5))
        #    mne.viz.plot_topomap(coef[:,sig_p_cluster[i]], raw_file.info, cmap='RdBu_r', axes=ax)
        #    ax.set_title('Spatial patterns of the classifier during the ' + str(sig_p_cluster[i]) + 'th second', fontsize=12)
        #    file_to_be_saved = os.path.join(output_dir, "Spatial_Patterns_" + str(sig_p_cluster[i]) + ".png")
        #    plt.savefig(file_to_be_saved, bbox_inches='tight', dpi=300)
        #    plt.close()
        #
        pb.set(1)
        progress_text.configure(text = '')
        root.update()
        pb.destroy()
        messagebox.showinfo(message='The MVPA analysis is completed!')

    except(RuntimeError, ValueError, TypeError, NameError):
        
        progress_text.configure(text = '')
        root.update()
        pb.destroy()
        return
        messagebox.showerror(message='There was an error')


def show_progress_bar():
    pb.configure(fg_color = ('#939BA2', '#4A4D50'))
    pb.update()
    pb.start()


def change_appearance_mode_event():
        customtkinter.set_appearance_mode(appearance_mode_menu.get())
        root.deiconify()

def _quit():
    root.quit()     
    root.destroy()
    


### Now create a GUI so that the user can easily perform this analysis    

if __name__ == "__main__":
    
    ### Create the main window
    customtkinter.set_appearance_mode("Dark")
    root = customtkinter.CTk()
    root.title('Naturalistic_MVPA_Toolbox')
    
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    win_width = 1400
    win_height = 600
    x = (screen_width/2) - (win_width/2)
    y = (screen_height/2) - (win_height/2)
    root.geometry(f'{win_width}x{win_height}+{int(x)}+{int(y)}')
    root.resizable(0, 0)
    
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=1)
    root.rowconfigure(0, weight=1)
    root.rowconfigure(1, weight=10)
    root.rowconfigure(2, weight=3)
    
    
    ### Create the directory frame
    directory_frame = customtkinter.CTkFrame(root, fg_color = ('gray80', 'gray30'))
    directory_frame.columnconfigure(0, weight=1)
    
    customtkinter.CTkLabel(root, text = 'Directory and File setup', font = customtkinter.CTkFont(family='Arial', size=22, slant='italic')).grid(column=0, row=0, sticky=tk.SE, pady=(20,0), padx=(0,240))

    # Input directory
    customtkinter.CTkLabel(directory_frame, text='Input directory:', font = customtkinter.CTkFont(family='Arial', size=18)).grid(column=0, row=0, sticky=tk.E)
    input_dir = customtkinter.CTkEntry(directory_frame, width=350, font=customtkinter.CTkFont(family='Arial', size=18), placeholder_text = "e.g., D:/folder_with_eeg_data")
    input_dir.grid(column=1, row=0, sticky=tk.NW)
    
    # Type of files
    type_of_files = customtkinter.CTkOptionMenu(directory_frame,
                                              values=[".mat", ".edf", ".fif"],
                                              width=80,
                                              font = customtkinter.CTkFont(family='Arial', size=18),
                                              dropdown_font = customtkinter.CTkFont(family='Arial', size=18))
    type_of_files.grid(column=2, row=0, sticky=tk.E)
    type_of_files.set(".mat")

    # Output directory
    customtkinter.CTkLabel(directory_frame, text='Output directory:', font = customtkinter.CTkFont(family='Arial', size=18)).grid(column=0, row=1, sticky=tk.E)
    output_dir = customtkinter.CTkEntry(directory_frame, width=350, font=customtkinter.CTkFont(family='Arial', size=18), placeholder_text = "e.g., D:/output")
    output_dir.grid(column=1, row=1, sticky=tk.W)

    # Input excel file wityh the division of subjects
    customtkinter.CTkLabel(directory_frame, text='Division of subjects:', font = customtkinter.CTkFont(family='Arial', size=18)).grid(column=0, row=2, sticky=tk.E)
    scores = customtkinter.CTkEntry(directory_frame, width=350, font=customtkinter.CTkFont(family='Arial', size=18), placeholder_text = "e.g., D:/memberships.xlsx")
    scores.grid(column=1, row=2, sticky=tk.W)
   
    for widget in directory_frame.winfo_children():
        widget.grid(padx=10, pady=25)

    directory_frame.grid(column=0, row=1)
    directory_frame.grid(padx=25, pady=10, sticky=tk.NE)
       
       
    ### Create the parameter frame
    parameter_frame = customtkinter.CTkFrame(root, fg_color = ('gray80', 'gray30'))
    parameter_frame.columnconfigure(0, weight=1)
    
    customtkinter.CTkLabel(root, text = 'Parameter setup', font = customtkinter.CTkFont(family='Arial', size=22, slant='italic')).grid(column=1, row=0, sticky=tk.SW, pady=(20,0), padx=(230,0))

    # Sampling frequency
    customtkinter.CTkLabel(parameter_frame, text='Sampling frequency:', font = customtkinter.CTkFont(family='Arial', size=18)).grid(column=0, row=0, sticky=tk.E)
    fs = customtkinter.CTkOptionMenu(parameter_frame,
                                              values=["100 Hz", "250 Hz", "500 Hz", "1000 Hz"],
                                              width=300,
                                              font = customtkinter.CTkFont(family='Arial', size=18),
                                              dropdown_font = customtkinter.CTkFont(family='Arial', size=18))
    fs.grid(column=1, row=0, sticky=tk.E)
    fs.set("250 Hz")

    # Window size
    customtkinter.CTkLabel(parameter_frame, text='Window size:', font = customtkinter.CTkFont(family='Arial', size=18)).grid(column=0, row=1, sticky=tk.E)
    window_size = customtkinter.CTkOptionMenu(parameter_frame,
                                              values=["-", "0.2 seconds", "0.5 seconds", "1 second", "2 seconds", "5 seconds", "10 seconds"],
                                              width=300,
                                              font = customtkinter.CTkFont(family='Arial', size=18),
                                              dropdown_font = customtkinter.CTkFont(family='Arial', size=18))
    window_size.grid(column=1, row=1, sticky=tk.E)
    window_size.set("1 second")
    
    
    # Minimum cluster
    customtkinter.CTkLabel(parameter_frame, text='Minimum size of temporal clusters:', font = customtkinter.CTkFont(family='Arial', size=18)).grid(column=0, row=2, sticky=tk.E)
    k = customtkinter.CTkEntry(parameter_frame, width=300, font=customtkinter.CTkFont(family='Arial', size=18), placeholder_text = "e.g., 3")
    k.grid(column=1, row=2, sticky=tk.W)
    
    
    # Number of permutations
    customtkinter.CTkLabel(parameter_frame, text='Number of permutations:', font = customtkinter.CTkFont(family='Arial', size=18)).grid(column=0, row=3, sticky=tk.E)
    Npermutations = customtkinter.CTkOptionMenu(parameter_frame,
                                              values=["100", "200", "500", "1000", "2000", "5000", "10000"],
                                              width=300,
                                              font = customtkinter.CTkFont(family='Arial', size=18),
                                              dropdown_font = customtkinter.CTkFont(family='Arial', size=18))
    Npermutations.grid(column=1, row=3, sticky=tk.E)
    Npermutations.set("1000")
    
    for widget in parameter_frame.winfo_children():
        widget.grid(padx=10, pady=25)
        
    parameter_frame.grid(column=1, row=1)
    parameter_frame.grid(padx=25, pady=10, sticky=tk.NW)
    
    
    ### Create the running and exit button frame
    button_frame = customtkinter.CTkFrame(root, fg_color = 'transparent')
    button_frame.grid(row=2, columnspan=2)
    button_frame.grid(padx=0, pady=10, sticky = tk.N)
    
    customtkinter.CTkButton(master=button_frame,
                                 text="Run MVPA analysis",
                                 font = customtkinter.CTkFont(family='Arial', size=20),
                                 command = lambda: eeg_MVPA(input_dir.get(), output_dir.get(), fs.get(), window_size.get(), Npermutations.get(), scores.get(), type_of_files.get(), k.get()),
                                 border_width=0,
                                 corner_radius=8).grid(column=0, row=0)
    
    customtkinter.CTkButton(master=button_frame,
                                 text="Close the GUI",
                                 fg_color = 'firebrick',
                                 hover_color = 'darkred',
                                 font = customtkinter.CTkFont(family='Arial', size=20),
                                 command=_quit,
                                 border_width=0,
                                 corner_radius=8).grid(column=1, row=0)
    
   
    # Create an updating progress text
    progress_text = customtkinter.CTkLabel(button_frame, text = '', font = customtkinter.CTkFont(family='Arial', size=18))
    progress_text.grid(column=0, row=1)
    
    for widget in button_frame.winfo_children():
        widget.grid(padx=20, pady=20)
    
    
    # Create an updating progress bar
    pb = customtkinter.CTkProgressBar(master=button_frame,
                         orientation='horizontal',
                         mode='determinate',
                         fg_color = button_frame.cget("bg_color"),
                         corner_radius = 0,
                         width=200,
                         height=15)
    pb.grid(column=0, row=2)
    pb.set(0)
    
    
    ### Add an option to change from light to dark mode and back
    appearance_mode_menu = customtkinter.StringVar(value="Dark")
    switch_theme = customtkinter.CTkCheckBox(root, text="Dark theme",
                                           #switch_width = 50, switch_height = 20,
                                           font = customtkinter.CTkFont(family='Arial', size=16),
                                           command = change_appearance_mode_event, variable = appearance_mode_menu,
                                           #progress_color = 'gray', button_color = 'blue', button_hover_color = 'darkblue',
                                           onvalue="Dark", offvalue="Light")
    switch_theme.grid(column=0, row=2, sticky=tk.SW, padx = 40, pady = 40)
    
    ### Finally, run the app!
    root.mainloop()


#################
# To do list:
# 1. Visualize the spatial patterns of the classifier
# 2. Allow the user to specify thetype of classifier for the decoding (but it should be linear in case we want spatial patterns)
# 3. Allow sliding time windows - with overlap



