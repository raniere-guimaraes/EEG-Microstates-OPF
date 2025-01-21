%% Extraction of EEG microstates and their spatiotemporal characteristics.
%
% This script should be used to run both normal and abnormal EEG files.
%
% Author: Raniere Rocha Guimarães
% Federal University of Ceará
% Fortaleza/Ceará/Brazil

%% Data preloading
% Clear workspace and command window
clear;clc;

% Start EEGLAB to load all dependent paths
eeglab

% Set the path to the directory with the channel location EEG files, and directory with the montage EEG files
elc_path = fullfile('C:/eeglab2024.0/plugins/dipfit/standard_BEM/elec/standard_1020.elc');
ced_path = fullfile('C:/eeglab2024.0/plugins/dipfit/standard_BEM/elec/channel Location MNI.ced');

% Retrieve a list of all EEG Files in EEGdir
EEGFiles = dir('*.edf');

%% 1. Data selection and aggregation
% 1.1 Loading datasets in EEGLAB
for i=1:length(EEGFiles)
    EEG = pop_biosig([EEGFiles(i).name], 'blockrange', [180 190]);
    [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG );
end

% Updates EEGLAB datasets
eeglab redraw

% 1.2 Range definition
start_range = 1;
end_range = length(EEGFiles);

range = [start_range end_range];

fprintf('# MicroStates:\t%d\n', 16);

%% 2. PREPROCESSING STEPS
% 2.1 Select 21 Channels present across all the sets
for i=1:length(EEGFiles)

    [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET,'retrieve',i,'study',0);
    
    EEG = pop_select( EEG, 'channel',{'FP1-REF','FP2-REF','F3-REF','F4-REF','C3-REF','C4-REF','P3-REF','P4-REF', ...
        'O1-REF','O2-REF','F7-REF','F8-REF','T3-REF','T4-REF','T5-REF','T6-REF','A1-REF','A2-REF','FZ-REF','CZ-REF','PZ-REF'});

    EEG = eeg_checkset( EEG );
    [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET,'overwrite','on','gui','off');
    
    % Stores channel locations
    EEG = pop_chanedit(EEG, 'lookup',elc_path,'load',{ced_path,'filetype','chanedit'}, ...
        'eval','chans = pop_chancenter( chans, [],[]);');

    % Stores channel locations
    [ALLEEG, EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);

    [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, 'retrieve', i, 'study', 0);
    
    % Filtering
    EEG = pop_eegfiltnew(EEG, 'locutoff',1,'hicutoff',30);
    
    % Overwrite and rename set
    [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET,'overwrite','on', 'gui', 'off');
end

eeglab redraw % updates EEGLAB datasets interface

%% 2.2 RESAMPLE IF NECESSARY
% Most sets are sampled at 250 Hz
% MS Analysis requires consistent Srates in all samples
for i=1:length(EEGFiles)
    if ALLEEG(i).srate ~= 250
        EEG = pop_resample( EEG, 250);
        CURRENTSET = i;
        [ALLEEG, EEG, ~] = pop_newset(ALLEEG, EEG, CURRENTSET,'overwrite','on','gui','off');%only overwrite
    end
end

%% 3. Select data for microstate analysis
CURRENTSET = length(ALLEEG);
[EEG, ALLEEG] = pop_micro_selectdata( EEG, ALLEEG, ...
    'datatype', 'spontaneous', ...
    'avgref', 1, ...
    'normalise', 1, ...
    'MinPeakDist', 10, ...
    'Npeaks', 100, ...
    'GFPthresh', 1, ...
    'dataset_idx', range);

% Store data in a new EEG structure
[ALLEEG, EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
eeglab redraw % updates EEGLAB datasets

%% 3.1 Microstate segmentation
% Select the "GFPpeak" dataset and make it the active set
[ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, 'retrieve',CURRENTSET+1,'study',0);
eeglab redraw % updates EEGLAB datasets

% Perform the microstate segmentation
EEG = pop_micro_segment(EEG, 'algorithm' , 'modkmeans', ...
    'sorting', 'Global explained variance', ...
    'Nmicrostates', 16, ...
    'verbose', 1, ...
    'normalise', 1, ...
    'Nrepetitions', 50, ...
    'max_iterations', 1000, ...
    'threshold', 1e-06, ...
    'fitmeas', 'CV', ...
    'optimised',1);

% Store data in a new EEG structure
[ALLEEG, EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
 
%% 3.3 Select active number of microstates (maps)
EEG = pop_micro_selectNmicro(EEG, 'Nmicro', 16);
[ALLEEG, EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);

% Import microstate prototypes from other dataset to the datasets that should be back-fitted
% note that dataset number i is the GFPpeaks dataset with the microstate prototypes
for i = 1:length(EEGFiles)
    fprintf('Importing prototypes and backfitting for dataset %i\n',i)
    [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET,'retrieve',i,'study',0);
    EEG = pop_micro_import_proto( EEG, ALLEEG, length(ALLEEG));
    
    % 3.6 Back-fit microstates on EEG
    EEG = pop_micro_fit( EEG, 'polarity', 0 );

    % 3.7 Temporally smooth microstates labels
    EEG = pop_micro_smooth( EEG, 'label_type', 'backfit', ...
        'smooth_type', 'reject segments', ...
        'minTime', 30, ...
        'polarity', 0 );
    % 3.9 Calculate microstate statistics
    EEG = pop_micro_stats( EEG, 'label_type', 'backfit', 'polarity', 0 );
    [ALLEEG, EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);

    ALLEEG(i).msproc = true;
end

eeglab redraw