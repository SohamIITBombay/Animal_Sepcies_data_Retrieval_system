import torch
import torch.nn as nn
import os, glob, re
from Modelbuilder_V2 import Classifier
from typing import Union, Optional, List, Type, Any, Dict


def get_gpu_status() -> bool:
    if torch.cuda.is_available():
        gpu = True
    else:
        gpu = None
    return gpu

# Functions for reading the data written during previous experiment
def get_latest_exp(performance_directory: str):

    if not glob.glob(performance_directory + '\\Exp_data\\*.txt') == []:
        latest_file = max(glob.glob(performance_directory + \
                        '\\Exp_data\\*.txt'), 
                        key=os.path.getctime)
        return latest_file
    else:
        return None

def get_latest_cp(performance_directory: str):

    if not glob.glob(performance_directory + '\\Checkpoints\\*.pth') == []:
        latest_cp = max(glob.glob(performance_directory + \
                        '\\Checkpoints\\*.pth'), 
                        key=os.path.getctime)
        return latest_cp
    else:
        return None

def get_valid_loss(checkpoint, performance_directory, Model):

    name = (re.findall('\\\\([^\\\\]+)\.pth', checkpoint))[0]
    exp_no = (re.findall('_([0-9]+)', name))[0]
    found = False

    for filename in glob.glob(performance_directory + '\\Exp_data\\*.txt'):
        if re.search('_' + exp_no, filename):
            found = True
            with open(filename, 'r') as tfile:
                lines = tfile.readlines()
                for line in lines:
                    if re.search('Minimum Valid loss:', line):
                        valid_loss = float(re.findall(': (.+)', line)[0])
                        print(f"{name} loaded Successfully")
    if not found:
        delete_file(Model, stats_delete=False)
        raise FileNotFoundError("Text file for the checkpoint" + checkpoint + "not found")

    return valid_loss

def get_hyperparameters(checkpoint, performance_directory, Model):

    name = (re.findall('\\\\([^\\\\]+)\.pth', checkpoint))[0]
    exp_no = (re.findall('_([0-9]+)', name))[0]
    #print('exp no', exp_no)
    for filename in glob.glob(performance_directory + '\\Exp_data\\*.txt'):
        if re.search('_' + str(exp_no), filename):
            found = True
            with open(filename, 'r') as tfile:
                #print(filename)
                lines = tfile.readlines()
                found_f, found_t, found_p, found_s = False, False, False, False
                found_g, found_o, found_sh = False, False, False
                factor, threshold, patience, step_size, gamma = None, None, None, None, None
                for line in lines:
                    if re.search('Best Learning rate that gave min valid loss:', line):
                        learning_rate = float(re.findall(': (.+)', line)[0])                        
                    if re.search('Factor:', line) and not found_f:
                        factor = float(re.findall(': (.+)', line)[0])
                        found_f = True
                    if re.search('Threshold:', line) and not found_t:
                        threshold = float(re.findall(': (.+)', line)[0])
                        found_t = True
                    if re.search('Patience:', line) and not found_p:
                        patience = int(re.findall(': (.+)', line)[0])
                        found_p = True
                    if re.search('Step size:', line) and not found_s:
                        step_size = int(re.findall(': (.+)', line)[0])
                        found_s = True
                    if re.search('Gamma:', line) and not found_g:
                        gamma = float(re.findall(': (.+)', line)[0])
                        found_g = True
                    if re.search('Optimizer name:', line) and not found_o:
                        optimizer = re.findall(': (.+)', line)[0]
                        found_o = True
                    if re.search('Scheduler name:', line) and not found_sh:
                        scheduler = re.findall(': (.+)', line)[0]
                        found_sh = True

    if not found:
        delete_file(Model, stats_delete=False)
        raise FileNotFoundError("Text file for the checkpoint: " + checkpoint + " not found")

    #print(factor, threshold, patience, step_size, gamma)
    return learning_rate, factor, threshold, patience, step_size, gamma, optimizer, scheduler



# Instantiate the model
def instantiate_model(model_to_train: str, 
                    dataset_directory: str, 
                    performance_directory: str, 
                    gpu: Optional[bool] = None):

    file = get_latest_exp(performance_directory)
    if file is not None:
        filename = re.findall('\\\\([^\\\\]+)\.txt', file)
        exp_no = int((re.findall('_([0-9]+)', filename[0]))[0])
        exp_no += 1
    else:
        exp_no = 1

    Model = Classifier(exp_no, model_to_train, dataset_directory, performance_directory, gpu=gpu)
    return Model


# Loads the data and Imports model
def load_data(Model, num_fcl: int = 1, features: Optional[List] = None) -> None:

    Model.load_data(num_workers=1)
    Model.load_model(freeze_conv_layers=True, num_FCL=num_fcl, features=features)


# Sets up optimizer and scheduler
def setup_opt(Model, optimizer: str, scheduler: str, performance_directory: str, 
                custom_optim_setup: bool = False, checkpoint: Optional[str] = None) -> None:
    if custom_optim_setup:
        lr = round(float(input("Enter Learning rate:\n>>>>")), 3)
        f = round(float(input("Enter the Factor for reduceOnPlateau scheduler \
                        (Enter 0 if not applicable):\n>>>>")), 2)
        t = round(float(input("Enter the Threshold of reduceOnPlateau scheduler \
                        (Enter 0 if not applicable:\n>>>>")), 5)
        p = int(input("Enter the Patience of reduceOnPlateau scheduler \
                        (Enter 0 if not applicable:\n>>>>"))
        s = int(input("Enter step size for stepLR scheduler \
                        (Enter 0 if NA):\n>>>>"))
        g = float(input("Enter gamma for stepLR scheduler \
                        (Enter 0 if NA):\n>>>>"))
        
        Model.setup_optimization(optimizer, scheduler, learning_rate=lr, factor=f, 
                                threshold=t, patience=p, step_size=s, gamma=g)
    else:
        lr, f, t, p, s, g, opt, sch = get_hyperparameters(checkpoint, performance_directory, Model)
        Model.setup_optimization(optimizer_name=opt, scheduler_name=sch, learning_rate=lr, factor=f, 
                                threshold=t, patience=p, step_size=s, gamma=g)


# Training process
def train(Model, 
        optimizer: str, 
        scheduler: str, 
        performance_directory: str, 
        n_epochs: int = 30, 
        custom_path_bool: bool = False, 
        load_from_checkpoint: bool = False, 
        custom_optim_setup: bool = True) -> None:

    
    if custom_path_bool and load_from_checkpoint:
        delete_file(Model, stats_delete=False)
        raise RuntimeError("load_prev should be False when Custom path is given and Vice Versa.")

    if custom_path_bool:
        custom_path = input("Enter the path of the checkpoint below:\n>>>>")
    else:
        custom_path = None

    #######################--- Previous Checkpoint loading ---##########################        
    if load_from_checkpoint:
        latest_checkpoint = get_latest_cp(performance_directory)
        if latest_checkpoint is not None:
            valid_loss = get_valid_loss(latest_checkpoint, performance_directory, Model)
            setup_opt(Model, optimizer, scheduler, performance_directory, 
                    custom_optim_setup, latest_checkpoint)
        else:
            delete_file(Model, stats_delete=False)
            raise FileNotFoundError("A checkpoint file exptected \
                                    if loading from previous checkpoint!")
        
        checkpoint = latest_checkpoint

    #######################--- Custom Checkpoint loading ---##########################
    elif custom_path is not None:

        if not os.path.exists(custom_path):
            delete_file(Model, stats_delete=False)
            raise TypeError("File name expected as \
                                '../yourPath/checkpointFileName.pth'. Got " + custom_path)
        if not os.path.isfile(custom_path):
            delete_file(Model, stats_delete=False)
            raise FileNotFoundError("Custom Checkpoint entered not found: " + custom_path)
           
        
        setup_opt(Model, optimizer, scheduler, performance_directory, 
                custom_optim_setup, custom_path)
                
        valid_loss = get_valid_loss(custom_path, performance_directory, Model)
        checkpoint = None

    else:
        checkpoint = None
        valid_loss = None
        setup_opt(Model, optimizer, scheduler, performance_directory, custom_optim_setup=True)

    Model.train(n_epochs, custom_path, load_prev=checkpoint, valid_loss=valid_loss)


def delete_file(Model, stats_delete=True):

    exp_no = Model.exp_no
    Model.delete_file(exp_no, stats_delete)


if __name__ == "__main__":
    
    # Experimentation Controls
    #### Primary controls
    #If True, the module will load the last experiment's learning rate and 
    # other parameters of optmization and lr scheduler
    custom_optim_setup = False 
    #If True, module will load from checkpoint created by last experiment
    load_from_checkpoint = False
    #If True, model will begin training from a checkpoint saved at a custom path 
    # (or any desired checkpoint)
    custom_path_bool = False
    Num_epochs = 10
    Model_to_train = 'resnet18'
    num_fcl = 2
    features_in_fc_layers = [128]
    optimizer = 'SGD'
    scheduler = 'reduceOnPlateau'
    #### Secondary controls
    #If True, Plots images with predicted names and actual names as plot titles
    want_to_visualize = False
    #If True, plots image on left hand side and histogram of class probabilities on right
    class_histogram_plot = False
    #If True, saves the whole model so as to use for inference 
    # (Enabled after reaching satisfactory performance)
    wish_to_save = True


    # Down below is a data set directory
    dataset_dir = 'C:\\Users\\soham\\Desktop\\ME781_project\\Code\\Animal_dataV2'
    #data_dir = 'C:\\Users\\soham\\Desktop\\ME781_project\\Code\\Animal10'
    
    # Directory where all the info of the experiment is stored
    performance_directory = 'C:\\Users\\soham\\Desktop\\ME781_project\\Summary'

    gpu = get_gpu_status()
    classifier = instantiate_model(Model_to_train, dataset_dir, performance_directory, gpu=gpu)
    load_data(classifier, num_fcl, features_in_fc_layers)

    train(classifier, optimizer, scheduler, performance_directory, Num_epochs, 
        custom_path_bool, load_from_checkpoint, custom_optim_setup)
