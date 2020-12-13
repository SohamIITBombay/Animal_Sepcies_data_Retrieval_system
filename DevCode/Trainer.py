import torch
import torch.nn as nn
import os, glob, re
from Modelbuilder_V2 import Classifier
from typing import Union, Optional, List, Type, Any, Dict
import argparse



def get_gpu_status() -> bool:
    
    """
    A function for Getting the available device
    between cuda or cpu

    Args:
    None.

    Returns:
    None.

    """

    if torch.cuda.is_available():
        gpu = True
    else:
        gpu = None
    return gpu



def get_latest_exp(performance_directory: str):
    
    """
    Function for get the Text file of the latest 
    experiment performed.

    Args:
    performance directory (str): The directory where the generated text, checkpoints
                                model_stats will be saved.

    Returns:
    None.

    """

    if not glob.glob(performance_directory + '\\Exp_data\\*.txt') == []:
        latest_file = max(glob.glob(performance_directory + \
                        '\\Exp_data\\*.txt'), 
                        key=os.path.getctime)
        return latest_file
    else:
        return None



def get_latest_cp(performance_directory: str):
    
    """
    Function for getting the latest checkpoint file for the latest 
    experiment performed.

    Args:
    performance directory (str): The directory where the generated text, checkpoints
                                model_stats will be saved.

    Returns:
    None.

    """

    if not glob.glob(performance_directory + '\\Checkpoints\\*.pth') == []:
        latest_cp = max(glob.glob(performance_directory + \
                        '\\Checkpoints\\*.pth'), 
                        key=os.path.getctime)
        return latest_cp
    else:
        return None



def get_valid_loss(checkpoint, performance_directory: str, Model):
    
    """
    A function for reading the minimum valid loss from the text file obtained 
    during a previous experiment (to compare during current experiment).

    Args:
    checkpoint (str):Path of the checkpoint desired to be loaded. The hyperparameters for this
                        checkpoint will be read from the correspoiding Text file.
    performance directory (str): The directory where the generated text, checkpoints
                                model_stats will be saved.
    Model (Classifier): The object of the class 'Classifier' instantiated before.

    Returns:
    None.

    """

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



def get_hyperparameters(checkpoint, performance_directory: str, Model):
    
    """
    A Function for reading hyperparameters like learning rate, factor, patience 
    pertaining to the optimizer and learning rate scheduler. Reads from a text 
    file generated during a previous experiment.

    Args:
    checkpoint (str):Path of the checkpoint desired to be loaded. The hyperparameters for this
                        checkpoint will be read from the correspoiding Text file.
    performance directory (str): The directory where the generated text, checkpoints
                                model_stats will be saved.
    Model (Classifier): The object of the class 'Classifier' instantiated before.

    Returns:
    None.

    """

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



def instantiate_model(model_to_train: str, 
                    dataset_directory: str, 
                    performance_directory: str, 
                    gpu: Optional[bool] = None):

    """
    A function to create the instance of the imported Class,
    Classifier.

    Args:
    model_to_train (str): name of the pretrained model to train
    dataset directory (str): Directory containing the data
    performance directory (str): The directory where the generated text, checkpoints
                                model_stats will be saved.
    gpu (bool): Boolean indicating availability of a GPU

    Returns:
    None.

    """

    file = get_latest_exp(performance_directory)
    if file is not None:
        filename = re.findall('\\\\([^\\\\]+)\.txt', file)
        exp_no = int((re.findall('_([0-9]+)', filename[0]))[0])
        exp_no += 1
    else:
        exp_no = 1

    Model = Classifier(exp_no, model_to_train, dataset_directory, performance_directory, gpu=gpu)
    return Model



def load_data(Model, num_fcl: int = 1, features: Optional[List] = None) -> None:

    """
    A function to load the data from the Dataset Directory provided to the 
    'instantiate_model()' function.

    Args:
    Model (Classifier): The object of the class 'Classifier' instantiated before.
    num_FCL (int): Number of fully connected layers attached to the end of the 
                    convolutional blocks
    features (list): If number of fully connected layers in more than 1, features list gives 
                    the number of hidden units desired in the extra fully connected layers.

    Returns:
    None.

    """

    Model.load_data(num_workers=1)
    Model.load_model(freeze_conv_layers=True, num_FCL=num_fcl, features=features)



def setup_opt(Model, optimizer: str, scheduler: str, custom_optim_setup: bool = False, 
                checkpoint: Optional[str] = None) -> None:

    """
    A function for setting up the optimization parameters like learning rate. Also 
    sets up Learning rate scheduler parameters like factor, threshold, patience etc.

    Args:
    Model (Classifier): The object of the class 'Classifier' instantiated before.
    optimizer (str): The name of the optimizer. Choose between ('adam', 'SGD')
    scheduler (str): The name of the scheduler. Choose between ('StepLR', 'reduceOnPlateau')
    custom_optim_setup (bool): A boolean specifying whether to use a custom, user-input setup
                                for optimization or to load from some desired checkpoint.
    checkpoint (str): Path of the checkpoint desired to be loaded. The hyperparameters for this
                        checkpoint will be read from the correspoiding Text file.

    Returns:
    None.

    """

    performance_directory = Model.Performance_dir
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



def train(Model, 
        optimizer: str, 
        scheduler: str,  
        n_epochs: int = 30, 
        custom_path_bool: bool = False, 
        load_from_checkpoint: bool = False, 
        custom_optim_setup: bool = True) -> None:

    """
    A function for setting up the Training method of the class Classifier.

    Args:
    Model (Classifier): The object of the class 'Classifier' instantiated before.
    optimizer (str): The name of the optimizer. Choose between ('adam', 'SGD')
    scheduler (str): The name of the scheduler. Choose between ('StepLR', 'reduceOnPlateau')
    n_epochs (int): Number of epochs to train
    custom_path_bool (bool): Specifies whether to load a custom, user-input checkpoint or
                            from a previous checkpoint.
    load_from_checkpoint (bool): Specifies whether to load a custom, user-input checkpoint or
                            from a previous checkpoint.
    custom_optim_setup (bool): A boolean specifying whether to use a custom, user-input setup
                                for optimization or to load from some desired checkpoint.

    Returns:
    None.

    """

    performance_directory = Model.Performance_dir
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
            setup_opt(Model, optimizer, scheduler, custom_optim_setup, latest_checkpoint)
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
           
        
        setup_opt(Model, optimizer, scheduler, custom_optim_setup, custom_path)
                
        valid_loss = get_valid_loss(custom_path, performance_directory, Model)
        checkpoint = None

    else:
        checkpoint = None
        valid_loss = None
        setup_opt(Model, optimizer, scheduler, custom_optim_setup=True)

    Model.train(n_epochs, custom_path, load_prev=checkpoint, valid_loss=valid_loss)



def classification_histogram(Model, visualize_histogram_plot: bool = False):

    """
    A function for creating a figure which consists of an image and the
    predicted class is the form of a histogram.

    Args:
    Model (Classifier): The object of the class 'Classifier' instantiated before.
    visualize_histogram_plot (bool): A boolean specifying whether to execute plotting histogram

    Returns:
    None.

    """

    if visualize_histogram_plot == True:
        Model.visualize_histogram()


def plot_image_class(Model, visualize_image_and_class: bool = False):

    """
    A function for creating a figure in which images will be plotted in subplots
        with their title representing the class that image belongs to.

    Args:
    visualize_image_and_class (bool): A boolean specifying whether to execute plotting

    Returns:
    None.

    """

    if visualize_image_and_class:
        Model.plot_image_class()



def save(Model, wish_to_save: bool = False):
    
    """
    A function for saving the whole model to be used for inference ahead.

    Args:
    Model (Classifier): The object of the class 'Classifier' instantiated before.
    wish_to_save (bool): A boolean specifying whether to save the model

    Returns:
    None.

    """

    if wish_to_save:
        Model.save_model(wish_to_save)



def delete_file(Model, stats_delete=True):

    """
    A function to delete the text file and model stats (tensorboard) if an error occurs and
    training could not be completed. Incomplete text files are not desirable.

    Args:
    Model (Classifier): The object of the class 'Classifier' instantiated before.
    stats_delete (bool): A boolean specifying whether to delete the model stats for tensorboard.
                        Default False since a Model stats file may not be generated while
                        attempting to delete text file.

    Returns:
    None.

    """

    exp_no = Model.exp_no
    Model.delete_file(exp_no, stats_delete)



if __name__ == "__main__":
    
    ####################### --- Control Variables --- #########################
    custom_optim_setup = True 
    load_from_checkpoint = True
    custom_path_bool = False
    Num_epochs = 10
    Model_to_train = 'resnet18'
    num_fcl = 2
    features_in_fc_layers = [128]
    optimizer = 'SGD'
    scheduler = 'reduceOnPlateau'
    visualize_image_and_class = False
    visualize_histogram_plot = False
    wish_to_save = False
    dataset_dir = 'C:\\Users\\soham\\Desktop\\ME781_project\\Code\\Animal_dataV2'
    performance_directory = 'C:\\Users\\soham\\Desktop\\ME781_project\\Summary'

    
    gpu = get_gpu_status()

    classifier = instantiate_model(Model_to_train, dataset_dir, performance_directory, gpu=gpu)
    load_data(classifier, num_fcl, features_in_fc_layers)

    train(classifier, optimizer, scheduler, Num_epochs, custom_path_bool, 
        load_from_checkpoint, custom_optim_setup)

    classification_histogram(classifier, visualize_histogram_plot)
    plot_image_class(classifier, visualize_image_and_class)
    save(classifier, wish_to_save)

