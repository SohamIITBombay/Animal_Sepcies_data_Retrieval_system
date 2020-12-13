# Animal_Sepcies_data_Retrieval_system
ResNET based CNN for Animal Species Classification
- Dataset agnostic: Can take in any dataset with any number of classes. Only requirement is the arrangement of data.
 -- Arrangement should be like ~/dataset/train/class_name/xyz.png and ~/dataset/test/class_name/xyz.png
 
 - Imports Torchvision's pre-trained models for training on custom dataset.
 - Ability to replace the last fully connected layer with any number of custom fully connected layers. Use variable: ````num_fcl````
 
 ### Trainer.py
 #### The main file for beginning training
 - Must input: Dataset Directory, Performance directory (Used to store the generated result data)
 - Other input variables are available in the Trainer.py script for modification
