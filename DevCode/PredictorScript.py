from torchvision import transforms
import os, glob, re
import time
import numpy as np
import torch
import PIL
import wikipedia, webbrowser
# Animal species that algorithm is trained on
classes = ['SecretaryBird', 'baboon', 'buffalo', 'cheetah', 'dik-dik', 'Taurotragus', \
            'elephant', "Grant's gazelle", 'giraffe', 'Guineafowl', 'hartebeest', 'hippopotamus', \
            'human', "Spotted hyena", 'impala', 'jackal', 'kori bustard', 'lion', 'ostrich', \
            'rhebok', 'topi', 'warthog', 'wildebeast', 'zebra']

# Checks for the saved, trained model checkpoint
pth_file = \
    glob.glob('C:\\Users\\soham\\Desktop\\ME781_project\\Model_summary_fc2\\whole_model\\*.pth')

predictor = torch.load(pth_file[0])

transform = transforms.Compose([transforms.Resize(255), 
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                        [0.229, 0.224, 0.225])])



########## Scan for New File ##########
with open('processed_files.txt', "r") as f:
    processed_files = [line.rstrip() for line in f]

# Scan Minutes: Script will run idle for maximum of scan_minutes
scan_minutes = 10

time_elapsed = 0
try:
    tic = time.time()
    print('Beginning Scanning..')
    while True:

        for name in glob.glob('C:\\Users\\soham\\Desktop\\ME781_project\\from_cam\\*'):
            
            filename = re.findall('\\\\([^\\\\]+)\.jpg', name)  # <--- Regex for finding filename
            if filename[0] not in processed_files:
                new_file = filename[0] 

                
                #Importing the detected image
                img = PIL.Image.open(name, "r")
                img = transform(img)
                img = img.view(1, 3, 224, 224)
                img = img.cuda()

                # Predicting Species
                output = predictor(img)
                _, pred = torch.max(output, 1)
                predicted_species = classes[pred]
                print('Animal Species Predicted ------> ', predicted_species)

                # Searching on Web and Opening Wikipedia page
                link = wikipedia.page(predicted_species).url
                #link = wikipedia.page(new_file).url
                webbrowser.open(link)
                
                
                # Saving the names of files processed (aids in detecting new, unprocessed ones)
                processed_files.append(new_file)
                with open('processed_files.txt', "a") as F:
                    F.write(new_file + ' \n')
        
        
        toc = time.time()
        time_elapsed += (toc - tic)/60
        tic = toc
        
        if time_elapsed >= scan_minutes:
            print(f'Idle for {scan_minutes} Minutes...Stopping scan.')
            break


except KeyboardInterrupt:
    print('Scanning Stopped. Re-Run the PredictorScript.py to start scanning.')
