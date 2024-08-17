import os
from natsort import natsorted
import shutil
from tqdm import tqdm
import re # substring manipulation

class DatasetTransformerAdquisition:

    def __init__(self, source_dir:str, target_dir:str) -> None:
        """
        Class to transform a dataset from its original structure to a structure 
        based off the type of image adquisition used.

        Args:
            source_dir (str): Path to the original dataset.
            target_dir (str): Path where the transformed dataset will be saved.
        """
        self.source_dir = source_dir
        self.target_dir = target_dir

        # Root for the new transformed dataset
        self.root = os.path.join(self.target_dir, 'Meningioma_Adquisition')

        # Patient folder within the raw data (exclude .xslx and other .log files)
        self.patient_folders = [folder for folder in os.listdir(self.source_dir) 
                                if os.path.isdir(os.path.join(self.source_dir, folder))]
        # Sort them
        self.patient_folders = natsorted(self.patient_folders)

        # Image technologies applied
        self.adquisition_types = ['RM', 'TC']

        # RM pulses 
        self.rm_pulses = ['T1', 'T1SIN', 'SUSC', 'T2']

    def _generate_new_structure(self) -> None:
        """
        Generate the structure of the new dataset structure. This new structure is based
        on the image adquisition type and the RM pulse.

        Args:
            None
        """
        # Clear previous directories 
        if os.path.exists(self.root): shutil.rmtree(self.root)

        # Create root directory
        os.makedirs(self.root, exist_ok=True)
        for type in self.adquisition_types:
            os.makedirs(os.path.join(self.root, type), exist_ok=True)
            if type == 'RM':
                for pulse in self.rm_pulses:
                    os.makedirs(os.path.join(self.root, type, pulse), exist_ok=True)

    def _move_RM_Pulse(self, pulse: str, verbose: bool = False) -> None:
        """
        Examine the raw data structure for any RM pulse as an input. This function identifies:
            1. Patients with both image and segmentation.
            2. Control patients (image but no segmentation).
            3. Patients without the pulse folder.
            4. Patients without files within the pulse folder.

        Args:
            pulse (str): T1, T1SIN, SUSC or T2
            verbose (bool) = False: If set to true, display detailed process information in the terminal.
        """

        pulse_root_path = os.path.join('RM', pulse)

        for patient_folder in tqdm(iterable=self.patient_folders, desc=f'Moving {pulse} pulse ...', total=len(self.patient_folders)):
            patient_path = os.path.join(self.source_dir, patient_folder)
            pulse_path = os.path.join(patient_path, pulse_root_path)

            if os.path.exists(pulse_path):  # Only process if the pulse exists for that patient
                files = os.listdir(pulse_path)
                seg = False

                if len(files) == 0:  # Some patients have the pulse folder but nothing inside
                    if verbose: print(f'Patient {patient_folder} does not have {pulse} pulse (missing files). \n============')
                else:
                    if 'Segmentation.nrrd' in files:  # Control patients
                        segmentation = files.pop(files.index('Segmentation.nrrd'))
                        seg = True
                    else:
                        if verbose: print(f'Patient {patient_folder} with {pulse} pulse but without Segmentation.nrrd')

                    image = [file for file in files if not file.endswith(('.png', '.seg.nrrd', '.mrml'))][0]
                    if verbose: print(f'Patient: {patient_folder}\n Segmentation: {segmentation}\n File: {image}\n============')

                    # SECTION: Copy the image and segmentation files into the new patient folder for the specific pulse

                    # Strip the TC substring at the end of some patient forlders regardless of it being lowercase or uppercase
                    stripped = re.sub(r'tc', '', patient_folder, flags=re.IGNORECASE)
                    target_pulse_folder = os.path.join(self.root, 'RM', pulse, f'P{stripped}') # Strip the TC at the end
                    os.makedirs(target_pulse_folder, exist_ok=True)

                    # Copy and change the name of the image and segmentation files into the new transformed dataset structure
                    new_image_name = f'{pulse}_P{stripped}.nrrd'
                    image_source = os.path.join(pulse_path, image)
                    image_destination = os.path.join(target_pulse_folder, new_image_name)
                    shutil.copy2(image_source, image_destination)

                    if seg: # If the image has a segmentation file ...
                        new_segmentation_name = f'{pulse}_P{stripped}_seg.nrrd'
                        segmentation_source = os.path.join(pulse_path, segmentation)
                        segmentation_destination = os.path.join(target_pulse_folder, new_segmentation_name)
                        shutil.copy2(segmentation_source, segmentation_destination)

            else:
                if verbose: print(f'Patient {patient_folder} does not have {pulse} pulse (missing folder).\n============')


    def _move_TC(self, verbose: bool = False) -> None:
        """
        Move the TC images from the raw dataset to the newly transformed dataset. 

        Args:
            verbose (bool) = False: If set to true, display detailed process information in the terminal.
        """
        for patient in tqdm(iterable=self.patient_folders, desc='Moving TC images ... ', total=len(self.patient_folders)):
            patient_path = os.path.join(self.source_dir, patient)
            tc_root = os.path.join(patient_path, 'TC')

            if not os.path.exists(tc_root): # Some patients may have TC missing
                if verbose: print(f'Patient {patient} does not have TC study (missing folder).')                    

            else:
                files = os.listdir(tc_root)
                seg = False

                if len(files) == 0:
                    if verbose: print(f'Patient {patient} does not have TC study (missing files).')                    

                else:

                    if 'Segmentation.nrrd' not in files:  
                        if verbose: print(f'Patient {patient} without Segmentation.nrrd')

                    else:
                        segmentation = files.pop(files.index('Segmentation.nrrd'))
                        seg = True
                    
                    image = [file for file in files if (not file.endswith(('.png', '.seg.nrrd', '.mrml'))) 
                             and (os.path.isfile(os.path.join(tc_root, file)) and file.endswith('.nrrd'))]
                    if image:
                        image = image[0]
                    else:
                        # This patient does not have any nrrd files (error while building the dataset) e.g. Patient 70
                        continue
                    # Some TC folders have other folders inside 
                    if verbose: print(f'Patient: {patient}\n Segmentation: {segmentation}\n File: {image}\n============')

                    # SECTION: Copy the image and segmentation files into the new patient folder for the specific pulse

                    target_TC_folder = os.path.join(self.root, 'TC', f'P{patient}') 
                    os.makedirs(target_TC_folder, exist_ok=True)

                    # Copy and change the name of the image and segmentation files into the new transformed dataset structure
                    new_image_name = f'TC_P{patient}.nrrd'
                    image_source = os.path.join(tc_root, image)
                    image_destination = os.path.join(target_TC_folder, new_image_name)
                    shutil.copy2(image_source, image_destination)

                    if seg: # If the image has a segmentation file ...
                        new_segmentation_name = f'TC_P{patient}_seg.nrrd'
                        segmentation_source = os.path.join(tc_root, segmentation)
                        segmentation_destination = os.path.join(target_TC_folder, new_segmentation_name)
                        shutil.copy2(segmentation_source, segmentation_destination)

    def transform(self, verbose: bool = False) -> None:
        """
        Generate a new, clear dataset based on the image adquisition format from the raw data. 
        
        Calls:
            self._generate_new_structure(): Generate folder hierarchy
            self._move_RM_Pulse(pulse=pulse): Move the RM pulses 
            self._move_TC(): Move TC images

        Args:
            verbose (bool) = False: If set to true, display detailed process information in the terminal.
        """

        # Generate the folder hierarchy
        self._generate_new_structure()

        # Move and rename RM images
        for pulse in self.rm_pulses:
            self._move_RM_Pulse(pulse=pulse, verbose=verbose)

        self._move_TC(verbose=verbose)

def main() -> int:
    source = '/home/mariopasc/Python/Datasets/Meningiomas/Meningioma_RM_nrrd'
    target = '/home/mariopasc/Python/Datasets/Meningiomas'

    ds_transformer = DatasetTransformerAdquisition(source_dir=source, target_dir=target)
    ds_transformer.transform()

if __name__ == '__main__':
    main()