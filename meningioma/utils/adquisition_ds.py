import os
from natsort import natsorted

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
        self.root = os.path.join(self.target_dir, 'Meningioma_Adquisition')
        self.patient_folders = [folder for folder in os.listdir(self.source_dir) 
                                if os.path.isdir(os.path.join(self.source_dir, folder))]
        self.adquisition_types = ['RM', 'TC']
        self.rm_pulses = ['T1', 'T1SIN', 'SUSC', 'T2']

    def _generate_new_structure(self) -> None:
        """
        Generate the structure of the new dataset structure. This new structure is based
        on the image adquisition type and the RM pulse.

        Args:
            None
        """

        os.makedirs(self.root, exist_ok=True)
        for type in self.adquisition_types:
            os.makedirs(os.path.join(self.root, type), exist_ok=True)
            if type == 'RM':
                for pulse in self.rm_pulses:
                    os.makedirs(os.path.join(self.root, type, pulse), exist_ok=True)

    def _move_RM_Pulse(self, pulse: str) -> None:
        """
        Examine the raw data structure for any RM pulse as an input. This function identifies:
            1. Patients with both image and segmentation.
            2. Control patients (image but no segmentation).
            3. Patients without the pulse folder.
            4. Patients without files within the pulse folder.

        Args:
            pulse (str): T1, T1SIN, SUSC or T2
        """

        pulse_root_path = os.path.join('RM', pulse)

        # Sort patient folders using natsort
        sorted_patient_folders = natsorted(self.patient_folders)

        for patient_folder in sorted_patient_folders:
            patient_path = os.path.join(self.source_dir, patient_folder)
            pulse_path = os.path.join(patient_path, pulse_root_path)

            if os.path.exists(pulse_path):  # Only process if the pulse exists for that patient
                files = os.listdir(pulse_path)

                if len(files) == 0:  # Some patients have the pulse folder but nothing inside
                    print(f'Patient {patient_folder} does not have {pulse} pulse (missing files). \n============')
                else:
                    if 'Segmentation.nrrd' in files:  # Control patients
                        segmentation = files.pop(files.index('Segmentation.nrrd'))
                    else:
                        print(f'Patient {patient_folder} with {pulse} pulse but without Segmentation.nrrd')

                    image = [file for file in files if not file.endswith(('.png', '.seg.nrrd', '.mrml'))][0]
                    print(f'Patient: {patient_folder}\n Segmentation: {segmentation}\n File: {image}\n============')

                    #TODO: Copy and change the name of the image and segmentation files into the new transformed dataset structure


            else:
                print(f'Patient {patient_folder} does not have {pulse} pulse (missing folder).\n============')


    def transform(self) -> None:
        self._generate_new_structure()

        self._move_RM_Pulse(pulse='T1')

def main() -> int:
    source = '/home/mariopasc/Python/Datasets/Meningiomas/Meningioma_RM_nrrd'
    target = '/home/mariopasc/Python/Datasets/Meningiomas'

    ds_transformer = DatasetTransformerAdquisition(source_dir=source, target_dir=target)
    ds_transformer.transform()

if __name__ == '__main__':
    main()