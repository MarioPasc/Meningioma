import os
import numpy as np
import nrrd
from typing import Tuple


class NRRDToNPZConverter:
    def __init__(
        self, base_ds_path: str, output_base_path: str, log_file: str
    ):
        self.base_ds_path = base_ds_path
        self.output_base_path = output_base_path
        self.log_file = log_file

    def load_nrrd_files(self, path: str) -> np.ndarray:
        try:
            data, header = nrrd.read(path)
            return data
        except Exception as e:
            raise IOError(f"Error loading NRRD file {path}: {e}")

    def reorganize_axes(
        self, images: np.ndarray, masks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            images_reordered = np.transpose(images, (2, 1, 0))
            masks_reordered = np.transpose(masks, (2, 1, 0))
            return images_reordered, masks_reordered
        except Exception as e:
            raise ValueError(f"Error reorganizing axes: {e}")

    def combine_images_and_masks(
        self, images: np.ndarray, masks: np.ndarray
    ) -> np.ndarray:
        try:
            combined = np.stack((images, masks), axis=-1)
            return combined
        except Exception as e:
            raise ValueError(f"Error combining images and masks: {e}")

    def save_to_npz(
        self, combined: np.ndarray, patient_id: str, sequence_type: str
    ) -> None:
        try:
            output_path = os.path.join(self.output_base_path, sequence_type)
            os.makedirs(output_path, exist_ok=True)
            npz_filename = f"{patient_id}_{sequence_type}.npz"
            np.savez(os.path.join(output_path, npz_filename), combined)
            print(f"Saved: {os.path.join(output_path, npz_filename)}")
        except Exception as e:
            raise IOError(
                f"Error saving NPZ file for patient {patient_id}, sequence {sequence_type}: {e}"
            )

    def log_error(self, message: str) -> None:
        with open(self.log_file, "a") as log:
            log.write(message + "\n")

    def convert_patient_data(self, patient_id: str) -> None:
        try:
            patient_path = os.path.join(self.base_ds_path, patient_id, "RM")
            sequences = [
                os.path.join(patient_path, seq)
                for seq in os.listdir(patient_path)
                if os.path.isdir(os.path.join(patient_path, seq))
            ]

            for seq in sequences:
                sequence_type = os.path.basename(seq)
                try:
                    nrrd_files = [
                        os.path.join(seq, file)
                        for file in os.listdir(seq)
                        if file.endswith(".nrrd")
                    ]
                    image_path = [
                        file
                        for file in nrrd_files
                        if os.path.basename(file) != "Segmentation.nrrd"
                    ][0]
                    mask_path = [
                        file
                        for file in nrrd_files
                        if os.path.basename(file) == "Segmentation.nrrd"
                    ][0]

                    images = self.load_nrrd_files(image_path)
                    masks = self.load_nrrd_files(mask_path)

                    print(f"Sequence: {sequence_type}")
                    print(f"Original image shape: {images.shape}")
                    print(f"Original mask shape: {masks.shape}")

                    images_reordered, masks_reordered = self.reorganize_axes(
                        images, masks
                    )

                    print(f"Reorganized image shape: {images_reordered.shape}")
                    print(f"Reorganized mask shape: {masks_reordered.shape}")

                    if images_reordered.shape != masks_reordered.shape:
                        raise ValueError(
                            f"Shapes do not match for patient {patient_id}, sequence {sequence_type}"
                        )

                    combined = self.combine_images_and_masks(
                        images_reordered, masks_reordered
                    )
                    self.save_to_npz(combined, patient_id, sequence_type)
                except Exception as e:
                    error_message = f"Error converting data for patient {patient_id}, sequence {sequence_type}: {e}"
                    print(error_message)
                    self.log_error(error_message)
        except Exception as e:
            error_message = (
                f"Error converting data for patient {patient_id}: {e}"
            )
            print(error_message)
            self.log_error(error_message)

    def convert_all_patients(self) -> None:
        try:
            patients = [
                d
                for d in os.listdir(self.base_ds_path)
                if os.path.isdir(os.path.join(self.base_ds_path, d))
            ]
            for patient_id in patients:
                self.convert_patient_data(patient_id)
        except Exception as e:
            error_message = f"Error converting all patients: {e}"
            print(error_message)
            self.log_error(error_message)


# Uso de la clase
base_ds_path = "/home/mariopasc/Python/Datasets/Meningiomas/Meningioma_RM_nrrd"
output_base_path = os.path.join(base_ds_path, "..", "npz_dataset")
log_file = os.path.join(output_base_path, "conversion_errors.log")

converter = NRRDToNPZConverter(base_ds_path, output_base_path, log_file)
converter.convert_all_patients()
