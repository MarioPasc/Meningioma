#!/usr/bin/env python3
"""
Patient Spacing Filter Utility

This script analyzes medical imaging data (NRRD files) and identifies patients
with all sequences having Z-spacing less than or equal to a specified threshold.
It can also find patients common to specified pulse sequences.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import nrrd
from tqdm import tqdm # type: ignore


def get_voxel_spacing(file_path):
    """Extract voxel spacing from an NRRD file."""
    try:
        header = nrrd.read_header(file_path)
        
        # Option 1: Direct spacing field
        spacing = header.get('spacing', None)
        
        # Option 2: Calculate from space directions
        if spacing is None:
            space_directions = header.get('space directions', None)
            if space_directions is not None:
                # Handle different space_directions formats
                try:
                    # For matrices stored as nested lists
                    spacing = [np.linalg.norm(np.array(vec, dtype=float)) for vec in space_directions]
                except TypeError:
                    # For matrices stored as numpy arrays or other formats
                    space_array = np.array(space_directions, dtype=float)
                    spacing = [np.linalg.norm(space_array[i]) for i in range(space_array.shape[0])]
            else:
                return None
        
        return spacing
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return None


def find_patients_with_spacing(dataset_root, z_spacing_threshold):
    """Find patients with all sequences having Z-spacing <= threshold."""
    # Initialize lists to store data
    data = []

    # Process TC directory
    tc_dir = os.path.join(dataset_root, 'TC')
    if os.path.exists(tc_dir):
        # Get all patient directories
        patient_dirs = [d for d in os.listdir(tc_dir) if os.path.isdir(os.path.join(tc_dir, d)) and d.startswith('P')]
        
        for patient_dir_name in tqdm(patient_dirs, desc="Processing CT files"):
            patient_id = patient_dir_name
            patient_dir_path = os.path.join(tc_dir, patient_dir_name)
            scan_file = os.path.join(patient_dir_path, f"TC_{patient_id}.nrrd")
            
            if os.path.exists(scan_file):
                spacing = get_voxel_spacing(scan_file)
                if spacing:
                    data.append({
                        'Patient': patient_id,
                        'Modality': 'CT',
                        'Sequence': 'N/A',
                        'X_spacing': spacing[0],
                        'Y_spacing': spacing[1],
                        'Z_spacing': spacing[2],
                        'File': scan_file
                    })

    # Process RM directory with its subdirectories
    rm_dir = os.path.join(dataset_root, 'RM')
    if os.path.exists(rm_dir):
        sequence_dirs = [d for d in os.listdir(rm_dir) if os.path.isdir(os.path.join(rm_dir, d))]
        
        for sequence_dir_name in sequence_dirs:
            sequence = sequence_dir_name
            sequence_dir_path = os.path.join(rm_dir, sequence_dir_name)
            
            # Get all patient directories for this sequence
            patient_dirs = [d for d in os.listdir(sequence_dir_path) 
                            if os.path.isdir(os.path.join(sequence_dir_path, d)) and d.startswith('P')]
            
            for patient_dir_name in tqdm(patient_dirs, desc=f"Processing MRI {sequence} files"):
                patient_id = patient_dir_name
                patient_dir_path = os.path.join(sequence_dir_path, patient_dir_name)
                scan_file = os.path.join(patient_dir_path, f"{sequence}_{patient_id}.nrrd")
                
                if os.path.exists(scan_file):
                    spacing = get_voxel_spacing(scan_file)
                    if spacing:
                        data.append({
                            'Patient': patient_id,
                            'Modality': 'MRI',
                            'Sequence': sequence,
                            'X_spacing': spacing[0],
                            'Y_spacing': spacing[1],
                            'Z_spacing': spacing[2],
                            'File': scan_file
                        })

    # Create a DataFrame
    df = pd.DataFrame(data)
    
    if df.empty:
        print("No data found. Please check the dataset path.")
        return [], [], df
    
    # Find patients with all sequences having Z-spacing <= threshold
    all_patients = df['Patient'].unique()
    
    # Find patients with all sequences having Z-spacing <= threshold
    patients_all_under_threshold = []
    patients_some_under_threshold = []
    
    for patient in all_patients:
        patient_data = df[df['Patient'] == patient]
        
        # Check if all Z-spacings are below threshold
        if (patient_data['Z_spacing'] <= z_spacing_threshold).all():
            patients_all_under_threshold.append(patient)
        elif (patient_data['Z_spacing'] <= z_spacing_threshold).any():
            patients_some_under_threshold.append(patient)
    
    return patients_all_under_threshold, patients_some_under_threshold, df


def find_patients_with_all_pulses(df, pulse_sequences):
    """Find patients that have data for all the specified pulse sequences."""
    if not pulse_sequences:
        return []
    
    # Convert to list if input is a string
    if isinstance(pulse_sequences, str):
        pulse_sequences = [seq.strip() for seq in pulse_sequences.split(',')]
    
    common_patients = set()
    
    # Initialize with all patients
    for i, sequence in enumerate(pulse_sequences):
        # Find patients with this sequence
        patients_with_sequence = set(df[(df['Modality'] == 'MRI') & 
                                       (df['Sequence'] == sequence)]['Patient'].unique())
        
        # For the first sequence, initialize common_patients
        if i == 0:
            common_patients = patients_with_sequence
        else:
            # Intersect with patients having previous sequences
            common_patients &= patients_with_sequence
    
    return sorted(list(common_patients))


def generate_report(patients_all_under, patients_some_under, df, z_spacing_threshold, 
                   pulse_sequences=None, output_file=None):
    """Generate a report of patients meeting the spacing criteria."""
    report = []
    
    report.append(f"Z-Spacing Threshold: {z_spacing_threshold} mm")
    report.append(f"Total unique patients in dataset: {len(df['Patient'].unique())}")
    report.append("")
    
    # Report patients with all sequences under threshold
    report.append(f"Patients with ALL sequences having Z-spacing <= {z_spacing_threshold}mm: {len(patients_all_under)}")
    if patients_all_under:
        report.append("Patient IDs: " + ", ".join(sorted(patients_all_under)))
        
        # Detailed information for these patients
        report.append("\nDetailed information for qualifying patients:")
        for patient in sorted(patients_all_under):
            patient_data = df[df['Patient'] == patient]
            report.append(f"\nPatient {patient}:")
            for _, row in patient_data.iterrows():
                if row['Modality'] == 'MRI':
                    report.append(f"  MRI-{row['Sequence']}: {row['Z_spacing']:.3f} mm")
                else:
                    report.append(f"  CT: {row['Z_spacing']:.3f} mm")
    
    report.append("")
    
    # Report patients with some sequences under threshold
    report.append(f"Patients with SOME (but not all) sequences having Z-spacing <= {z_spacing_threshold}mm: {len(patients_some_under)}")
    
    # Add information about patients common to specified sequences
    if pulse_sequences:
        common_patients = find_patients_with_all_pulses(df, pulse_sequences)
        pulse_str = ", ".join(pulse_sequences)
        report.append(f"\nPatients present in all specified sequences ({pulse_str}): {len(common_patients)}")
        if common_patients:
            report.append("Patient IDs: " + ", ".join(common_patients))
        else:
            report.append("No patients found with all specified sequences.")
    
    # Additional sequence statistics
    if not df.empty:
        report.append("\nImaging Sequence Statistics:")
        if 'MRI' in df['Modality'].values:
            mri_sequences = df[df['Modality'] == 'MRI']['Sequence'].unique()
            report.append(f"Available MRI sequences: {', '.join(sorted(mri_sequences))}")
        
        sequence_stats = df.groupby(['Modality', 'Sequence'])['Z_spacing'].describe()
        report.append("\nZ-Spacing Statistics by Sequence:")
        report.append(sequence_stats.to_string())
    
    # Join all report lines
    report_text = "\n".join(report)
    
    # Output report
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"Report saved to {output_file}")
    else:
        print(report_text)
    
    return report_text


def main():
    parser = argparse.ArgumentParser(
        description="Find patients with all imaging sequences having Z-spacing below a threshold."
    )
    parser.add_argument(
        "--dataset", "-d", 
        required=False,
        default="/media/mpascual/PortableSSD/Meningiomas/raw/source/Meningioma_Adquisition",
        help="Path to the dataset root directory"
    )
    parser.add_argument(
        "--threshold", "-t", 
        type=float, 
        default=5.0,
        help="Z-spacing threshold in mm (default: 1.0)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for the report (if not specified, prints to console)"
    )
    parser.add_argument(
        "--pulses", "-p",
        help="Comma-separated list of MRI pulse sequences to find common patients (e.g., T1,T1SIN,T2,FLAIR)"
    )
    
    args = parser.parse_args()
    
    # Validate dataset path
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset path '{args.dataset}' does not exist.")
        return 1
    
    print(f"Analyzing dataset at: {args.dataset}")
    print(f"Z-spacing threshold: {args.threshold} mm")
    
    # Find patients meeting criteria
    patients_all_under, patients_some_under, df = find_patients_with_spacing(
        args.dataset, 
        args.threshold
    )
    
    # Process pulse sequences if provided
    pulse_sequences = None
    if args.pulses:
        pulse_sequences = [seq.strip() for seq in args.pulses.split(',')]
        common_patients = find_patients_with_all_pulses(df, pulse_sequences)
        print(f"\nPatients present in all specified sequences ({args.pulses}):")
        if common_patients:
            print(", ".join(common_patients))
        else:
            print("No patients found with all specified sequences.")
    
    # Generate and output report
    generate_report(
        patients_all_under, 
        patients_some_under, 
        df, 
        args.threshold,
        pulse_sequences,
        args.output
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())