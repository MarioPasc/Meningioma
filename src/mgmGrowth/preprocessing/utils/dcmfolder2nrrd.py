import os
import SimpleITK as sitk
import pydicom

def dcmfolder_to_nrrd(input_folder: str, output_folder: str):
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    # Initialize the series reader
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()  # Load DICOM tags into metadata
    reader.LoadPrivateTagsOn()               # Also load private tags if present

    # Get all series IDs in the input folder (each corresponds to a sequence/volume)
    series_ids = reader.GetGDCMSeriesIDs(input_folder)
    if not series_ids:
        raise ValueError(f"No DICOM series found in {input_folder}")

    for sid in series_ids:
        # Get file names for this series
        file_list = reader.GetGDCMSeriesFileNames(input_folder, sid)
        reader.SetFileNames(file_list)
        # Read the series (stack slices into a volume)
        image = reader.Execute()

        # Derive output file name from series metadata
        ds = pydicom.dcmread(file_list[0], stop_before_pixels=True)
        series_num = getattr(ds, 'SeriesNumber', sid)
        series_desc = getattr(ds, 'SeriesDescription', f"Series{series_num}")
        # Sanitize filename: remove spaces and forbidden characters
        series_desc = series_desc.strip().replace(" ", "_").replace("/", "_")
        out_name = f"{series_num}_{series_desc}.nrrd"
        out_path = os.path.join(output_folder, out_name)

        # Write NRRD, preserving spatial metadata
        sitk.WriteImage(image, out_path)
        print(f"Converted {sid} to {out_name}")
    print(f"All series converted to NRRD format in {output_folder}")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert DICOM folder to NRRD format.")
    parser.add_argument("--input_folder", type=str, help="Path to the input DICOM folder.")
    parser.add_argument("--output_folder", type=str, help="Path to the output NRRD folder.")
    args = parser.parse_args()
    
    dcmfolder_to_nrrd(args.input_folder, args.output_folder)