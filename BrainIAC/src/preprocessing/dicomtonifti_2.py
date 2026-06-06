import SimpleITK as sitk
import os
import argparse
import glob
from tqdm import tqdm

def convert_dicom_series_to_nifti(dicom_dir, output_file):
    """
    Convert a single DICOM series to NIFTI format
    Args:
        dicom_dir: Directory containing DICOM files for one scan
        output_file: Output NIFTI file path
    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        reader = sitk.ImageSeriesReader()
        
        # get all the scans in the dir
        dicom_files = sorted(glob.glob(os.path.join(dicom_dir, "*.dcm")))
        
        if not dicom_files:
            print(f"No DICOM files found in: {dicom_dir}")
            return False
            
        reader.SetFileNames(dicom_files)
        
        # load dicom images
        image = reader.Execute()
        
        
        sitk.WriteImage(image, output_file)
        return True
        
    except Exception as e:
        print(f"Error converting {dicom_dir}: {str(e)}")
        return False

def convert_dicom_to_nifti(input_dir, output_dir):
    """
    Convert multiple DICOM series to NIFTI format
    Args:
        input_dir: Root directory containing subdirectories of DICOM series
        output_dir: Output directory for NIFTI files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    
    print(f"Looking for DICOM series in: {input_dir}")
    
    # Check if directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a directory")
        return
    
    
    scan_dirs = [d for d in os.listdir(input_dir) 
                 if os.path.isdir(os.path.join(input_dir, d))]
    
    if not scan_dirs:
        print("No subdirectories found in the input directory")
        return
    
    print(f"Found {len(scan_dirs)} potential scan directories")
    
    # Process each scan directory
    successful = 0
    failed = 0
    
    for scan_dir in tqdm(scan_dirs, desc="Converting scans"):
        input_path = os.path.join(input_dir, scan_dir)
        output_file = os.path.join(output_dir, f"{scan_dir}.nii.gz")
        
        if convert_dicom_series_to_nifti(input_path, output_file):
            successful += 1
        else:
            failed += 1
    
    print("\nConversion Summary:")
    print(f"Successfully converted: {successful} scans")
    print(f"Failed conversions: {failed} scans")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DICOM series to NIFTI format")
    parser.add_argument("--input", "-i", required=True, 
                      help="Input directory containing subdirectories of DICOM series")
    parser.add_argument("--output", "-o", required=True, 
                      help="Output directory for NIFTI files")
    
    args = parser.parse_args()
    
    convert_dicom_to_nifti(args.input, args.output)