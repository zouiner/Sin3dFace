import os

def create_image_path_txt(image_directory, output_txt_file):
    try:
        # Check if the directory exists
        if not os.path.exists(image_directory):
            raise FileNotFoundError(f"Directory not found: {image_directory}")

        # Open the output text file
        with open(output_txt_file, 'w') as file:
            # Walk through the directory and list all image files
            for root, _, files in os.walk(image_directory):
                for filename in files:
                    # Filter image files based on extensions
                    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                        # Construct the full path
                        full_path = os.path.join(root, filename)
                        # Write to the file
                        file.write(full_path + '\n')
        print(f"Image paths saved to {output_txt_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
image_directory = "/users/ps1510/scratch/Programs/SinSR/testdata/RealSet65"  # Replace with the actual image directory path
output_txt_file = "/users/ps1510/scratch/Programs/SinSR/traindata/image_paths.txt"  # Replace with the desired output text file name
create_image_path_txt(image_directory, output_txt_file)