import os

def write_image_paths(directory1, directory2, output_file):
    folders1 = [os.path.join(directory1, folder) for folder in os.listdir(directory1) if os.path.isdir(os.path.join(directory1, folder))]
    with open(output_file, 'w') as f:
        for folder1 in folders1:
            folder_name = os.path.basename(folder1)
            image_files = [os.path.join(folder1, file) for file in os.listdir(folder1) if file.lower().endswith(('jpg'))]
            for image_file in image_files:
                try:
                    corresponding_folder = os.path.join(directory2, folder_name)
                    corresponding_image_file = os.path.join(corresponding_folder, '1.jpg')
                    f.write(f"{image_file} {folder_name} {corresponding_image_file}\n")
                    print(f"Processed {image_file}")
                except Exception as e:
                    print(f"Failed to process {image_file}: {e}")


directory1 = ""
directory2 = ""
output_file = ""

write_image_paths(directory1, directory2, output_file)
