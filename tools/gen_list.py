import os

def write_image_paths(directory, output_file):
    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(directory):
            if files:
                for file in files:
                    if file.endswith('.jpg') or file.endswith('.png'):
                        image_path = os.path.join(root, file)
                        image_path = os.path.abspath(image_path)
                        img_target = os.path.dirname(image_path)
                        img_target = os.path.basename(img_target)
                        f.write(f"{image_path} {img_target} \n")
                        
directory = ''
output_file = ''

write_image_paths(directory, output_file)
