from pathlib import Path
import random

root_dir = Path('/home/ywan0794@acfr.usyd.edu.au/tartanair_tools/tartanair_kitti')

subfolders = [item for item in root_dir.iterdir() if item.is_dir()]

all_samples = []
easy_samples = []
hard_samples = []

for i, subfolder in enumerate(subfolders):
    print(f"Processing subfolder {i + 1}/{len(subfolders)}")
    
    sample_path = subfolder / subfolder.name / "imgs"
    img_lft_dir = sample_path / "image_02" / "data"
    
    file_names = [file.name for file in img_lft_dir.iterdir() if file.is_file()]
    
    for file_name in file_names:
        idx = file_name[:-4].lstrip('0')
        line = f"{sample_path.relative_to(root_dir)} {idx} l"
        all_samples.append(line)
        
        if "Easy" in subfolder.name:
            easy_samples.append(line)
        elif "Hard" in subfolder.name:
            hard_samples.append(line)

random.shuffle(all_samples)
num_train = int(0.9 * len(all_samples))

def write_txt(filename, sample_list):
    print(f"Writing {len(sample_list)} samples to {filename}")
    with open(filename, 'w') as f:
        for line in sample_list:
            f.write(line + '\n')

print(f"Total {len(all_samples)} samples, {len(easy_samples)} easy samples, {len(hard_samples)} hard samples")
write_txt('data_splits/tartanair/full.txt', all_samples)
write_txt('data_splits/tartanair/train.txt', all_samples[:num_train])
write_txt('data_splits/tartanair/test.txt', all_samples[num_train:])
write_txt('data_splits/tartanair/easy.txt', easy_samples)
write_txt('data_splits/tartanair/hard.txt', hard_samples)
