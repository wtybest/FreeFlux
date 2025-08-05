import cv2
import os
from tqdm import tqdm

psnr_dict = {f"block_{i}": [] for i in range(57)}

target_dir = 'rope_change_dataset_dir'
prompt_list = os.listdir(target_dir)

for each_prompt in tqdm(prompt_list):
    seed_list = os.listdir(os.path.join(target_dir, each_prompt))
    for each_seed in seed_list:
        sample_img = cv2.imread(os.path.join(target_dir, each_prompt, each_seed, 'sample.jpg'))
        position_list = [f for f in os.listdir(os.path.join(target_dir, each_prompt, each_seed)) if os.path.isdir(os.path.join(target_dir, each_prompt, each_seed, f))]
        for each_position in position_list:
            layer_name_list = os.listdir(os.path.join(target_dir, each_prompt, each_seed, each_position))
            for each_layer_name in layer_name_list:
                layer_img = cv2.imread(os.path.join(target_dir, each_prompt, each_seed, each_position, each_layer_name))
                layer_idx = "block_"+os.path.splitext(each_layer_name)[0]
                layer_psnr_value = cv2.PSNR(sample_img, layer_img)
                psnr_dict[layer_idx].append(layer_psnr_value)


averages = {dict_key: sum(dict_value) / len(dict_value) for dict_key, dict_value in psnr_dict.items()}
sorted_keys = sorted(averages, key=averages.get, reverse=True) #, reverse=True
for dict_key in sorted_keys:
    print(dict_key, averages[dict_key])


