from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from gaussian_smoothing import GaussianSmoothing
from scipy.ndimage import label
from scipy.spatial import distance
import cv2
import matplotlib.pyplot as plt
def resize_and_concat_images(image_list, resize_width=512, resize_height=512):
    # Resize all images
    resized_images = [
        img.resize((resize_width, resize_height), Image.Resampling.LANCZOS)
        for img in image_list
    ]
    # Concatenate images horizontally
    total_width = len(resized_images) * resize_width
    concat_image = Image.new("RGB", (total_width, resize_height))
    
    for i, img in enumerate(resized_images):
        concat_image.paste(img, (i * resize_width, 0))
    return concat_image

def find_ascending_sequence(lists):
    result = []
    solutions_found = 0
    first_solution = None
    other_solution = []

    def backtrack(index, prev_value):
        nonlocal solutions_found, first_solution
        if index == len(lists):
            solutions_found += 1
            if solutions_found == 1:
                first_solution = result[:]
            elif solutions_found > 1:
                other_solution.append(copy.deepcopy(result))
                # raise ValueError("Find the more than one solutioin for t-5!")
            return
        for num in sorted(lists[index]):
            if prev_value is None or num > prev_value:
                result.append(num)
                backtrack(index + 1, num)
                result.pop()

    backtrack(0, None)
    if solutions_found == 1:
        return first_solution
    elif solutions_found > 1:
        other_solution.append(first_solution)
        tight_solutioin = find_tightest_list(other_solution)
        print(f'for t_5 token list {lists} finds {other_solution}, but return the tightest {tight_solutioin}')
        return tight_solutioin
    else:
        raise ValueError("There is no right solution for t-5!")

def get_index_from_subject(pipe, input_prompt, subject_word):
    t5_prompt_token_list = pipe.tokenizer_2.encode(input_prompt)
    t5_word_token_list = pipe.tokenizer_2.encode(subject_word)[0:-1]
    t5_word_token_list = [each_t5_token for each_t5_token in t5_word_token_list if each_t5_token not in [0,1,2,3]]
    t_5_token_candidate_list = []
    for each_t5_word_token in t5_word_token_list:
        t_5_token_candidate_list.append([i for i, x in enumerate(t5_prompt_token_list) if x == each_t5_word_token])
    final_index_list = find_ascending_sequence(t_5_token_candidate_list)
    return final_index_list


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)  

def get_mask_by_click(img_pil_np, predictor=None, point_list=[[500, 500], [600, 500], [700, 500]], lable_list=[1,1,1], device='cuda:0'):
    np_img = img_pil_np

    input_point = np.array(point_list)
    input_label = np.array(lable_list)
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        predictor.set_image(np_img)
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(np_img)
    show_mask(masks, ax[0])
    show_points(input_point, input_label, ax[0])
    ax[0].axis('off')

    result_mask = masks[0][:,:,None].repeat(3,axis=2).astype(np.uint8)
    ax[1].imshow(255*result_mask)
    ax[1].axis('off')
    plt.tight_layout()
    plt.show()
    return masks[0].astype(np.uint8)

def resize_and_get_coordinates(array, target_shape, kernel_size=11, shift=(0, 0)): 
    if kernel_size % 2 == 0 or kernel_size <= 0:
        raise ValueError("kernel_size must be odd")
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated_array = cv2.dilate(array, kernel, iterations=0)

    dilated_array = np.array(dilated_array, dtype=np.float32)
    resized_dilated_array = cv2.resize(dilated_array, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    resized_dilated_binary = (resized_dilated_array > 0.5).astype(np.uint8)
    dilated_ori_coordinates = list(zip(*np.where(resized_dilated_binary == 1)))
    dilated_ori_indices = [coord[0] * target_shape[1] + coord[1] for coord in dilated_ori_coordinates]

    shifted_indices = []
    max_row, max_col = target_shape

    for coord in dilated_ori_coordinates:
        shifted_row = coord[0] + shift[0]
        shifted_col = coord[1] + shift[1]
        if not (0 <= shifted_row < max_row and 0 <= shifted_col < max_col):
            raise ValueError(f"Shifted coordinate ({shifted_row}, {shifted_col}) is out of bounds.")
        shifted_indices.append(shifted_row * target_shape[1] + shifted_col)
    excluded_indices = [idx for idx in dilated_ori_indices if idx not in shifted_indices]

    total_indices = set(range(target_shape[0] * target_shape[1]))
    remaining_indices = list(total_indices - set(dilated_ori_indices) - set(shifted_indices))

    return dilated_ori_indices, shifted_indices, excluded_indices, remaining_indices


def remove_small_white_regions(binary_image: np.ndarray) -> np.ndarray:
    binary_image = np.where(binary_image > 128, 255, 0).astype(np.uint8)
    structure = np.ones((3, 3), dtype=np.int8)
    labeled_array, num_features = label(binary_image, structure)
    sizes = np.bincount(labeled_array.ravel())
    sizes[0] = 0
    if len(sizes) <= 1:
        return binary_image
    largest_label = sizes.argmax()
    filtered_image = np.where(labeled_array == largest_label, 1, 0).astype(np.uint8)
    return filtered_image

def min_max_normalize(attn_map):
    min_val = torch.min(attn_map)
    max_val = torch.max(attn_map)
    return (attn_map - min_val) / (max_val - min_val + 1e-6) 

def derive_fg_mask_from_attn(global_store, add_idx_list, mask_threshold=0.2, height=64, width=64):
    target_block_list = [40,41,42,43]
    target_step = 7
    target_attn_list = []
    for select_block in target_block_list:
    
        select_attn_1 = global_store[f'block_{select_block}'][0][target_step]
        select_attn_2 = global_store[f'block_{select_block}'][1][target_step]
        cross_select_attn = ((select_attn_1 + select_attn_2.transpose(0,1)) / 2).reshape(-1, height, width)[add_idx_list,:,:].unsqueeze(0)
        cross_select_attn_64 = F.interpolate(cross_select_attn, size=(height, width), mode='bilinear', align_corners=False).squeeze(0)

        t5_smooth_attn_list = []
        for token_idx in range(cross_select_attn_64.shape[0]):
            image = cross_select_attn_64[token_idx, :, :]
            smoothing = GaussianSmoothing(channels=1, kernel_size=3, sigma=0.5, dim=2)
            input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
            image = smoothing(input).squeeze(0).squeeze(0)
            t5_smooth_attn_list.append(image)
        target_attn_list.append(torch.mean(torch.stack(t5_smooth_attn_list), dim=0))
    avg_attn = min_max_normalize(torch.mean(torch.stack(target_attn_list), dim=0))
    derived_mask = (avg_attn > mask_threshold).int()
    return derived_mask

def farthest_point_sampling(points: list, num_samples: int):
    if num_samples >= len(points):
        return np.array(points)
    
    sampled = [points[np.random.randint(len(points))]]
    for _ in range(num_samples - 1):
        dists = distance.cdist(sampled, points, metric='euclidean').min(axis=0)
        farthest_idx = np.argmax(dists)
        sampled.append(points[farthest_idx])
    
    return np.array(sampled)


def sample_indices(binary_image: np.ndarray, value: int, num_samples: int, farthest=False):
    if value not in (0, 1):
        raise ValueError("Value must be 0 or 1.")
    indices = np.argwhere(binary_image == value) * 16
    if len(indices) == 0:
        raise ValueError("No matching values found in the image.")
    num_samples = min(num_samples, len(indices))
    if farthest:
        sampled_indices = farthest_point_sampling(indices, num_samples=num_samples)
    else:
        sampled_indices = indices[np.random.choice(len(indices), num_samples, replace=False)]
    return [[y, x] for x, y in sampled_indices.tolist()]

def erode_image(binary_image: np.ndarray, kernel_size: int = 3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=3)
    return eroded_image

def validate_fg_mask_mode(fg_mask_mode):
    if fg_mask_mode == 'auto':
        return True
    elif fg_mask_mode == 'manual':
        return False
    else:
        raise ValueError("fg_mask_mode must be 'auto' or 'manual'")
