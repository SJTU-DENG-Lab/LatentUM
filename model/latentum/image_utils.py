import os
import json
import torch
import random
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode


class LLaVAMix665K(torch.utils.data.Dataset):
    def __init__(self, img_path, ann_path, des_path):
        self.img_path = img_path
        self.ann_path = ann_path
        llava_data = json.load(open(ann_path, "r"))
        gpt_data = json.load(open(des_path, "r"))
        data = llava_data + gpt_data
        print(f"llava_data: {len(llava_data)}, gpt_data: {len(gpt_data)}")
        print(f"data: {len(data)}")
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if "image" in data:
            # load image
            img_path = os.path.join(self.img_path, data["image"])
            # load Q&A pair
            num_qa_pair = len(data["conversations"]) // 2
            qa_index = random.randint(0, num_qa_pair - 1)
            assert data["conversations"][2*qa_index]["from"] == "human"
            assert data["conversations"][2*qa_index+1]["from"] == "gpt"
            question = data["conversations"][2*qa_index]["value"]
            answer = data["conversations"][2*qa_index+1]["value"]

            if "<image>\n" in question:
                question = question.replace("<image>\n", "")
            elif "\n<image>" in question:
                question = question.replace("\n<image>", "")

            item = {
                "question": question,
                "answer": answer,
                "image": img_path,
            }
            return item
        else:
            item = {
                "question": None,
                "answer": None,
            }
            return item

def get_llava_mix665k_dataloader(config, tokenizer):
    from .internvl.conversation import get_conv_template

    IMG_START_TOKEN = "<img>"
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    IMG_END_TOKEN = "</img>"

    def collate_fn(batch):
        pixel_values = []
        input_ids = []
        attention_mask = []
        answer_mask = []

        for item in batch:
            if "image" not in item:
                continue

            image_path = item["image"]
            question = item["question"]
            answer = item["answer"]

            pixel_value = load_image(image_path, max_num=12)
            if pixel_value is None:
                continue
            question = "<image>\n" + question

            template = get_conv_template("internvl2_5")
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            num_patches_list = [pixel_value.shape[0]] if pixel_value is not None else []

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * config.num_image_token * num_patches_list[0] + IMG_END_TOKEN
            query = query.replace("<image>", image_tokens, 1)

            query_tokens = tokenizer(query, return_tensors="pt")
            answer_tokens = tokenizer(answer, return_tensors="pt")

            tokenizer_output = tokenizer(
                query + answer,
                return_tensors = "pt",
                padding        = "max_length",
                padding_side   = "right",
                truncation     = True,
                max_length     = config.max_seq_length,
            )

            input_ids_batch = tokenizer_output["input_ids"]
            attention_mask_batch = tokenizer_output["attention_mask"]
            
            # answer_mask是指，在input_ids_batch中，属于answer的部分，为True
            # 计算query和answer的长度
            query_length = query_tokens["input_ids"].shape[1]
            answer_length = answer_tokens["input_ids"].shape[1]
            
            # 创建answer_mask，初始化为False
            answer_mask_batch = torch.zeros_like(input_ids_batch, dtype=torch.bool)
            
            # 在答案部分设置为True（考虑padding和截断的影响）
            if query_length + answer_length <= config.max_seq_length:
                # 如果没有被截断，直接设置答案部分
                answer_mask_batch[0, query_length - 1:query_length + answer_length - 1] = True
            else:
                continue

            # 收集到列表中
            pixel_values.append(pixel_value)
            input_ids.append(input_ids_batch[0])  # 移除batch维度
            attention_mask.append(attention_mask_batch[0])  # 移除batch维度
            answer_mask.append(answer_mask_batch[0])  # 移除batch维度

        if len(pixel_values) == 0:
            return None
        else:
            pixel_values = torch.stack(pixel_values).squeeze(1)
            input_ids = torch.stack(input_ids)
            attention_mask = torch.stack(attention_mask)
            answer_mask = torch.stack(answer_mask)

            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "answer_mask": answer_mask,
            }


    dataloader = torch.utils.data.DataLoader(
        LLaVAMix665K(config.img_path, config.ann_path, config.des_path),
        batch_size  = config.batch_size,
        shuffle     = True,
        num_workers = config.num_workers,
        pin_memory  = True,
        drop_last   = True,
        collate_fn  = collate_fn,
    )

    return dataloader


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    if isinstance(image_file, Image.Image):
        image = image_file
    else:
        try:
            image = Image.open(image_file).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_file}: {e}")
            return None
    image = image.resize((input_size, input_size))
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

@torch.no_grad()
def load_image_llamagen_recon(image_file, tok, input_size=448, max_num=12):
    from torchvision.transforms import ToTensor, ToPILImage

    if isinstance(image_file, Image.Image):
        image = image_file
    else:
        try:
            image = Image.open(image_file).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_file}: {e}")
            return None

    image = image.resize((input_size, input_size))
    image = ToTensor()(image).unsqueeze(0).cuda()
    image = image * 2.0 - 1.0
    # reconstruction
    dec, diff = tok(image)
    dec = ((dec + 1) / 2).clamp(0, 1)
    rec = ToPILImage()(dec[0])

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(rec, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values
