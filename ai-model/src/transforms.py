from torchvision import transforms


# ============================================================
# IMAGE SIZE
# ============================================================

IMAGE_SIZE = 224


# ============================================================
# IMAGENET NORMALIZATION
# ============================================================

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


# ============================================================
# TRAIN TRANSFORMS
# ============================================================

def get_train_transforms():

    return transforms.Compose([

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        # ----------------------------------------------------
        # Lightweight augmentation
        # ----------------------------------------------------

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=10
        ),

        # ----------------------------------------------------
        # Convert to tensor
        # ----------------------------------------------------

        transforms.ToTensor(),

        # ----------------------------------------------------
        # ImageNet normalization
        # ----------------------------------------------------

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


# ============================================================
# VALIDATION / TEST TRANSFORMS
# ============================================================

def get_eval_transforms():

    return transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])