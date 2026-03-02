# GROUP README – Dataset Instructions
Project: Multi-Temporal Change Detection for Urban Expansion Using CNN

---

## 1. Dataset: LEVIR-CD

Official Website:
https://www.kaggle.com/datasets/mdrifaturrahman33/levir-cd


Dataset Name:
LEVIR-CD (Large-scale building change detection dataset)

Description:
LEVIR-CD contains paired high-resolution satellite images of the same region at two different times (T1 and T2) along with pixel-wise binary change masks.

Task:
Binary Change Detection (Urban/Building expansion)

---

## 2. What Each Image Contains

For each sample:
- Image A (T1)
- Image B (T2)
- Ground Truth Mask (Change / No Change)

Mask:
- White (1) = Changed
- Black (0) = No Change

---

## 3. Download Instructions

1. Go to:
   https://www.kaggle.com/datasets/mdrifaturrahman33/levir-cd

2. Download the full LEVIR-CD dataset.

3. Extract it locally.

---

## 4. Folder Structure (IMPORTANT)

After extraction, place the dataset in:

project_root/data/LEVIR-CD/

Expected structure:

data/
 └── LEVIR-CD/
      ├── train/
      ├── val/
      ├── test/

(Adjust if dataset structure differs — we will standardize during preprocessing.)

---

## 5. DO NOT Upload Dataset to GitHub

The dataset is large (several GB).
DO NOT commit it to the repository.

Add to .gitignore:

data/LEVIR-CD/

Only upload:
- Code
- Preprocessing scripts
- Train/val/test split files
- Sample images (max 2–3 for demo)

---

## 6. Preprocessing Plan (Planned)

- Resize to 256x256 tiles
- Normalize RGB values
- Apply data augmentation:
  - Horizontal flip
  - Vertical flip
  - Rotation
- Convert masks to binary tensors

---

## 7. Notes

- Ensure image pairs remain aligned.
- Do not shuffle T1 and T2 independently.
- Class imbalance exists (few changed pixels) – we will use Dice + BCE loss.

---

If any issues occur during download or setup, notify the group immediately.