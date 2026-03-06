from src.utils.dataset import LevirCDDataset

ds = LevirCDDataset("data/LEVIR CD", "train", img_size=(256, 256))

print("Dataset length:", len(ds))

x, y, name = ds[0]

print("Input shape:", x.shape)   # Should be [6,256,256]
print("Mask shape:", y.shape)    # Should be [1,256,256]
print("File name:", name)