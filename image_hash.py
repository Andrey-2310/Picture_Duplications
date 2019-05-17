from PIL import Image
import imagehash


original1 = Image.open('./pictures/coca-cola-default.jpg')
original2 = Image.open('./pictures/coca-cola-default-90.jpg')
hash1 = imagehash.dhash(original1)
hash2 = imagehash.dhash(original2)
print(hash1)
print(hash2)
print(hash1 - hash2)