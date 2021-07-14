import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

imgf = cbook.get_sample_data("ada.png")
img = plt.imread(imgf)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
ax1.imshow(img, origin = "upper")
ax2.imshow(img, origin = "lower")
# img.shape (803, 512, 4), then transpose 
# he first two axes from (0, 1, 2) to (1, 0, 2)
ax3.imshow(img.transpose(1, 0, 2), origin="upper")
ax4.imshow(img.transpose(1, 0, 2), origin="lower")