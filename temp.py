import numpy as np
import matplotlib.pyplot as plt
import time
import cv2

def _dillate_axis(input, output, kernel_size):
    i = 0
    counter = 0
    while i < len(input)-1:
        i += 1
        if input[i]:
            counter = kernel_size
        if counter:
            counter -= 1
            output[i] = True


def fast_dillate(image, kernel_size):
    padded = np.pad(image,
                     [(kernel_size, kernel_size), (kernel_size, kernel_size)],
                     mode='constant')

    rows_dillate = np.zeros(padded.shape, dtype=np.bool)
    result = np.zeros(padded.shape, dtype=np.bool)
    for row_in, row_out in zip(padded, rows_dillate):
        _dillate_axis(row_in, row_out, kernel_size)
    for col_in, col_out in zip(rows_dillate.T, result.T):
        _dillate_axis(col_in, col_out, kernel_size)

    # remove padding
    # result = np.delete(result, range(0,kernel_size), axis=0)
    h =  kernel_size//2
    result = np.delete(result, range(0, kernel_size+h), axis=0)
    result = np.delete(result, range(0, kernel_size+h), axis=1)
    result = np.delete(result, range(-kernel_size+h, 0), axis=0)
    result = np.delete(result, range(-kernel_size+h, 0), axis=1)
    return result


input2d = np.zeros((2000, 2000), dtype=np.bool)
k = 100
input2d[25, 5] = 1
input2d[10, 10:15] = 1
input2d[10:15, 10] = 1
input2d[5:10, 20:25] = 1
for i in range(5):
    input2d[20 + i, 15 + i] = 1

fig, axes = plt.subplots(3)

axes[0].imshow(input2d)

start = time.time()
result = fast_dillate(input2d, k)
diff_fast = time.time() - start
axes[1].imshow(result)

kernel = np.ones((k,k),np.uint8)
img_32 = np.array(input2d,dtype=np.float32)
start = time.time()
cv2dil = cv2.dilate(src=img_32, kernel=kernel, )
diff_cv = time.time() - start

axes[2].imshow(cv2dil)
plt.show()

print("diff_fast " + str(diff_fast))
print("diff_cv " + str(diff_cv))