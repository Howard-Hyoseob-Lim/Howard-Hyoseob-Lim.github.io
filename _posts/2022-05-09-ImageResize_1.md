---
layout : single
title : "Image Resizing Code for Online shopping mall product"
categories : python
tag : [python, image, business, cv2]
toc : true
publised : true
author_profile : false
---

# 이미 폴더링된 이미지 파일을 Re-sizing 해보자

## 1) 온라인 쇼핑몰마다 이미지 규격이 달라요

이커머스 현업에서는 각 쇼핑몰별로 정책으로 설정한 상품이미지의 규격이 다르다.
현재 내가 일하고 있는 쇼핑몰의 상품이미지 규격은 640 x 960으로 세로형이미지를 사용한다.
따라서, 보통 정방형(정사각형)의 이미지를 그대로 가져다 사용한다면
이미지가 깨져보일 수 있기 때문에 적절하게 리사이징을 하여 사용해야 한다.

따라서 이번 게시물에서는 현업에서 어떻게 이미지를 리사이징하여 사용하는 지 알아볼 것이다.

리사이징에 사용할 타겟 파일은 아래와 같은 구조를 가진다.

타겟폴더 하위로<br>
N개의 상품코드(폴더) 하위로<br>
N개의 상품코드(이미지)
- 상품코드(이미지1) <br>
- 상품코드(이미지2) <br>
- 상품코드(이미지3)

※이번 코드에서 리사이징하는 방식은
정사각형에서 세로형으로 변경하면서
빈공간을 흰색으로 채우는 방식이다.

※해당 작업을 엑셀파일을 통해서 상품 업로드 유무를 파악하고, <br>
폴더의 이름을 변경한 뒤에 폴더 이름에 따라 파일이름을 변경하는 프로세스임.<br>


```python
import os
import glob
import cv2
import shutil
import numpy as np
os.chdir('파일 경로')
path_dir = r'파일 경로'
fileExt = r".jpg"
```


```python
item_list = os.listdir(path_dir)
```


```python
item_list
```


```python
os.getcwd()
```


```python
def resize(image, item_list, num):
    image_re = str(item_list) + "_THNAIL_" + str(num) + ".jpg"
    img = cv2.imread(image)
    if img is None:
        pass
        print(item_list)
        print("Wrong path:",image)
    else:
        img = cv2.resize(img, dsize=(600, 600), interpolation=cv2.INTER_LINEAR)
        old_image_height, old_image_width, channels = img.shape

        # create new image of desired size and color (blue) for padding
        new_image_width = 640
        new_image_height = 960
        # full with white color
        color = (255,255,255)
        result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)
        # compute center offset
        x_center = (new_image_width - old_image_width) // 2
        y_center = (new_image_height - old_image_height) // 2

        # copy img image into center of result image
        result[y_center:y_center+old_image_height, x_center:x_center+old_image_width] = img
        # save result
        cv2.imwrite(image_re, result)
```


```python
for i in range(len(item_list)):
    item_list_dirinner = path_dir + "\\\\" + item_list[i]
    file_name = item_list[i]
    #현재 경로 위치를 지정해주고, 파일이름으로 imread를 해야 먹힘
    os.chdir(item_list_dirinner)
    #print(item_list_dirinner)
    #print(type(item_list_dirinner))
    target = [ _ for _ in os.listdir(item_list_dirinner) if _.endswith(fileExt)]
    #print(target)
    list_num = range(0,len(target))
    num_list = list(list_num)
    remove_target = [ _ for _ in os.listdir(item_list_dirinner) if not _.endswith(fileExt)]
    if remove_target == 0:
        print("no_remove_target")
    else:
        for j in remove_target:
            remove_path = item_list_dirinner + "\\\\" + j
            if os.path.isdir(remove_path):
                shutil.rmtree(remove_path)
            else:
                pass
    for k, l in zip(target, num_list):
        #print(k)
        #print(k)
        #item_list_dirinner_file = item_list_dirinner + "\\\\" + k
        resize(k, file_name, l)
        os.remove(k)
    print("{} 완료".format(item_list[i]))
print("변환 완료")
```

# -----------------------------------------
