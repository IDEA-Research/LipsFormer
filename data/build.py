# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from timm.data.transforms import _pil_interp

from .cached_image_folder import CachedImageFolder
from .samplers import SubsetRandomSampler

from PIL import Image
import random

img_list = ["/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02850732/n02850732_2993.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n01620735/n01620735_1406.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n10235024/n10235024_13754.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03221351/n03221351_4401.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03171228/n03171228_397.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02119477/n02119477_1663.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04963307/n04963307_2301.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n10746931/n10746931_8057.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n07931001/n07931001_6055.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03046802/n03046802_1837.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03246933/n03246933_3240.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03397266/n03397266_5192.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n10559288/n10559288_6980.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n07723559/n07723559_4116.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03242506/n03242506_1505.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n07938149/n07938149_6354.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03914438/n03914438_2967.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02692877/n02692877_4397.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04371563/n04371563_10681.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03012897/n03012897_6751.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04474035/n04474035_16466.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03358380/n03358380_15087.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02063224/n02063224_1716.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04477548/n04477548_4044.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03956922/n03956922_29989.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03026907/n03026907_3732.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n11968704/n11968704_2797.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03683708/n03683708_878.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03070059/n03070059_6553.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n09475179/n09475179_24992.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n12154773/n12154773_6166.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04067818/n04067818_9697.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n12277578/n12277578_9559.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n11924445/n11924445_21.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02867715/n02867715_2009.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n09895561/n09895561_3625.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03503233/n03503233_6198.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n12749049/n12749049_5193.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n11552133/n11552133_3360.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n11815918/n11815918_3130.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n00466273/n00466273_6177.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n11879722/n11879722_12028.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n07722485/n07722485_9567.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n07917272/n07917272_12.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n01636352/n01636352_2089.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04293119/n04293119_1043.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n10215623/n10215623_8286.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n12273114/n12273114_8238.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03965456/n03965456_36579.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02942699/n02942699_2539.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04008634/n04008634_13730.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02535258/n02535258_3381.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02639605/n02639605_2470.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n12174311/n12174311_27887.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02233943/n02233943_10227.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03235042/n03235042_5201.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03978966/n03978966_5819.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n10055410/n10055410_56807.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02987492/n02987492_19243.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02120079/n02120079_850.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n01665541/n01665541_15659.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n00441073/n00441073_8055.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02933649/n02933649_3546.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02398521/n02398521_2232.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04590021/n04590021_6496.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n09901502/n09901502_4343.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n10120330/n10120330_3443.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04354182/n04354182_419.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03308152/n03308152_18425.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04467307/n04467307_111.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04312654/n04312654_984.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03782006/n03782006_904.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n12822955/n12822955_7952.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02514041/n02514041_7676.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n12282737/n12282737_3800.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04045397/n04045397_2680.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n01730563/n01730563_731.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n12886600/n12886600_290.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02095050/n02095050_1395.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n09838370/n09838370_10936.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n09335809/n09335809_3543.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n10369095/n10369095_3450.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03173387/n03173387_1289.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03448956/n03448956_5799.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n01756291/n01756291_7691.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n01494041/n01494041_1193.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n12704343/n12704343_7244.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03131574/n03131574_1447.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n10686885/n10686885_7054.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02379630/n02379630_2603.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03356858/n03356858_35752.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02166229/n02166229_2974.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04028764/n04028764_13572.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03236217/n03236217_2899.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n11707827/n11707827_8522.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03488438/n03488438_8538.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n07696403/n07696403_5887.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03773835/n03773835_6601.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n10628644/n10628644_24240.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n15091846/n15091846_20229.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03168107/n03168107_11756.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04161981/n04161981_12086.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n09691729/n09691729_11429.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02274024/n02274024_1911.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03349599/n03349599_7829.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n07685730/n07685730_5140.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n09606527/n09606527_11698.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02565324/n02565324_7513.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n06273986/n06273986_1617.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02516188/n02516188_6015.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n10663315/n10663315_84228.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02776205/n02776205_20359.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04457767/n04457767_2794.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03294833/n03294833_13951.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n12290748/n12290748_9816.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n11872146/n11872146_2301.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03650551/n03650551_14330.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02863536/n02863536_4946.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n07897975/n07897975_468.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03176594/n03176594_5221.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03041632/n03041632_1585.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03927539/n03927539_3213.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n10205231/n10205231_12889.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03684143/n03684143_9146.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03121431/n03121431_8120.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02966687/n02966687_366.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04258438/n04258438_1540.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n07694659/n07694659_2709.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03819448/n03819448_7383.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n08596076/n08596076_4388.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04328946/n04328946_6914.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n13209808/n13209808_4896.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04097866/n04097866_1798.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03488887/n03488887_7862.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03607923/n03607923_4351.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n09326662/n09326662_57025.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n11794519/n11794519_2950.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n01847089/n01847089_845.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03871724/n03871724_6918.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n12089320/n12089320_3691.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n12340755/n12340755_12550.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n11733548/n11733548_919.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n09752023/n09752023_3639.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04173907/n04173907_9518.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03489162/n03489162_2605.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n11915214/n11915214_13218.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04230808/n04230808_5712.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n09688804/n09688804_8222.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n09303528/n09303528_4109.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n09899671/n09899671_8994.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03766322/n03766322_9922.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03735963/n03735963_895.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02028900/n02028900_10457.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n07825972/n07825972_20127.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n11823436/n11823436_5492.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04325704/n04325704_4446.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n07852614/n07852614_12381.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03291819/n03291819_4406.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02820556/n02820556_1251.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n09824135/n09824135_20248.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n01538630/n01538630_4997.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02086646/n02086646_1757.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n07928367/n07928367_3911.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02588286/n02588286_5861.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03201776/n03201776_49961.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03238586/n03238586_4664.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n09384106/n09384106_836.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03522634/n03522634_10351.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n12204032/n12204032_944.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n07693048/n07693048_7247.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n11939491/n11939491_2793.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n00447540/n00447540_18292.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04104770/n04104770_11746.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n10634849/n10634849_3228.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n12018271/n12018271_1114.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02794474/n02794474_4249.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02607201/n02607201_7531.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04607869/n04607869_5381.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04005630/n04005630_27217.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02180427/n02180427_1380.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n11974888/n11974888_1542.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n12276477/n12276477_2656.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04557648/n04557648_4273.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n11753700/n11753700_17593.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n11937360/n11937360_186.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n07891726/n07891726_29350.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03560430/n03560430_19067.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n11925898/n11925898_5267.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n07804543/n07804543_3998.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n07727048/n07727048_4931.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03490884/n03490884_3506.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n07842605/n07842605_6852.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n09410224/n09410224_6048.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03415252/n03415252_96.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02588286/n02588286_9216.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04398951/n04398951_5196.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n09836160/n09836160_8048.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04338963/n04338963_5267.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n07732168/n07732168_3666.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n07609840/n07609840_5111.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04209613/n04209613_13205.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n12320010/n12320010_4231.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n09715427/n09715427_5520.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n09764201/n09764201_572.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n12951835/n12951835_6002.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02094258/n02094258_817.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n09742315/n09742315_2444.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03326795/n03326795_4299.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n07609632/n07609632_16687.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02842809/n02842809_4163.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04525305/n04525305_1580.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n00463543/n00463543_7946.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04284572/n04284572_2500.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n01950731/n01950731_10364.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n13128582/n13128582_3534.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03696301/n03696301_11386.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n11903671/n11903671_27291.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n11723770/n11723770_9131.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03587205/n03587205_8466.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02816768/n02816768_2802.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03391770/n03391770_15492.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04289576/n04289576_1682.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n00449054/n00449054_2120.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03488438/n03488438_4214.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n01664492/n01664492_11610.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n00475273/n00475273_1660.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04185529/n04185529_5241.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n09337253/n09337253_6577.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04056180/n04056180_906.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03279153/n03279153_2136.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n12306089/n12306089_2006.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02475078/n02475078_9503.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n01880716/n01880716_472.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n10147262/n10147262_4054.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03016953/n03016953_3504.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03994614/n03994614_6041.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03390786/n03390786_761.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04186268/n04186268_7177.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02756977/n02756977_4583.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n11939491/n11939491_5343.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n07907161/n07907161_1421.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n09991867/n09991867_97625.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04968139/n04968139_10250.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n07597145/n07597145_11944.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n07771891/n07771891_7157.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04546194/n04546194_8836.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02296276/n02296276_3774.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n07861983/n07861983_3133.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03610418/n03610418_6995.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03652729/n03652729_7068.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04403524/n04403524_7010.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04045397/n04045397_19732.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03003091/n03003091_8141.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02086753/n02086753_7482.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n12162181/n12162181_5310.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03914337/n03914337_14346.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n12256920/n12256920_2574.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02107574/n02107574_3335.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02802544/n02802544_9124.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02907082/n02907082_20650.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n15019030/n15019030_18034.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n01445429/n01445429_246.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n10493419/n10493419_8350.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n10262445/n10262445_12908.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03783430/n03783430_520.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n11857875/n11857875_5315.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04118635/n04118635_285.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n04395106/n04395106_5902.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n07768694/n07768694_2282.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02836035/n02836035_8908.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02213788/n02213788_2171.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03898395/n03898395_2471.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n07723330/n07723330_5132.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03543394/n03543394_1381.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n11757653/n11757653_2602.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03369276/n03369276_868.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n09732170/n09732170_5420.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03793850/n03793850_7494.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02952485/n02952485_17123.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02754656/n02754656_11408.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02084732/n02084732_15806.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03105306/n03105306_738.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n13061348/n13061348_1633.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n11736694/n11736694_2448.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n03766044/n03766044_15114.JPEG", "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image/n02275560/n02275560_786.JPEG"]

def img_loader(img_path):
    try:
        return Image.open(img_path).convert('RGB')
    except:
        img_path = random.choice(img_list)
        return Image.open(img_path).convert('RGB')



def build_22k_loader(config):
    config.defrost()

    transform = build_transform(True, config)
    root = "/comp_robot/cv_public_dataset/imagenet22k/imgnet22k-image"
    dataset_train = datasets.ImageFolder(root, transform=transform, loader=img_loader)
    config.MODEL.NUM_CLASSES = 21842
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, data_loader_train, mixup_fn


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
    sampler_val = SubsetRandomSampler(indices)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # if config.DATA.TRANSFORM_TYPE != 'easy':
            # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform
        # else:
        #     t = []
        #     t.append(
        #         transforms.Resize(config.DATA.IMG_SIZE + 32, interpolation=_pil_interp(config.DATA.INTERPOLATION)))
        #     t.append(transforms.RandomCrop(config.DATA.IMG_SIZE))
        #     t.append(transforms.RandomHorizontalFlip())
        #     t.append(transforms.ToTensor())
        #     t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        #     return transforms.Compose(t)

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
