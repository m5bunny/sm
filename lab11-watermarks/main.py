import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def mse(i, k):
    val = np.float64(0)
    i = i.astype(np.int32)
    k = k.astype(np.int32)
    for x in range(i.shape[0]):
        for y in range(i.shape[1]):
            for z in range(i.shape[2]):
                val += (np.abs(i[x,y,z]-k[x,y,z]))**2

    return val / (np.prod(i.shape))

def nmse(i, k):
    return mse(i, k) / mse(np.zeros(k.shape), k)

def psnr(i, k):
    return 10 * np.log10(255**2 / mse(i, k))

def m_if(i, k):
    i = i.astype(np.int32)
    k = k.astype(np.int32)
    multiSum = np.float64(0)
    for x in range(i.shape[0]):
        for y in range(i.shape[1]):
            for z in range(i.shape[2]):
                multiSum += i[x,y,z] * k[x,y,z]

    sums = np.float64(0)
    for x in range(i.shape[0]):
        for y in range(i.shape[1]):
            for z in range(i.shape[2]):
                sums += (i[x,y,z] + k[x,y,z])**2

    return 1-(sums / multiSum)


def m_ssim(i, k):
    return ssim(i, k, channel_axis=2)

def put_data(img,data,binary_mask=np.uint8(1)):
    assert img.dtype==np.uint8 , "img wrong data type"
    assert binary_mask.dtype==np.uint8, "binary_mask wrong data type"
    un_binary_mask=np.unpackbits(binary_mask)
    if data.dtype!=bool:
        unpacked_data=np.unpackbits(data)
    else:
        unpacked_data=data
    dataspace=img.shape[0]*img.shape[1]*np.sum(un_binary_mask)
    assert (dataspace>=unpacked_data.size) , "too much data"
    if dataspace==unpacked_data.size:
        prepered_data=unpacked_data.reshape(img.shape[0],img.shape[1],np.sum(un_binary_mask).astype(np.uint8)).astype(np.uint8)
    else:
        prepered_data=np.resize(unpacked_data,(img.shape[0],img.shape[1],np.sum(un_binary_mask).astype(np.uint8))).astype(np.uint8)
    mask=np.full((img.shape[0],img.shape[1]),binary_mask)
    img=np.bitwise_and(img,np.invert(mask))
    bv=0
    for i,b in enumerate(un_binary_mask[::-1]):
        if b:
            temp=prepered_data[:,:,bv]
            temp=np.left_shift(temp,i)
            img=np.bitwise_or(img,temp)
            bv+=1
    return img

def pop_data(img,binary_mask=np.uint8(1),out_shape=None):
    un_binary_mask=np.unpackbits(binary_mask)
    data=np.zeros((img.shape[0],img.shape[1],np.sum(un_binary_mask))).astype(np.uint8)
    bv=0
    for i,b in enumerate(un_binary_mask[::-1]):
        if b:
            mask=np.full((img.shape[0],img.shape[1]),2**i)
            temp=np.bitwise_and(img,mask)           
            data[:,:,bv]=temp[:,:].astype(np.uint8)             
            bv+=1
    if out_shape!=None:
        tmp=np.packbits(data.flatten())        
        tmp=tmp[:np.prod(out_shape)]
        data=tmp.reshape(out_shape)
    return data

def string_to_uint8_array(string):
    uint8_array = np.array([], dtype=np.uint8)
    for char in string:
        ascii_value = ord(char)
        uint8_value = np.uint8(ascii_value)
        uint8_array = np.append(uint8_array, uint8_value)
    return uint8_array

def uint8_array_to_string(uint8_array):
    char_array = [chr(uint8) for uint8 in uint8_array]
    string = ''.join(char_array)
    return string

def convert_img_to_mask(path):

    img = plt.imread(path)
    mask = np.full([img.shape[0], img.shape[1], 1], True)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j][0] == 255:
                mask[i][j] = False
            else:
                mask[i][j] = True
    
    return mask

img = plt.imread('./img.jpg');

text1 = string_to_uint8_array("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed nibh metus, tempor ac urna vel, vestibulum tempus elit. Ut vulputate iaculis dui, id fermentum dolor. Nam lobortis maximus feugiat. Quisque malesuada laoreet est mattis lobortis. Integer egestas diam purus, vel tincidunt nisi eleifend in. Pellentesque erat felis, aliquet in molestie rhoncus, hendrerit et dolor. Sed accumsan aliquam dui, feugiat molestie sem tincidunt eget. Sed at libero a neque tincidunt rhoncus id vel velit. Nunc auctor finibus nisi, eu pulvinar ipsum mattis eu. Cras mi libero, semper sed diam scelerisque, rhoncus elementum dolor. Curabitur facilisis mauris est. Aliquam lacinia quam vel pharetra maximus. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec porttitor sem sit amet est iaculis, facilisis interdum diam pretium.")
text2 = string_to_uint8_array("In sodales mollis auctor. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin nec hendrerit libero. Fusce vehicula consectetur consequat. Cras dignissim finibus vulputate. Sed molestie pharetra lorem ac porttitor. Maecenas tincidunt ligula sit amet venenatis semper. In lobortis luctus libero in commodo. Duis ante elit, blandit at aliquam non, tempus vitae dolor. In eu sapien ex. Morbi euismod facilisis erat. Nullam laoreet augue est, vel varius magna porttitor eu. Integer porttitor porta turpis et elementum. Cras elit diam, pretium at nulla nec, vulputate ullamcorper turpis. Nunc vulputate diam ut erat molestie vehicula. ");
text3 = string_to_uint8_array("Praesent vehicula nulla porttitor risus malesuada convallis. Integer vestibulum gravida nisi, sit amet iaculis nunc porttitor volutpat. Curabitur fringilla ac tortor quis pellentesque. Donec mattis, tellus nec ornare facilisis, enim diam porttitor lectus, sed lacinia ante magna quis erat. Pellentesque eget augue ac lacus varius finibus. In id nunc interdum urna pellentesque cursus. Curabitur velit turpis, malesuada sagittis pellentesque vel, lacinia non lorem. Nullam dignissim mattis lorem nec mattis. Nullam est nisi, sagittis sed augue et, pulvinar porttitor augue. Mauris nec pulvinar nibh, nec dictum eros. Mauris est ante, laoreet non imperdiet eu, porttitor eget elit. ")

img_with_text_0 = put_data(img[:,:,0], text1, np.uint8(12));
img_with_text_1 = put_data(img[:,:,1], text2, np.uint8(12));
img_with_text_2 = put_data(img[:,:,2], text3, np.uint8(12));

img_with_text = np.dstack([img_with_text_0, img_with_text_1, img_with_text_2])

mse_value = round(mse(img, img_with_text), 2);
nmse_value = round(nmse(img, img_with_text), 2);
psnr_value = round(psnr(img, img_with_text), 2);
m_if_value = round(m_if(img, img_with_text), 2);
m_ssim_value = round(m_ssim(img, img_with_text), 2);

plt.title(f'mse:{mse_value}, nmse:{nmse_value}, psnr:{psnr_value}, m_if:{m_if_value}, m_ssim:{m_ssim_value}')

plt.title('Original')
plt.savefig(f'orig.png')
plt.imshow(img_with_text)
plt.title('With coded text')
plt.savefig(f'with_coded.png')
print(uint8_array_to_string(pop_data(img_with_text[:,:,0], np.uint8(12), text1.shape)))
print(uint8_array_to_string(pop_data(img_with_text[:,:,1], np.uint8(12), text2.shape)))
print(uint8_array_to_string(pop_data(img_with_text[:,:,2], np.uint8(12), text3.shape)))

img_to_code = convert_img_to_mask('./watermark.jpg')
img_with_img_0= img[:,:,0];
img_with_img_1 = put_data(img[:,:,1], img_to_code, np.uint8(12));
img_with_img_2 = img[:,:,2]


img_with_img = np.dstack([img_with_img_0, img_with_img_1, img_with_img_2])
mse_value = round(mse(img, img_with_img), 2);
nmse_value = round(nmse(img, img_with_img), 2);
psnr_value = round(psnr(img, img_with_img), 2);
m_if_value = round(m_if(img, img_with_img), 2);
m_ssim_value = round(m_ssim(img, img_with_img), 2);
plt.title(f'mse:{mse_value}, nmse:{nmse_value}, psnr:{psnr_value}, m_if:{m_if_value}, m_ssim:{m_ssim_value}')
plt.imshow(img_with_img)
plt.savefig(f'with_coded_img.png')
plt.show()

def water_mark(img,mask,alpha=0.25):
    assert (img.shape[0]==mask.shape[0]) and (img.shape[1]==mask.shape[1]), "Wrong size"
    assert (mask.dtype==bool), "Wrong type - mask"
    if len(img.shape)<3:
        flag=True
        t_img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGBA)
    else:
        flag=False
        t_img=cv2.cvtColor(img,cv2.COLOR_RGB2RGBA)       
    t_mask=cv2.cvtColor((mask*255).astype(np.uint8),cv2.COLOR_GRAY2RGBA)
    t_out=cv2.addWeighted(t_img,1,t_mask,alpha,0)
    if flag:
        out=cv2.cvtColor(t_out,cv2.COLOR_RGBA2GRAY)
    else:
        out=cv2.cvtColor(t_out,cv2.COLOR_RGBA2RGB)
    return out


def analise_water_mark(img, mask, aplha):
    img_with_watermark = water_mark(img, mask, aplha)
    mse_value = round(mse(img, img_with_watermark), 2);
    nmse_value = round(nmse(img, img_with_watermark), 2);
    psnr_value = round(psnr(img, img_with_watermark), 2);
    m_if_value = round(m_if(img, img_with_watermark), 2);
    m_ssim_value = round(m_ssim(img, img_with_watermark), 2);

    plt.title(f'alpha:{aplha}, mse:{mse_value}, nmse:{nmse_value}, psnr:{psnr_value}, m_if:{m_if_value}, m_ssim:{m_ssim_value}')
    plt.imshow(img_with_watermark)
    plt.savefig(f'{aplha}-watermark.png')

img = plt.imread("./img.jpg");
mask = convert_img_to_mask("./watermark.jpg")

for i in range(25, 76, 5):
    analise_water_mark(img, mask, round(i / 100, 2))
