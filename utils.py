import scipy.misc, numpy as np, os, sys

def save_img(out_path, img):
    '''
    @Descripttion: 保存图像
    @param {type} 
    @return: 
    '''
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(out_path, img)

def scale_img(style_path, style_scale):
    '''
    @Descripttion: 设置图像的尺寸
    @param {type} 
    @return: 
    '''
    scale = float(style_scale)
    o0, o1, o2 = scipy.misc.imread(style_path, mode='RGB').shape
    scale = float(style_scale)
    new_shape = (int(o0 * scale), int(o1 * scale), o2)
    style_target = _get_img(style_path, img_size=new_shape)
    return style_target

def get_img(src, img_size=False):
    '''
    @Descripttion: 通过链接获取图像
    @param {type} 
    @return: 
    '''
    img = scipy.misc.imread(src, mode='RGB') # misc.imresize(, (256, 256, 3))
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img,img,img))
    if img_size != False:
        img = scipy.misc.imresize(img, img_size)
    return img

def exists(p, msg):
    assert os.path.exists(p), msg

def list_files(in_path):
    '''
    @Descripttion: 列出所有文件
    @param {type} 
    @return: 
    '''
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break

    return files