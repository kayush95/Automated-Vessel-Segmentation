import numpy as np
import os, sys
import time
import scipy.misc
from PIL import Image

def merge(images, size):
    h, w = images.shape[0], images.shape[1]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h: j * h + h, i * w: i * w + w, :] = image

    return img


def imsave(images, size, path):
    return scipy.misc.toimage(merge(images, size), cmax=1.0, cmin=0.0).save(path)

def inverse_transform(images):
    return (images + 1.) / 2.

def save_images(images, size, image_path, transform_req=False):
    if transform_req:
      return imsave(inverse_transform(images), size, image_path)
    else:
      return imsave(images, size, image_path)



def gt_patch_to_value(gt_patches, patch_size, n_samples):
 #" Method to convert ground truth patches to a scalar label based on central pixel value"
  gt_value = np.zeros((n_samples))
  for i in range(n_samples):
    label = gt_patches[i, patch_size/2-1, patch_size/2-1,:] # assuming even patch_size
    gt_value[i] = label
  return gt_value

def margin_size(patch_size):

  if patch_size % 2 ==0:
    left_margin = patch_size/2 -1
    right_margin = patch_size/2
    top_margin = patch_size/2 -1
    bottom_margin = patch_size/2

    #check if the first and last patch can be fit in origianl image dimesnion columnwise ============


  else:
    left_margin = patch_size/2
    right_margin = patch_size/2
    top_margin = patch_size/2
    bottom_margin = patch_size/2

  return left_margin, right_margin, top_margin, bottom_margin


def image_pad(original_image,  patch_size):

  #assert(original_image.shape[0] == mask_image.shape[0])
  #assert(original_image.shape[1] == mask_image.shape[1])

  H = original_image.shape[0]
  W = original_image.shape[1]
  C = original_image.shape[2]
  #rows, cols = np.where(mask_image > 0 )  # locations of valid FOV
  #sorted_rows = sorted(r for r,c in zip(rows, cols))
  #first_row = sorted_rows[0]
  #last_row = sorted_rows[-1]
  #sorted_cols = sorted(c for r,c in zip(rows, cols))
  #first_col = sorted_cols[0]
  #last_col = sorted_cols[-1]

  
  left_margin, right_margin, top_margin, bottom_margin = margin_size(patch_size)
  padded_image = np.zeros((H+top_margin+bottom_margin, W+left_margin+right_margin,C)) 
  H_p = padded_image.shape[0]
  W_p = padded_image.shape[1]
  padded_image[top_margin:H_p-bottom_margin, left_margin:W_p-right_margin,:] = original_image[:,:,:]
  return padded_image


def extract_test_patches(image, patch_size, ori_H, ori_W):
  
  left_margin, right_margin, top_margin, bottom_margin = margin_size(patch_size)
  H, W, C = image.shape
  
  
  n_patch_h = H - top_margin - bottom_margin  #number of patches along height
  n_patch_w = W - left_margin - right_margin  #number of patches along width
  assert(n_patch_h*n_patch_w == ori_H*ori_W)
  patch_holder = np.zeros((n_patch_h*n_patch_w, patch_size, patch_size,C))
  print "** Placeholder::", patch_holder.shape
  print "** L, R, T, B Margins are: ", left_margin, right_margin, top_margin, bottom_margin
  
  counter = 0
  for r in range(top_margin, H-bottom_margin):
    for c in range(left_margin, W-right_margin):
      
      patch_holder[counter,:,:,:] =image[r-top_margin:r+bottom_margin+1, c-left_margin:c+right_margin+1,:]  
      counter += 1
      #print counter
  #for h in range(n_patch_h):
  #  for w in range(n_patch_w):
  #    patch = image[top_margin + h*patch_size: top_margin+(h*patch_size)+patch_size,
  #                                         left_margin + w*patch_size: left_margin + (w*patch_size)+patch_size,:]
  #    patch_holder[counter, :,:,:] = patch
  #    print counter
  #    counter += 1 
  assert(counter == ori_H*ori_W)
  return patch_holder


def recompose(predictions, patch_size, padded_H, padded_W, ori_H, ori_W):

  #assert (predictions.shape[0] == ori_H * ori_W)  # validate number of patches
  recomposed_image = np.zeros((padded_H, padded_W))
  left_margin, right_margin, top_margin, bottom_margin = margin_size(patch_size)
  counter = 0
  #print "Inside recompose:", predictions.shape
  #time.sleep(5)
  for r in range(top_margin, padded_H-bottom_margin):
    for c in range(left_margin, padded_W-right_margin):
      #print "trying to recompose: ",r,c, predictions[counter]
      recomposed_image[r,c] = predictions[counter]
      counter += 1
  recomposed_image = np.reshape(recomposed_image,(recomposed_image.shape[0],recomposed_image.shape[1],1))
  return recomposed_image


def inside_FOV(r, c, mask_image):
    assert (len(mask_image.shape)==2)  #2D arrays
    #assert (mask_image.shape[2]==1)  #masks is binary
    

    if (r >= mask_image.shape[0] or c >= mask_image.shape[1]): #my image bigger than the original
        return False

    if (mask_image[r,c]>0):  #0==black pixels
        
        return True
    else:
        return False


#return only the pixels contained in the FOV, for both images and masks
def pred_only_FOV(data_imgs,data_masks,original_imgs_border_masks):
    assert (len(data_imgs.shape)==3 and len(data_masks.shape)==3)  #3D arrays
    assert (data_imgs.shape[0]==data_masks.shape[0])
    assert (data_imgs.shape[1]==data_masks.shape[1])
    #assert (data_imgs.shape[2]==data_masks.shape[2])
    assert (data_imgs.shape[2]==1 and data_masks.shape[2]==1)  #check the channel is 1
    height = data_imgs.shape[0]
    width = data_imgs.shape[1]
    new_pred_imgs = []
    new_pred_masks = []
    #for i in range(data_imgs.shape[0]):  #loop over the full images
    for y in range(height):
      for x in range(width):
        if inside_FOV(y,x,original_imgs_border_masks)==True:
          new_pred_imgs.append(data_imgs[y,x,:])
          new_pred_masks.append(data_masks[y,x,:])
    new_pred_imgs = np.asarray(new_pred_imgs)
    new_pred_masks = np.asarray(new_pred_masks)
    #new_pred_imgs = np.reshape(new_pred_imgs,(new_pred_imgs.shape[0]))
    #new_pred_masks = np.reshape(new_pred_masks,(new_pred_masks.shape[0]))
    return new_pred_imgs, new_pred_masks


#visualize image (as PIL image, NOT as matplotlib!)
def visualize(data,filename):
    assert (len(data.shape)==3) #height*width*channels
    img = None
    if data.shape[2]==1:  #in case it is black and white
        data = np.reshape(data,(data.shape[0],data.shape[1]))
    if np.max(data)>1:
        img = Image.fromarray(data.astype(np.uint8))   #the image is already 0-255
    else:
        img = Image.fromarray((data*255).astype(np.uint8))  #the image is between 0-1
    img.save(filename + '.png')
    return img
