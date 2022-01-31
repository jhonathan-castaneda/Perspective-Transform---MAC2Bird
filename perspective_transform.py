import numpy as np
from numpy.linalg import inv
import math
import time
import argparse
import yaml
import cv2
import os

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#AUTHOR: JHONATHAN CASTAÑEDA - UNIVERSIDAD AUTÒNOMA DE OCCIDENTE - 01/27/2022 
#jhonathan.castaneda@uao.edu.co

#SCRIPT BASED ON: 
# Z. Yang, Y. Zhao, X. Hu, Y. Yin, L. Zhou and D. Tao, "A flexible vehicle surround view camera system by central-around coordinate mapping 
# model," Multimedia tools and applications. 2019.

#FIRST TEST, REVISED AND WORKING == 01/27/2022

#IMPORTANT!! THIS CODE IS CURRENTLY WORKING TO GENERATE AN IDEAL VIRTUAL IMAGE OF 1000 x 1000 PIXELS -- THIS CODE WILL BE UPDATED THEN.
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='CAM_1_PARAMS')

    parser.add_argument('-rt',  '--rt_matrix',     type=str,   default='yaml_config/b_calib_out.yaml',    help='transformation matrix(.yaml) file path')
    parser.add_argument('-fc1', '--fish_cam1',     type=str,   default='yaml_config/left_ir.yaml',        help='fish_cam1 config(.yaml) file path')
    parser.add_argument('-vc1', '--virtual_cam1',  type=str,   default='yaml_config/virtual_cam.yaml',    help='virtual_cam1 config(.yaml) file path')


    parser.add_argument('-im',  '--image_source',  type=str,      default='IMAGE_SOURCE/u_frame.jpg',     help='source of undistorted images(path)')
    parser.add_argument('-out', '--output_path',   type=str,      default='REMAPPED/',      help='exportation path for remapped images')

    
    args = parser.parse_args()
    print(args)  

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    
    if iteration == total: 
        print()
        
def remap_data_loader(rt_path, fc1_path, vc1_path): 

    rt = rt_path  
    with open(rt) as a:                              

        data = yaml.load(a, Loader=yaml.FullLoader)  

    rt_mtx_info     = data.get("rt_matrix")                 
    rt_inv_mtx_info = data.get("rt_inverse_matrix")  

    rt_mtx     = np.array(rt_mtx_info.get("data") )            
    rt_inv_mtx = np.array(rt_inv_mtx_info.get("data") )             
    
    
    fc1 =fc1_path
    with open(fc1) as b:  
    
        fc1 = yaml.load(b, Loader=yaml.FullLoader) 
        
    kc_sub_dic     = fc1.get("camera_matrix")      
    kc_data_list   = kc_sub_dic.get("data")         

    fax=kc_data_list[0];fay=kc_data_list[4];ua0=kc_data_list[2];va0=kc_data_list[5] 
    fc1_kc=np.array([[fax, 0, ua0],[0, fay, va0],[0,0,1]])                          
    fc1_kc_inv = inv(fc1_kc)                                                       
  
    vc1 = vc1_path
    with open(vc1) as c:  
    
        vc1 = yaml.load(c, Loader=yaml.FullLoader) 
        
    kcv_dic     = vc1.get("virtual_cam_mtx") 
    kcv_data    = kcv_dic.get("data")     

    inv_kcv_dic     = vc1.get("virtual_cam_mtx_inv") 
    inv_kcv_data    = inv_kcv_dic.get("data")      
    
    
    vc1_kc      =  np.array(kcv_data) 
    vc1_kc_inv  =  np.array(inv_kcv_data)
    
    h_vir       = vc1.get('h_vir')
    fov         = vc1.get('fov')
    side_length = vc1.get('side_length')
    m_pixels    = vc1.get('m_pixels')
    n_pixels    = vc1.get('n_pixels')
     
        
    return (rt_mtx, rt_inv_mtx, fc1_kc, fc1_kc_inv, vc1_kc, vc1_kc_inv, h_vir, fov, side_length, m_pixels, n_pixels )


def pix2cor(px, h, kc_inv): 
    cor_1   =  np.matmul( (h*kc_inv), px )   
    return cor_1


def cor2pix(cor, h, kc):
    pix   =    np.matmul(( (1/h)*kc ), cor )
    return pix

def find_s_cor(v_cor, rt):    
    s_cor   =  np.matmul(rt,v_cor)    
    return s_cor

#---------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------          
#MAIN SEQUENCE--------------------------------------------------------------------------------------------------------------------------------------------

rt_path  = args.rt_matrix  
fc1_path = args.fish_cam1 
vc1_path = args.virtual_cam1 
in_img_path = args.image_source
out_path  = args.output_path

print('CALIBRATION DATA READY')
print('\n')
rt_mtx, rt_inv_mtx, fc1_kc, fc1_kc_inv, vc1_kc, vc1_kc_inv, h_vir, fov, side_length, m_pixels, n_pixels  = remap_data_loader(rt_path, fc1_path, vc1_path)
print('------------------RT-------------------')
print(rt_mtx)
print('------------------RT_INV---------------')
print(rt_inv_mtx)
print('------------------FC1------------------')
print(fc1_kc)
print('------------------FC1_INV--------------')
print(fc1_kc_inv)
print('------------------VC1------------------')
print(vc1_kc)
print('------------------VC1_INC--------------')
print(vc1_kc_inv)
print('---------------------------------------')

img=cv2.imread(in_img_path, cv2.IMREAD_GRAYSCALE)   
v_img_rgb = np.zeros((1000,1000,3), np.uint8)        
v_img = cv2.cvtColor(v_img_rgb, cv2.COLOR_BGR2GRAY) 

height=500
width=1000

l = height*width

print('INITIALIZING TRANSFORMATION')
print('---------------------------')
print('PIXELS TO REMAP: '+str(l))
print('\n')

printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
i=0 

t = time.time()

for u in range(width):
    for v in range(height):
        
        v_pixel=np.array([[u],[v],[1]])
        
        v_cordinate = pix2cor(v_pixel, h_vir, vc1_kc_inv)
        
        s_cordinate = find_s_cor(v_cordinate, rt_mtx) 
         
        H_VAR=s_cordinate[2] 

        s_pix= cor2pix(s_cordinate, H_VAR, fc1_kc)             
        
        s_pix_int=np.array([[s_pix[0]],[s_pix[1]]],np.uint16) 
       
        v_img[v,u] = img[s_pix_int[1],s_pix_int[0]] 
                
        printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
        i=i+1
    
        
out_img_path = os.path.join(out_path,'Vimage.jpg')        
cv2.imwrite(out_img_path, v_img)

elapsed = time.time() - t
print('\n'+'PROJECTION COMPLETED'+'\n'+'---------------------'+'\n'+'TIME TO REMAP CURRENT FRAME:'+'\n'+ str(elapsed)+' '+'s' +'\n'+'---------------------' )
