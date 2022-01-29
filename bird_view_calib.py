import numpy as np
from numpy.linalg import inv
import math
import time
import argparse
import yaml

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#AUTHOR: JHONATHAN CASTAÑEDA - UNIVERSIDAD AUTÒNOMA DE OCCIDENTE - 01/01/2022 
#jhonathan.castaneda@uao.edu.co

#FIRST TEST, REVISED AND WORKING == 01/01/2022

#IMPORTANT!! THIS CODE IS CURRENTLY WORKING TO GENERATE AN IDEAL VIRTUAL IMAGE OF 1000 x 1000 PIXELS -- THIS CODE WILL BE UPDATED THEN.
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#FOR SQUARE BIRD VIEW-------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='initial parameters')

    parser.add_argument('-s', '--side_virtual',              type=int, default=1000,  help='bird visible área, lenght for each side (mm)')
    parser.add_argument('-f', '--fov_virtual_cam',           type=int, default=80,    help='fov of the virtual camera in degrees')
    parser.add_argument('-m', '--m_virtual_pixels',          type=int, default=1000,  help='#of virtual pixels in u direction')
    parser.add_argument('-n', '--n_virtual_pixels',          type=int, default=1000,  help='#of virtual pixels in v direction')
    
    parser.add_argument('-fc1','--fish_cam1',          type=str,      default='yaml_config/left_ir.yaml',            help='fish_cam1 config(.yaml) file path')
    parser.add_argument('-lm', '--landmarks_path',     type=str,      default='yaml_config/bird_calib_points.yaml',  help='registered landmarks(.yaml) file path')
    parser.add_argument('-cp', '--calib_path',        type=str,       default='yaml_config/b_calib_out.yaml',        help='out. path for calibration data(.yaml) file')
    parser.add_argument('-vc', '--v_cam_path',         type=str,      default='yaml_config/virtual_cam.yaml',        help='out. path for virtual cam info(.yaml) file')

    
    args = parser.parse_args()
    print(args)  
    
def find_kc_yaml(fc1_path):
    
    with open(fc1_path) as t:  
    
        fc1_data = yaml.load(t, Loader=yaml.FullLoader) 
        
    kc_sub_dic     = fc1_data.get("camera_matrix")      
    kc_data_list   = kc_sub_dic.get("data")             

    fax=kc_data_list[0];fay=kc_data_list[4];ua0=kc_data_list[2];va0=kc_data_list[5] 
    fc1_kc=np.array([[fax, 0, ua0],[0, fay, va0],[0,0,1]])                         
    fc1_kc_inv = inv(fc1_kc)                                                      
    
    return fc1_kc, fc1_kc_inv

def set_virtual_params(fov,s,m,n,v_cam_path):
   
    f=math.radians(90-(fov/2))                                                 
    h=math.tan(f)*(s/2);  fax=(m/s)*h;  fay=(n/s)*h; ua0=(1/2)*m; va0=(1/2)*n  
    Kc=np.array([[fax, 0, ua0],[0, fay, va0],[0,0,1]])                          
    Kc_inv = inv(Kc)                                                            
    h_vir=h                                                                     
    
    
    data2exp = { 'virtual_cam_mtx': {'rows': 3, 'cols': 3,'data': Kc.tolist()} , 
                'virtual_cam_mtx_inv': {'rows': 3, 'cols': 3,'data': Kc_inv.tolist()} ,
                'h_vir': h_vir, 'fov':f,'side_length': s, 'm_pixels': m, 'n_pixels': n}    
    
    
    with open(v_cam_path, 'w') as p:
        yaml.dump(data2exp, p,sort_keys=False)
        
    print ('\n'+ 'VIRTUAL CAM MTX SUCCESSFULLY STORED AT:')
    print (v_cam_path)   
    print('\n'+'---------------------')
      
    return Kc, Kc_inv, h_vir

def pix2cor(px, h, kc_inv): 

    cor_1   =  np.matmul( (h* kc_inv),px )
    
    return cor_1


def import_landmarks(lm_path, h_vir, vc_kc_inv):
    
    calib_points = lm_path
    h_vir        = h_vir 
    vc_kc_inv    = vc_kc_inv

    with open(calib_points) as c:                              

        points_data = yaml.load(c, Loader=yaml.FullLoader)     

    fish_points = points_data.get("fish_undistorted_points")   
    virtual_points = points_data.get("virtual_points")         

    f_points_list   = fish_points.get("data") 
    v_points_list   = virtual_points.get("data") 

    x1_v=v_points_list[0]; y1_v=v_points_list[4]       
    x2_v=v_points_list[1]; y2_v=v_points_list[5]
    x3_v=v_points_list[2]; y3_v=v_points_list[6]
    x4_v=v_points_list[3]; y4_v=v_points_list[7]


    v_points  =  np.array([ [ x1_v, x2_v, x3_v, x4_v ] , [ y1_v, y2_v, y3_v, y4_v ], [1,1,1,1] ]  )   
    v_cor     =  np.array([ [], [], [] ])    

    for i in range(4):    

        points = v_points[0:,i]                                               
        points_vec = np.array([ [points[0]] , [points[1]], [points[2]] ]  )  
        cord   = pix2cor(points_vec, h_vir, vc_kc_inv)                        

        if i==0:              
            v_cor = cord

        elif i>0:             

            v_cor= np.append(v_cor,cord,1)    

    x1_f=f_points_list[0]; y1_f=f_points_list[4]       
    x2_f=f_points_list[1]; y2_f=f_points_list[5]
    x3_f=f_points_list[2]; y3_f=f_points_list[6]
    x4_f=f_points_list[3]; y4_f=f_points_list[7]
    
    f_points  =  np.array([ [ x1_f, x2_f, x3_f, x4_f ] , [ y1_f, y2_f, y3_f, y4_f ], [1,1,1,1] ]  )  

    return(v_points, v_cor, f_points)


def get_rt_mtx(fc1_kc, v_cor, f_points):
    
    fc1_kc     =  fc1_kc    
    v_cor      =  v_cor 
    f_points   =  f_points

    fax_f = fc1_kc[0,0];   fay_f = fc1_kc[1,1];   ua0_f = fc1_kc[0,2];   va0_f = fc1_kc[1,2]    

    v11=v_cor[0,0]; v21=v_cor[1,0]; v31=v_cor[2,0]   
    v12=v_cor[0,1]; v22=v_cor[1,1]; v32=v_cor[2,1]   
    v13=v_cor[0,2]; v23=v_cor[1,2]; v33=v_cor[2,2]
    v14=v_cor[0,3]; v24=v_cor[1,3]; v34=v_cor[2,3]   

    a1 = f_points[0,0];     a3 = f_points[0,1];     a5 = f_points[0,2];     a7 = f_points[0,3]
    a2 = f_points[1,0];     a4 = f_points[1,1];     a6 = f_points[1,2];     a8 = f_points[1,3]


    ext =  np.array([ [fax_f*v11,fax_f*v21,fax_f*v31,0,0,0,(ua0_f-a1)*v11,(ua0_f-a1)*v21],  
                      [0,0,0,fay_f*v11,fay_f*v21,fay_f*v31,(va0_f-a2)*v11,(va0_f-a2)*v21],
                      [fax_f*v12,fax_f*v22,fax_f*v32,0,0,0,(ua0_f-a3)*v12,(ua0_f-a3)*v22],
                      [0,0,0,fay_f*v12,fay_f*v22,fay_f*v32,(va0_f-a4)*v12,(va0_f-a4)*v22],
                      [fax_f*v13,fax_f*v23,fax_f*v33,0,0,0,(ua0_f-a5)*v13,(ua0_f-a5)*v23],
                      [0,0,0,fay_f*v13,fay_f*v23,fay_f*v33,(va0_f-a6)*v13,(va0_f-a6)*v23],
                      [fax_f*v14,fax_f*v24,fax_f*v34,0,0,0,(ua0_f-a7)*v14,(ua0_f-a7)*v24],
                      [0,0,0,fay_f*v14,fay_f*v24,fay_f*v34,(va0_f-a8)*v14,(va0_f-a8)*v24]   ]) 

    ext_inv = inv(ext)                                                                     

    ext_mtx_p=np.array([[(a1-ua0_f)*v31],[(a2-va0_f)*v31],  [(a3-ua0_f)*v31],[(a4-va0_f)*v31],  [(a5-ua0_f)*v31],[(a6-va0_f)*v31], [(a7-ua0_f)*v31],[(a8-va0_f)*v31]])
    
    rt_pms=np.matmul(ext_inv, ext_mtx_p)  

    r11=rt_pms[0,0];   r12=rt_pms[1,0];   r13=rt_pms[2,0];   r21=rt_pms[3,0];   r22=rt_pms[4,0];   r23=rt_pms[5,0];   r31=rt_pms[6,0];   r32=rt_pms[7,0]  

    rt=np.array([[r11,r12,r13],[r21,r22,r23],[r31,r32,1]])
 
    rt_inv=inv(rt)   


    return(rt, rt_inv)

def export_mtx(calib_path, rt, rt_inv):

    
    data2exp = { 'rt_matrix': {'rows': 3, 'cols': 3,'data': rt.tolist()} , 'rt_inverse_matrix': {'rows': 3, 'cols': 3,'data': rt_inv.tolist()}   }

    matrix = rt
    with open(calib_path, 'w') as f:
        yaml.dump(data2exp, f,sort_keys=False)
    
    print ('\n'+ 'CALIBRATION RESULTS SUCCESSFULLY STORED AT:')
    print (calib_path)   
    print('\n'+'---------------------')
    
#---------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------          
#MAIN SEQUENCE--------------------------------------------------------------------------------------------------------------------------------------------

fc1_path     = args.fish_cam1          
lm_path      = args.landmarks_path    
calib_path  = args.calib_path        
v_cam_path   = args.v_cam_path           

fc1_kc, fc1_kc_inv            =  find_kc_yaml(fc1_path)  

vc_kc, vc_kc_inv, h_vir =  set_virtual_params(args.fov_virtual_cam, args.side_virtual, args.m_virtual_pixels, args.n_virtual_pixels, v_cam_path) 

v_points, v_cor, f_points  =  import_landmarks(lm_path, h_vir, vc_kc_inv)


t = time.time()
rt, rt_inv = get_rt_mtx(fc1_kc, v_cor, f_points) 

elapsed = time.time() - t
print('\n'+'CALIBRATION COMPLETED'+'\n'+'---------------------'+'\n'+'TIME TO EXTRACT EXTERNAL ROTATION-TRANSLATION PARAMETERS:'+'\n'+ str(elapsed)+' '+'s' +'\n'+'---------------------' )
print ('RT ESTIMATED MATRIX:'+'\n')
print(rt)
print('\n'+'---------------------')

export_mtx(calib_path, rt, rt_inv)
