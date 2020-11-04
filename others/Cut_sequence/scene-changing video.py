import numpy as np
import os

def y_import(video_path, height_frame, width_frame, nfs, startfrm):
    """Import Y channel from a yuv video.

    startfrm: start from 0
    return: (nfs * height * width), dtype=uint8."""

    d0 = height_frame // 2
    d1 = width_frame // 2
    y_size = height_frame * width_frame
    u_size = d0 * d1
    v_size = u_size

    fp = open(video_path,'rb')

    # target at startfrm
    blk_size = y_size + u_size + v_size
    fp.seek(blk_size * startfrm, 0)

    # extract
    y_batch = []
    for ite_frame in range(nfs):
        y_frame = [ord(fp.read(1)) for k in range(y_size)]
        y_frame = np.array(y_frame, dtype=np.uint8).reshape((height_frame, width_frame))
        fp.read(u_size + v_size)  # skip u and v
        y_batch.append(y_frame)
    fp.close()
    y_batch = np.array(y_batch)
    return y_batch


def write_yuv(video_path, height_frame, width_frame, frame,new_video_path,count):

    d0 = height_frame // 2
    d1 = width_frame // 2
    y_size = height_frame * width_frame
    u_size = d0 * d1
    v_size = u_size

    blk_size = (y_size + u_size + v_size)*frame

    with open(video_path,'rb') as fp:
        data = fp.read(blk_size)
        with open(new_video_path,'ab+') as fp_new:
            fp_new.seek(2)
            fp_new.write(data)




walk_file = 'C:/Users/PC/Desktop/scene-changingg video/rawvideo'
video_files = os.listdir(walk_file)#当前文件夹下的所有文件

count=0
for video_file in video_files:
    video_path = walk_file+'/'+video_file

    height = 240
    width = 416

    frame = 15

    new_video_path = 'C:/Users/PC/Desktop/scene-changingg video/D_video_15_new.yuv'

    write_yuv(video_path,height,width,frame,new_video_path,count)
    count=count+1

