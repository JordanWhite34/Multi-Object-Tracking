from numpy import genfromtxt
import glob
import cv2
import os
import numpy as np
from argparse import ArgumentParser
import imageio

def get_args():
    parser = ArgumentParser(add_help=False, usage=usageMsg())
    parser.add_argument("data", nargs=1, help="Path to <tracking_result>.")
    parser.add_argument('-ds', '--dstype', type=str, default='test', help="Data set type: validation or test.")
    parser.add_argument('--limit', type=int, default=1000, help="number of frames generated")
    return parser.parse_args()

def usageMsg():
    return """  python3 eval.py """

def usage(msg=None):
    """ Print usage information, including an optional message, and exit. """
    if msg:
        print("%s\n" % msg)
    print("\nUsage: %s" % usageMsg())
    exit()

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color   

if __name__ == '__main__':
    args = get_args()
    if not args.data or len(args.data) < 1:
        usage("Incorrect number of arguments. Must provide paths for the test (ground truth) and predicitons.")
    result = np.loadtxt(args.data[0], delimiter = ',')

    if args.dstype == 'test':
        camera = 75
    elif args.dstype == 'validation':
        camera = 72
    else:
        raise NotImplementedError('dstype need to be either validation or test, you use {}'.format(args.dstype))

    imgs = sorted(glob.glob('c0{}/img/*'.format(camera)))
  
    os.makedirs('{}_visualization'.format(args.dstype),exist_ok=True)

    print('Saving visualization result at {}_visualization'.format(args.dstype))

    images_for_gif = []

    for frame,img in enumerate(imgs):
        
        frame_result = result[result[:,2] == frame]

        if frame%100 == 0:
            print('processing frame: ',frame)

        if frame == args.limit:
            break

        im_out = cv2.imread(img)

        cv2.putText(im_out,str(frame),(30,100),cv2.FONT_HERSHEY_PLAIN, 5,(0,0,255),thickness = 3)
                    
        for _,tracking_id,_,x,y,w,h,_,_ in frame_result:
            cv2.putText(im_out,str(int(tracking_id)),(int(x),int(y)),cv2.FONT_HERSHEY_PLAIN, 3,(0,255,255),thickness = 3)
            cv2.rectangle(im_out,(int(x),int(y)),(int(x+w),int(y+h)),get_color(tracking_id),2)

        cv2.imwrite('{}_visualization/{}.jpg'.format(args.dstype,"%06d"%frame),im_out)
        frame += 1

        if frame > args.limit - 500:
            images_for_gif.append(im_out)

    imageio.mimsave('{}_visualization/result.gif'.format(args.dstype), images_for_gif, duration=40)
