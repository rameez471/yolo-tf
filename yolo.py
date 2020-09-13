import sys 
import argparse
from detect import YOLO, detect_video
from PIL import Image

def detect_img(yolo):
    while True:
        img = input('Input image file: ')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again')
        else:
            r_image = yolo.detect_image(image)
            r_image.show()

FLAGS = None

if __name__ == '__main__':

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument(
        '--model_path',type=str,
        help='path to model weights'
    )

    parser.add_argument(
        '--anchors_path', type=str,
        help='path to anchors definaition'
    )

    parser.add_argument(
        '--class_path', type=str,
        help='path to class definaion, default '+ YOLO.get_defaults('classes_path')
    )

    parser.add_argument(
        '--image',default=False,action='store_true',
        help='Image detection mode'
    )

    parser.add_argument(
        '--input',nargs='?',type=str,required=False,
        help='Video inpur path'
    )

    parser.add_argument(
        '--output',nargs='?',type=str,
        help='[OPtional] Video output path'
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        print('Image detection mode')
        if 'input' in FLAGS:
            print('Ignoring command line arguments: '+FLAGS.input +','+FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif 'input' in FLAGS:
        detect_video(YOLO(**vars(FLAGS)),FLAGS.input,FLAGS.output)
    else:
        print('Must specify video path. See usage with --help')