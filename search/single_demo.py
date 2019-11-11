import torch, os
import PIL.Image as Image
from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Resize
from torchvision.transforms import ToTensor, ToPILImage
from search.model39_278 import *
import numpy as np
from torchvision.utils import save_image
import random

gpu_ids = [0]
if "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_path = os.path.abspath(
    os.path.join(
        os.path.join(os.path.dirname(__file__), "model", "epoch1iter12000.pkl")
    )
)
output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "media")


class Demo:
    def __init__(self):
        self.net = model().cuda()
        self.net = nn.DataParallel(self.net, device_ids=gpu_ids)
        self.net.load_state_dict(torch.load(model_path))

        self.input_transform = Compose(
            [
                Resize((512, 512)),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def single_demo(self, image1, image2):
        self.net.eval()
        image1 = self.input_transform(Image.fromarray(image1))
        image2 = self.input_transform(Image.fromarray(image2))
        image1, image2 = image1.unsqueeze(0).cuda(), image2.unsqueeze(0).cuda()
        try:
            output1, output2 = self.net(image1, image2)
        except:
            return 0
        output1 = torch.argmax(output1, dim=1)
        output2 = torch.argmax(output2, dim=1)

        image1 = (image1 - image1.min()) / image1.max()
        image2 = (image2 - image2.min()) / image2.max()

        output1 = torch.cat(
            [
                torch.zeros(1, 512, 512).long().cuda(),
                output1,
                torch.zeros(1, 512, 512).long().cuda(),
            ]
        ).unsqueeze(0)
        output2 = torch.cat(
            [
                torch.zeros(1, 512, 512).long().cuda(),
                output2,
                torch.zeros(1, 512, 512).long().cuda(),
            ]
        ).unsqueeze(0)
        # return output1,output2
        seed = random.randint(0, 10000000)
        save_image(
            output1.float().data * 0.8 + image1.data,
            os.path.join(output_path, "1-{}.jpg".format(seed)),
            normalize=True,
        )
        save_image(
            output2.float().data * 0.8 + image2.data,
            os.path.join(output_path, "2-{}.jpg".format(seed)),
            normalize=True,
        )
        return (
            os.path.join(output_path, "1-{}.jpg".format(seed)),
            os.path.join(output_path, "2-{}.jpg".format(seed)),
        )


co_seg = Demo()
if __name__ == "__main__":
    import cv2

    im1 = cv2.cvtColor(
        cv2.imread("/home/dante0shy/PycharmProjects/dl_class_projecty/test/t1.jpg"),
        cv2.COLOR_BGR2RGB,
    )
    im2 = cv2.cvtColor(
        cv2.imread("/home/dante0shy/PycharmProjects/dl_class_projecty/test/t2.jpg"),
        cv2.COLOR_BGR2RGB,
    )
    demo = Demo()
    demo.single_demo(im1, im2)

    print("Finish!!!")
