import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import swin_base_patch4_window7_224 as create_model
import pandas as pd


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = 512
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img1_path = "dataset/test/test_images/"
    l_path = os.listdir(img1_path)
    for path in l_path:
        img_path = img1_path + path
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        plt.imshow(img)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        # read class_indict
        json_path = 'class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        json_file = open(json_path, "r")
        class_indict = json.load(json_file)

        # create model
        model = create_model(num_classes=2).to(device)
        # load model weights
        model_weight_path = "./weights/model-24.pth"
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        with open('./24pre.txt', 'a') as f:
            print("name:{:10}   class: {:10}   prob: {:.2}".format(path,class_indict[str(1)],
                                                  predict[1].numpy()),file=f)
        dict1['name'].append(path)
        dict1['class'].append(class_indict[str(1)])
        dict1['prob'].append(predict[1].numpy())
        #
        # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
        #                                              predict[predict_cla].numpy())
        # plt.title(print_res)
        # for i in range(len(predict)):
        #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
        #                                               predict[i].numpy()))
        # plt.show()
    # writer = pd.ExcelWriter('/home/lyf/SwinTransformer/swin_transformer/ATEC.xlsx')
    # data = pd.DataFrame(dict1)
    # data.to_excel(writer, sheet_name='sheet1')
    # writer._save()

if __name__ == '__main__':
    dict1 = {'name': [],'class':[],'prob':[]}
    main()
