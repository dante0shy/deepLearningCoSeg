# from search.image_form import ImageForm
from search.image_form import ImageForm
from search.single_demo import co_seg
from django.http import HttpResponse
from django.shortcuts import render, redirect, get_object_or_404
import logging
import cv2
import base64
import json
import os
import numpy as np
import hashlib

BASE = os.path.dirname(__file__)


def search(request):
    logging.debug("search")
    # return HttpResponse('success')
    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)

        logging.warn(form.is_valid())
        if form.is_valid():
            try:
                image_1 = form.cleaned_data["image_1"].read()
                image_2 = form.cleaned_data["image_2"].read()
                image_1 = np.fromstring(image_1, np.uint8)
                image_1 = cv2.imdecode(image_1, -1)
                image_2 = np.fromstring(image_2, np.uint8)
                image_2 = cv2.imdecode(image_2, -1)

            except Exception as e:
                print(e)
                return render(request, "index.html")
            logging.info("get infos")
            try:
                # print(image)
                tmp_img_1 = cv2.resize(image_1, (512, 512))
                cv2.imwrite(
                    os.path.join(
                        BASE,
                        "..",
                        "media",
                        hashlib.sha1(tmp_img_1.tostring()).hexdigest() + ".jpg",
                    ),
                    tmp_img_1,
                )
                tmp_img_1 = hashlib.sha1(tmp_img_1.tostring()).hexdigest() + ".jpg"
                tmp_img_2 = cv2.resize(image_2, (512, 512))
                cv2.imwrite(
                    os.path.join(
                        BASE,
                        "..",
                        "media",
                        hashlib.sha1(tmp_img_2.tostring()).hexdigest() + ".jpg",
                    ),
                    tmp_img_2,
                )
                tmp_img_2 = hashlib.sha1(tmp_img_2.tostring()).hexdigest() + ".jpg"
            except Exception as e:
                print(e)
                return render(request, "index.html")

            logging.info("send")
            try:
                cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
                response = co_seg.single_demo(
                    cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB),
                    cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB),
                )
                assert response
                logging.warn("send success")
            except Exception as e:
                print(e)
                return render(request, "index.html")

            if response:
                res = [
                    {
                        "o": "/media/" + tmp_img_1,
                        "p": "/media/" + response[0].split("/")[-1],
                    },
                    {
                        "o": "/media/" + tmp_img_2,
                        "p": "/media/" + response[1].split("/")[-1],
                    },
                ]
            else:
                res = []

            print(res)
            return render(
                request,
                "index.html",
                {
                    # 'img_o1': '/media/' + tmp_img_1,
                    # 'img_o2': '/media/' + tmp_img_2,
                    "result": res
                },
            )
        else:
            image = None
            return render(request, "index.html")
    else:
        return render(request, "index.html")


def index(request):
    return render(request, "index.html")
