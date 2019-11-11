
from django import forms


class ImageForm(forms.Form):
    image_1 = forms.ImageField(required=False)
    image_2 = forms.ImageField(required=False)
    # wayPick = forms.IntegerField()
    # image_url = forms.CharField()