from blur import Blur
from deblur import Deblur
from noise import Noise
from denoise import Denoise
from get_input import get_input

DEBLUR = 'deblur'
DENOISE = 'denoise'
MY_NETS = [DEBLUR, DENOISE]


def select_model():

    net_options = list('to ' + net_name + ' press ' + str(i) for i, net_name in enumerate(MY_NETS))
    message = 'Please select the requested net:\n'
    net_name = get_input(message, net_options, MY_NETS)

    if net_name == DEBLUR:
        deblur = Deblur()
        blur = Blur()
        return deblur, blur, net_name
    else:
        denoise = Denoise()
        noise = Noise()
        return denoise, noise, net_name
