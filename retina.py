from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.networks import Network
from utilities import cutoff_fastest, add_lowpass_filter

def create_retina(net: Network):
    add_lowpass_filter(net, cutoff=cutoff_fastest, name='Retina')