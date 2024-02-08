from LivingMachines2023.utilities import load_data
import torch
# torch.set_num_threads(1)
import torch.nn as nn
from motion_vision_net import SNSMotionVisionMerged
import torch.autograd.profiler as profiler
import torch.utils.benchmark as benchmark
from timeit import default_timer


params_sns = load_data('LivingMachines2023/params_net_20230327.pc')

dtype = torch.float32
device = 'cpu'

params = nn.ParameterDict({
    'reversalEx': nn.Parameter(torch.tensor([5.0], dtype=dtype).to(device)),
    'reversalIn': nn.Parameter(torch.tensor([-2.0], dtype=dtype).to(device)),
    'reversalMod': nn.Parameter(torch.tensor([-0.1], dtype=dtype).to(device)),
    'freqFast': nn.Parameter(torch.tensor([params_sns['in']['cutoff']],dtype=dtype).to(device)),
    'kernelConductanceInBO': nn.Parameter(torch.tensor([params_sns['bp']['g']['center']], dtype=dtype).to(device)),
    'kernelReversalInBO': nn.Parameter(torch.tensor([-2.0], dtype=dtype).to(device)),
    'freqBOFast': nn.Parameter(torch.tensor([params_sns['bp']['cutoffHigh']],dtype=dtype).to(device)),
    'freqBOSlow': nn.Parameter(torch.tensor([params_sns['bp']['cutoffLow']],dtype=dtype).to(device)),
    'kernelConductanceInL': nn.Parameter(torch.tensor([params_sns['lp']['g']['center']], dtype=dtype).to(device)),
    'kernelReversalInL': nn.Parameter(torch.tensor([-2.0], dtype=dtype).to(device)),
    'freqL': nn.Parameter(torch.tensor([params_sns['in']['cutoff']],dtype=dtype).to(device)),
    'kernelConductanceInBF': nn.Parameter(torch.tensor([params_sns['bp']['g']['center']], dtype=dtype).to(device)),
    'kernelReversalInBF': nn.Parameter(torch.tensor([-2.0], dtype=dtype).to(device)),
    'freqBFFast': nn.Parameter(torch.tensor([params_sns['bp']['cutoffHigh']],dtype=dtype).to(device)),
    'freqBFSlow': nn.Parameter(torch.tensor([params_sns['bp']['cutoffLow']],dtype=dtype).to(device)),
    'conductanceLEO': nn.Parameter(torch.tensor([params_sns['e']['g']],dtype=dtype).to(device)),
    'freqEO': nn.Parameter(torch.tensor([params_sns['e']['cutoff']],dtype=dtype).to(device)),
    'conductanceBODO': nn.Parameter(torch.tensor([params_sns['d_on']['g']],dtype=dtype).to(device)),
    'freqDO': nn.Parameter(torch.tensor([params_sns['d_on']['cutoff']],dtype=dtype).to(device)),
    'conductanceDOSO': nn.Parameter(torch.tensor([params_sns['s_on']['g']],dtype=dtype).to(device)),
    'freqSO': nn.Parameter(torch.tensor([params_sns['s_on']['cutoff']],dtype=dtype).to(device)),
    'conductanceLEF': nn.Parameter(torch.tensor([params_sns['e']['g']],dtype=dtype).to(device)),
    'freqEF': nn.Parameter(torch.tensor([params_sns['e']['cutoff']],dtype=dtype).to(device)),
    'conductanceBFDF': nn.Parameter(torch.tensor([params_sns['d_off']['g']],dtype=dtype).to(device)),
    'freqDF': nn.Parameter(torch.tensor([params_sns['d_off']['cutoff']],dtype=dtype).to(device)),
    'conductanceDFSF': nn.Parameter(torch.tensor([params_sns['s_off']['g']],dtype=dtype).to(device)),
    'freqSF': nn.Parameter(torch.tensor([params_sns['s_off']['cutoff']],dtype=dtype).to(device)),
    'conductanceEOOn': nn.Parameter(torch.tensor([params_sns['on']['g']['enhance']],dtype=dtype).to(device)),
    'conductanceDOOn': nn.Parameter(torch.tensor([params_sns['on']['g']['direct']],dtype=dtype).to(device)),
    'conductanceSOOn': nn.Parameter(torch.tensor([params_sns['on']['g']['suppress']],dtype=dtype).to(device)),
    'conductanceEFOff': nn.Parameter(torch.tensor([params_sns['off']['g']['enhance']],dtype=dtype).to(device)),
    'conductanceDFOff': nn.Parameter(torch.tensor([params_sns['off']['g']['direct']],dtype=dtype).to(device)),
    'conductanceSFOff': nn.Parameter(torch.tensor([0.5],dtype=dtype).to(device)),
})

shape = [24,64]
with torch.no_grad():
    # torch.jit.enable_onednn_fusion(True)  # slower
    model_torch = SNSMotionVisionMerged(params_sns['dt'], shape, 1, params=params, dtype=dtype, device=device)
    model_torch.eval()
    model_torch = torch.jit.freeze(model_torch)
    model_torch = torch.compile(model_torch)
    # model_torch = torch.jit.optimize_for_inference(model_torch)   # slows down

    stim = torch.rand(shape,dtype=dtype, device=device)

    shape_emd = [x - 2 for x in shape]
    state_input = torch.zeros(shape, dtype=dtype).to(device)
    state_bp_on_input = torch.ones(shape, dtype=dtype).to(device)
    state_bp_on_fast = torch.zeros(shape, dtype=dtype).to(device)
    state_bp_on_slow = torch.zeros(shape, dtype=dtype).to(device)
    state_bp_on_output = torch.ones(shape, dtype=dtype).to(device)
    state_lowpass = torch.ones(shape, dtype=dtype).to(device)
    state_bp_off_input = torch.ones(shape, dtype=dtype).to(device)
    state_bp_off_fast = torch.zeros(shape, dtype=dtype).to(device)
    state_bp_off_slow = torch.zeros(shape, dtype=dtype).to(device)
    state_bp_off_output = torch.ones(shape, dtype=dtype).to(device)
    state_enhance_on = torch.ones(shape, dtype=dtype).to(device)
    state_direct_on = torch.zeros(shape, dtype=dtype).to(device)
    state_suppress_on = torch.zeros(shape, dtype=dtype).to(device)
    state_enhance_off = torch.ones(shape, dtype=dtype).to(device)
    state_direct_off = torch.zeros(shape, dtype=dtype).to(device)
    state_suppress_off = torch.zeros(shape, dtype=dtype).to(device)
    state_ccw_on = torch.zeros(shape_emd, dtype=dtype).to(device)
    state_cw_on = torch.zeros(shape_emd, dtype=dtype).to(device)
    state_ccw_off = torch.zeros(shape_emd, dtype=dtype).to(device)
    state_cw_off = torch.zeros(shape_emd, dtype=dtype).to(device)
    state_horizontal = torch.zeros(2, dtype=dtype).to(device)
    states = [state_input, state_bp_on_input, state_bp_on_fast, state_bp_on_slow, state_bp_on_output, state_lowpass,
              state_bp_off_input, state_bp_off_fast, state_bp_off_slow, state_bp_off_output, state_enhance_on,
              state_direct_on, state_suppress_on, state_enhance_off, state_direct_off, state_suppress_off, state_ccw_on,
              state_cw_on, state_ccw_off, state_cw_off, state_horizontal]

    # Warmup
    for i in range(5):
        model_torch(stim.to(device))

    num_samples = 1000
    num_threads = torch.get_num_threads()
    # num_threads = 1
    print(f'Benchmarking on {num_threads} threads')

    # with profiler.profile(with_stack=False, profile_memory=True) as prof:
    #     # states = model_torch(stim.to(device), states)
    #     states = model_torch(stim, states)
    #     output = states[20]#.to('cpu')
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by='cpu_time_total', row_limit=10))

    timer = benchmark.Timer(
        stmt='model_torch(x)',
        setup='from __main__ import model_torch',
        globals={'x': stim},
        num_threads=num_threads
    )

    print(timer.timeit(num_samples))
    times = torch.zeros(num_samples)
    for i in range(num_samples):
        start = default_timer()
        states = model_torch(stim)
        # output = states[20]#.to('cpu')
        end = default_timer()
        times[i] = (end-start)*1000
    print(torch.mean(times))
    print(torch.std(times))
    print(torch.var(times))