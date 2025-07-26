import numpy as np

"""
below augmentation functions are copied from https://github.com/uchidalab/time_series_augmentation/blob/master/utils/augmentation.py
"""
def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def window_slice(x, reduce_ratio=0.9):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i,:,dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i],dim]).T
    return ret

def window_warp(x, window_ratio=0.1, scales=[0.5, 2.]):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)
        
    window_starts = np.random.randint(low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)
            
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i],dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
            end_seg = pat[window_ends[i]:,dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))                
            ret[i,:,dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped).T
    return ret

def batch_generator(*data, batch_size, augmentation=[], cycle=True, random_apply=False, return_tensor=False):
    """
    iterable batch generator

    data :
        datas for batch generator.(x, y, ...)
        
    batch_size :
        batch size

    augmentation : 
        OPTIONAL, augmentation function's array.

        data's first array (x) is randomly applied this functions.

        Does not apply to second or later arrays.

    cycle : 
        OPTIONAL, default is True.

        if is not True, generator is stopped when iterating one cycle on data.

    random_apply :
        OPTIONAL

        it works only when `augmentation` parameters is given.

        if is `True`, apply augmentation functions randomly.
    """

    assert isinstance(augmentation, list), "augmentation parameter is must be list of callables."
    assert batch_size <= len(data[0]), f"batch size is {batch_size}, it is larger than data length({len(data[0])})"

    d = [None] * len(data)
    for i in range(len(data)):
        d[i] = np.asarray(data[i])

    batch_cnt = d[0].shape[0] // batch_size
    select_func = [True] * len(augmentation) if len(augmentation) else []

    if return_tensor:
        import tensorflow as tf

    while 1:
        seq = np.random.permutation(d[0].shape[0])
        if random_apply:
            select_func = np.random.choice([0, 1], len(augmentation)) if len(augmentation) else []
        for begin in range(0, batch_cnt * batch_size, batch_size):
            x = [d[i][seq[begin:begin+batch_size]] for i in range(len(data))]
            for func, use in zip(augmentation, select_func):
                if use:
                    x[0] = func(x[0])
            if return_tensor:
                for i in range(len(x)):
                    x[i] = tf.convert_to_tensor(x[i])

            yield x
        if not cycle:
            break