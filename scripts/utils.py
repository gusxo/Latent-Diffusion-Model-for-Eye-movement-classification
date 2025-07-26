import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing
from scipy import signal
import glob
from numpy.typing import NDArray

def number_load(dir, filecnt=54):
    data = [None]*10
    t = np.zeros((10, filecnt), int)
    for p in range(10):
        d = [None]*filecnt
        for i in range(filecnt):
            d[i] = np.load(f"{dir}/{p}/{i+1}.npy")
            t[p,i] = d[i].shape[0]
        data[p] = d
    return data, t


def match_length(d, t:int):
    """
    INPUT : first return value of `number_load_all_data()`

    return
        x : (N, times(t), 2) shape numpy array,
        y : (N)
    """
    N = sum([len(d[i]) for i in range(len(d))])
    x = np.zeros((N, t, 2), np.float64)
    y = np.zeros((N), np.float64)
    target_timepoints = np.linspace(0, 1, t)
    start_at = 0
    for r in range(len(d)):
        for c in range(len(d[r])):
            origin_timepoints = np.linspace(0, 1, d[r][c].shape[0])
            x[start_at + c, :, 0] = np.interp(target_timepoints, origin_timepoints, d[r][c][:,0])
            x[start_at + c, :, 1] = np.interp(target_timepoints, origin_timepoints, d[r][c][:,1])
        y[start_at:start_at + len(d[r])] = r
        start_at += len(d[r])
    return x, y 

def apply_normalize(d, scale = 1.0, add = 0.0):
    for r in range(len(d)):
        for c in range(len(d[r])):
            channels = d[r][c].shape[1]
            min_vals = np.min(d[r][c][:, :], axis=0)
            max_vals = np.max(d[r][c][:, :], axis=0)
            min_max_diff = np.array([max_vals[j] - min_vals[j] for j in range(channels)])
            factor = min_max_diff / np.max(min_max_diff)
            for ch in range(channels):
                d[r][c][:, ch] = ((d[r][c][:, ch] - min_vals[ch])/min_max_diff[ch])*factor[ch]
                d[r][c][:, ch] = d[r][c][:, ch] * scale + add

#https://stackoverflow.com/questions/10252412/matplotlib-varying-color-of-line-to-capture-natural-time-parameterization-in-da/10253183#10253183
# #plot line with color through time
from matplotlib.collections import LineCollection
def draw_gradation(x, y, axes, cmap_name="jet", xlim=None, ylim=None):
    t = np.linspace(0, 1, x.shape[0])
    points = np.array([x, y]).transpose().reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1],points[1:]],axis=1)
    # make the collection of segments
    cmap = plt.get_cmap(cmap_name)
    lc = LineCollection(segs, cmap=plt.get_cmap(cmap_name))
    lc.set_array(t) # color the segments by our parameter

    # plot the collection
    axes.add_collection(lc) # add the collection to the plot
    if xlim is None:
      axes.set_xlim(x.min(), x.max()) # line collections don't auto-scale the plot
    else:
      axes.set_xlim(xlim[0], xlim[1])
    if ylim is None:
      axes.set_ylim(y.min(), y.max())
    else:
      axes.set_ylim(ylim[0], ylim[1])

def draw_timeseries_writingdata(datas, save=None, titles=None, cmap_name="jet_r", xlim=None, ylim=None, feature_name=["x", "y"], figblocksize=4, xyfiglength=4, suptitle=""):
    """
        matplotlib draw function for specific time-series data : eye-writing, hand-writing(signature), etc..
        
        datas : list of {data}. figure size is proportional to the length of {datas}.
                {data} : (times, 2) shape numpy array for drawing

        save : None or {file-path}. default None.
                if save = None, this function will ploting result.

                if save = {file-path}, don't ploting, plot-image save at {file-path}.

        titles : None or list of string. default None.
                if title = None, No titles above each data.

                if title = list of string, add titles above each data. list length must be equal {datas} length.

        cmap_name : name of matplotlib colormap to use. default is "jet_r".
                    choosing colormap at : https://matplotlib.org/stable/tutorials/colors/colormaps.html

        xlim : data's range of x. if None, set to min ~ max value.

        ylim : data's range of y. if None, set to min ~ max value.

        feature_name : feature(x, y)'s name. default is ['x', 'y'], feature name is used to result's label names.
    
    """
    plotlen = len(datas)
    mosaic_input = [None] * (plotlen << 1)
    for i in range(plotlen):
        j = i << 1
        mosaic_input[j] = [f"{i}a"] + [f"{i}x"] * xyfiglength
        mosaic_input[j+1] = [f"{i}a"] + [f"{i}y"] * xyfiglength
    fig, axes = plt.subplot_mosaic(mosaic_input, figsize=((xyfiglength + 1) * figblocksize, figblocksize * plotlen))
    for i in range(plotlen):
        if titles is not None:
            axes[f"{i}a"].set_title(titles[i])
        draw_gradation(datas[i][:,0], datas[i][:,1], axes[f"{i}a"], cmap_name, xlim, ylim)
        x_indexes = np.arange(datas[i].shape[0])
        draw_gradation(x_indexes, datas[i][:,0], axes[f"{i}x"], cmap_name, None, xlim)
        draw_gradation(x_indexes, datas[i][:,1], axes[f"{i}y"], cmap_name, None, ylim)

        #add labels
        fontsize = 14
        axes[f"{i}a"].set_xlabel(feature_name[0], fontsize=fontsize)
        axes[f"{i}a"].set_ylabel(feature_name[1], fontsize=fontsize)
        axes[f"{i}x"].set_ylabel(feature_name[0], fontsize=fontsize)
        axes[f"{i}y"].set_ylabel(feature_name[1], fontsize=fontsize)
        axes[f"{i}x"].set_xlabel("times", fontsize=fontsize)
        axes[f"{i}y"].set_xlabel("times", fontsize=fontsize)
    fig.suptitle(suptitle)

    plt.tight_layout()
    if save is None:
        plt.show()
    else:
        plt.savefig(save)
        plt.close()
    
def resample(x, original_hz, target_hz):
    time_len = x.shape[0]/original_hz
    target_len = int(time_len * target_hz)
    result = np.zeros((target_len, x.shape[1]), np.float32)
    for ch in range(x.shape[1]):
        result[:,ch] = signal.resample(x[:,ch], target_len)
    return result

def median_filter(x, kernel_size=7):
    med_filtered = np.zeros_like(x)
    for ch in range(med_filtered.shape[0]):
        med_filtered[ch,:] = np.median(x[max([0, ch - (kernel_size//2)]):ch+(kernel_size//2), :], axis=0)
    return med_filtered

def katakana_load_all_data(dir, split_by_subject=False, set_length=256, norm_range=(-1,1), label_is_zerobased=True, normalize_together=False, test_rate:float=0., seed=None):
    SUBJECTS = 6
    x = [None] * SUBJECTS
    y = [None] * SUBJECTS

    target_timepoints = np.linspace(0, 1, set_length)

    for sub_n in range(1, SUBJECTS+1):
        filenames = glob.glob(f"{dir}/{sub_n:03}/isolated_strokes/EOG_{sub_n:03}_??_???.csv")
        x[sub_n-1] = np.zeros((len(filenames), set_length, 2), np.float32)
        y[sub_n-1] = np.zeros((len(filenames)), np.float32)
        for i in range(len(filenames)):
            d = pd.read_csv(filenames[i], header=None)
            d = np.array(d, np.float32)
            d = resample(d, 1000, 64)
            d = median_filter(d)

            #change axis : rawdata's first axis is `y`, second axis is `-x`.
            d = d[:, ::-1]
            d[:, 0] *= -1

            origin_timepoints = np.linspace(0, 1, d.shape[0])
            x[sub_n-1][i, :, 0] = np.interp(target_timepoints, origin_timepoints, d[:, 0])
            x[sub_n-1][i, :, 1] = np.interp(target_timepoints, origin_timepoints, d[:, 1])
            y[sub_n-1][i] = int(filenames[i][-7:-4])
    
    if normalize_together:
        apply_normalize(x, norm_range[1]-norm_range[0], norm_range[0])
    else:
        for i in range(len(x)):
            x[i] = (x[i] - np.min(x[i], axis=1, keepdims=True))/(np.max(x[i], axis=1, keepdims=True) \
                                                                 - np.min(x[i], axis=1, keepdims=True)) * (norm_range[1]-norm_range[0]) + norm_range[0]

    if label_is_zerobased:
        for i in range(len(y)):
            y[i] -= 1
    
    if not split_by_subject:
        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)

    if test_rate > 0.:
        if not split_by_subject:
            return train_test_split(x, y, test_size=test_rate, seed=seed)
        else:
            assert int(len(x) * test_rate) > 0, "test rate is converted to 0. please increase value."
            test_subject = np.random.RandomState(seed).choice(len(x), replace=False, size=int(len(x) * test_rate))
            test_sub_mask = np.zeros(6, bool)
            for sub_n in test_subject:
                test_sub_mask[sub_n] = True
            train_sub_mask = ~test_sub_mask
            train_subject = np.where(train_sub_mask)[0]

            train_x = np.concatenate([x[i] for i in train_subject], axis=0)
            train_y = np.concatenate([y[i] for i in train_subject],)
            test_x = np.concatenate([x[i] for i in test_subject], axis=0)
            test_y = np.concatenate([y[i] for i in test_subject],)
            return train_x, test_x, train_y, test_y
        
    return x, y
        

def train_test_split(*x, axis:int = 0, test_size:typing.Union[int,float]=0.3, seed=None):
    """
    return : [x0_train, x0_test, x1_train, x1_test, ...]
    """


    assert axis >= 0 and axis < len(x[0].shape), "axis argument is must in range 0 < axis < len(x.shape)"
    for i in range(len(x) - 1):
        assert np.prod(x[i].shape[:axis+1]) == np.prod(x[i].shape[:axis+1]), "all of data's length is not matched"
    new_size = np.prod(x[0].shape[:axis+1])
    if isinstance(test_size, float):
        assert test_size > 0. and test_size < 1., "test_size(float) is must in range 0 < test_size < 1"
        test_size = int(new_size * test_size)
        assert test_size != 0, "test_size(float) is converted to 0. please increase test_size value."
    assert isinstance(test_size, int), "test_size is must int or float"
    
    test_index = np.random.RandomState(seed).choice(new_size, replace=False, size=test_size)
    test_mask = np.zeros(new_size, bool)
    test_mask[test_index] = True
    train_mask = ~test_mask

    result = []
    for d in x:
        #merge axis before target axis
        d = np.reshape(d, [*(new_size, *(d.shape[axis+1:]))])
        result.append(d[train_mask])
        result.append(d[test_mask])
    return result
    
def number_load_and_split(path:str, seq_len:int=256, val_person:int=9, seed:int=None):
    """
    return : train_x, train_y, test_x, test_y
    """

    #load data
    data, _ = number_load(path)
    apply_normalize(data, 2.0, -1.0) #normalize to [-1, 1](for ddpm training)
    x, y = match_length(data, seq_len)
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    PERSONS = 18
    
    #get mask to separate subjects
    subjects = np.zeros((PERSONS,30)).astype(int)
    for i in range(PERSONS):
        for p in range(10):
            begin = p*54 + i*3
            subjects[i, p*3:(p+1)*3] = [begin, begin+1, begin + 2]

    #split data
    seq = np.random.RandomState(seed).permutation(PERSONS)
    train_seq = np.concatenate([subjects[seq[i]] for i in range(val_person, PERSONS)])
    val_seq = np.concatenate([subjects[seq[i]] for i in range(0, val_person)])
    train_x = x[train_seq]
    train_y = y[train_seq]
    val_x = x[val_seq]
    val_y = y[val_seq]

    return train_x, train_y, val_x, val_y

def plot_data(x, suptitle:str="", cmap_name:str="jet_r", save:str=None, figblocksize:int=3, remove_tick=False, cols=4):
    """
    ploting eyewriting datas with row X `cols` format.

    parameters
    ----------
    `x` : (n, len, 2) shape's time-series numpy array

    `suptitle` : suptitle of total plot images

    `cmap_name` : colormap. see matplotlib's colormap.

    `save` : if given string, skip ploting and save as image.

    `figblocksize` : set plot size of each datas.

    `remove_tick` : if given `True`, remove boundary line of each data's image.

    `cols` : determine data count of each rows.
    """
    rows = int(np.ceil(x.shape[0]/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * figblocksize, rows * figblocksize), squeeze=False)
    
    for i in range(x.shape[0]):
        r = i//cols
        c = i%cols
        draw_gradation(x[i,:,0], x[i,:,1], axes[r][c], cmap_name, xlim=[-1,1], ylim=[-1,1])
        if remove_tick:
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
    
    fig.suptitle(suptitle)
    fig.tight_layout()
    if save:
        fig.savefig(save)
        fig.close()
    else:
        fig.show()

    return

def plot_PCA(*data:NDArray, axes:plt.Axes=None, figsize:typing.Tuple[int, int]=(4,4), label:typing.List[str]=None, title:str="PCA results", **kwargs):
    """
    ploting PCA results from given `data`.
    """
    from sklearn.decomposition import PCA
    if axes is None:
        _, axes = plt.subplots(1,1,figsize=figsize)
    
    pca = PCA(n_components=2)

    for i in range(len(data)):
        pca_result = pca.fit_transform(data[i].reshape(data[i].shape[0], -1))
        axes.scatter(pca_result[:,0], pca_result[:,1], **kwargs)
        
    axes.set_title(title)
    axes.set_xlabel("Principal Component 1")
    axes.set_ylabel("Principal Component 2")
    if label is not None:
        axes.legend(label)
    
    return

def plot_confusion_matrix(cm, class_names, save=None, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # 행렬 값을 표시
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save is None:
        plt.show()
    else:
        plt.savefig(save)