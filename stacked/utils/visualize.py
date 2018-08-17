from matplotlib import pyplot as plt
import os


def plot_kernels(kernel_dict, max_rows, max_cols, save_path):
    fig = plt.figure(figsize=(len(kernel_dict), max_rows))
    sorted_kernels = sorted(kernel_dict.items())
    col = 0

    for num_kernels, tensors in sorted_kernels:
        for tensor in tensors:
            k = col * max_rows + 1
            col += 1
            tensor = tensor.data.numpy()
            for i in range(num_kernels):
                ax1 = fig.add_subplot(max_cols, max_rows, k)
                k += 1
                ax1.imshow(tensor[i])
                ax1.axis('off')
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig('%s.png' % save_path, bbox_inches="tight")
    plt.show()


def plot_model(model, save_path):
    # also possible to get the average
    # shape_dict = get_shape_dict(model)
    # average_dict = get_average_dict(shape_dict)
    weights = model.state_dict()
    kernels = {}
    max_rows = -1
    max_cols = 0

    for k in weights:
        if 'conv' in k and 'weight' in k:
            v = weights[k]
            size = v.size()
            if size[-1] != 3 or len(size) != 4:
                continue
            max_cols += 1
            num_kernels = size[0]
            if num_kernels not in kernels:
                kernels[num_kernels] = []
            kernels[num_kernels].append(v)
            if max_rows < num_kernels:
                max_rows = num_kernels

    plot_kernels(kernels, max_rows, max_cols, save_path)