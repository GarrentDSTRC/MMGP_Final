import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import torch
device = torch.device("cpu")
from matplotlib import pyplot as plt
import pandas as pd
Frame2 = pd.read_excel('.\ROM\BF_search.xlsx', sheet_name="HL")
from GPy import findpoint_interpolate
"-----------------------PLOT FUNCTION-----------------------------------------------"
def plot_interplate(model, likelihood):
    i = np.linspace(0.1, 0.3, 7)
    x = np.linspace(0.1, 0.6, 6)
    y = np.linspace(5, 70, 14)
    z = np.linspace(0, 180, 19)
    I, X, Y, Z = np.meshgrid(i, x, y, z)
    I = I.flatten()
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    pointX = []
    for j in range(len(X)):
        px = np.array([I[j], X[j], Y[j], Z[j]])
        pointX.append(px)

    model.eval()
    likelihood.eval()
    pointX=np.array(pointX)
    K=torch.tensor(pointX).to(device).to(torch.float32)
    segment=np.linspace(0,len(X),10).astype(int)
    Y=np.array([])
    for i in range(9):
        A = likelihood(model(   K[segment[i]:segment[i+1],:]   )).mean.cpu().detach().numpy()
        Y=np.concatenate((Y,A))
    Y=np.expand_dims(Y,axis=1)
    Test=np.concatenate((pointX,Y),axis=1)
    np.savetxt("test.csv", Test, delimiter=',')
def plot3D(model, likelihood,num_task=1,scale=[1]*18):
    "num_task<0    ->use the multi-fidelity kernel"
    #num_task=1 Single GP; num_task=0 Raw;num_task=2 Multitask;num_task=-2  and multifidelity ;num_task=-1 OR num_task=1 MultiKernel=1 multifidelity
    #Multikernel means multifidelity kernel;
    #Raw means just plot the raw data
    model.eval()
    likelihood.eval()

    if num_task==0:
        Raw=1
        num_task=2
    else:
        Raw=0

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #for i in [0.1, 0.15, 0.2, 0.25, 0.4]:
        value = []
        for i in [0.1]:
            x = np.linspace(0.1, 0.6, 6)
            y = np.linspace(5, 40, 8)
            z = np.linspace(0, 180, 7)
            X, Y, Z = np.meshgrid(x, y, z)
            X = X.flatten()
            Y = Y.flatten()
            Z = Z.flatten()

            pointx = []
            pointy = []
            for j in range(len(X)):
                px, py = findpoint_interpolate(np.array([i, X[j], Y[j], Z[j]]), Frame2,num_task,"nearest")
                # if np.isnan(py):
                #     px, py = findpoint_interpolate(np.array([i, X[j], Y[j], Z[j]]), Frame2, num_task,"nearest")

                pointx.append(px)
                pointy.append(py)
            pointX = np.asarray(pointx)
            values = np.asarray(pointy).T.squeeze(np.abs(num_task)-1)


            if np.abs(num_task)==1:
                value.append(values)

                if num_task<0:
                    values=likelihood(model( torch.tensor(pointX).to(torch.float32), torch.full((pointX.shape[0], 1), dtype=torch.long, fill_value=0))).mean
                    values0=likelihood(model( torch.tensor(pointX).to(torch.float32), torch.full((pointX.shape[0], 1), dtype=torch.long, fill_value=1))).mean
                    value.append(values0)
                else: values = likelihood(model(torch.tensor(pointX).to(torch.float32).to(device))).mean
                value.append(values)

            else:
                values=values.T
                value.append(values[:,0])
                value.append(values[:,1])

                if num_task==-2:
                    values = likelihood(*model(
                        (torch.tensor(pointX).to(torch.float32),
                         torch.full((pointX.shape[0], 1), dtype=torch.long, fill_value=0)),
                        (torch.tensor(pointX).to(torch.float32),
                         torch.full((pointX.shape[0], 1), dtype=torch.long, fill_value=0))
                    ))
                    value.append(values[0].mean)
                    value.append(values[1].mean)
                    values = likelihood(*model(
                        (torch.tensor(pointX).to(torch.float32),
                         torch.full((pointX.shape[0], 1), dtype=torch.long, fill_value=1)),
                        (torch.tensor(pointX).to(torch.float32),
                         torch.full((pointX.shape[0], 1), dtype=torch.long, fill_value=1))
                    ))

                    value.append(values[0].mean)
                    value.append(values[1].mean)
                    print("realx2,lowx2,highx2")
                elif num_task==2 and Raw==0:#independent multitask
                    values = likelihood(*model(
                        torch.tensor(pointX).to(torch.float32).to(device),
                        torch.tensor(pointX).to(torch.float32).to(device),

                    ))
                    value.append(values[0].mean.detach().cpu().numpy())
                    value.append(values[1].mean.detach().cpu().numpy())
                # elif Raw==0:
                #     value.append(likelihood(model(torch.tensor(pointX).to(torch.float32).to(device))).mean[:,0])
                #     value.append(likelihood(model(torch.tensor(pointX).to(torch.float32).to(device))).mean[:,1])

    # S = np.array([value[0], value[1].cpu().detach().numpy()]).T
    # Test=np.concatenate((pointX,S),axis=1)
    # np.savetxt("test.csv", Test, delimiter=',')
    scale2=[2,1.5,2,1.5,2,1.5,3,3,3,3,3,3]
    #scale2 = [1,1,3,3]
    for p in range(len(value)):
        fig = go.Figure(data=go.Isosurface(
            x=X,
            y=Y,
            z=Z,
            value=(value[p] * scale[p]).tolist(),
            isomin=-2 * scale[p] * scale2[p],
            # isomin=min(values)
            isomax=2 * scale[p] * scale2[p],
            # surface_fill=0.7,
            # opacity=0.9,  # 改变图形的透明度
            colorscale='jet',  # 改变颜色

            surface_count=5,
            colorbar_nticks=7,
            caps=dict(x_show=False, y_show=False, z_show=False),

            # slices_z = dict(show=True, locations=[-1, -9, -5]),
            # slices_y = dict(show=True, locations=[20]),

            # surface=dict(count=3, fill=0.7, pattern='odd'),  # pattern取值：'all', 'odd', 'even'
            # caps=dict(x_show=True, y_show=True),
            # surface_pattern = "even"
        ))
        fig.update_scenes(yaxis=dict(title=r'θ', tickfont=dict(size=13), titlefont=dict(size=18)))
        fig.update_scenes(yaxis_nticks=5)
        fig.update_scenes(xaxis_nticks=4)
        fig.update_scenes(xaxis_range=list([0, 0.7]))
        # fig.update_scenes(zaxis_nticks=3)
        fig.update_scenes(xaxis=dict(title=r'y', tickfont=dict(size=13), titlefont=dict(size=18)))
        fig.update_scenes(zaxis=dict(title='ψ', tickfont=dict(size=13), titlefont=dict(size=18)))
        fig.update_coloraxes(colorbar_tickfont_size=20)
        fig.update_layout(
            height=500,
            width=500,
        )
        fig.show()
        #pio.write_image(fig, f'3D_True_{x[0,0]}_{j}eposide.png')


def ax_plot(f, ax, y_labels, title):
    im = ax.imshow(y_labels)
    ax.set_title(title)
    return im


def plot_2d(observed_pred_y1, observed_pred_y2, test_y_actual1, test_y_actual2, delta_y1, delta_y2):
    # Plot our predictive means
    f, observed_ax = plt.subplots(2, 3, figsize=(4, 3))
    ax_plot(f, observed_ax[0, 0], observed_pred_y1, 'observed_pred_y1 (Likelihood)')

    # Plot the true values
    # f, observed_ax2 = plt.subplots(1, 1, figsize=(4, 3))
    ax_plot(f, observed_ax[1, 0], observed_pred_y2, 'observed_pred_y2 (Likelihood)')

    # Plot the absolute errors
    ax_plot(f, observed_ax[0, 1], test_y_actual1, 'test_y_actual1')

    # Plot the absolute errors
    ax_plot(f, observed_ax[1, 1], test_y_actual2, 'test_y_actual2')

    # Plot the absolute errors
    ax_plot(f, observed_ax[0, 2], delta_y1, 'Absolute Error Surface1')

    # Plot the absolute errors
    im = ax_plot(f, observed_ax[1, 2], delta_y2, 'Absolute Error Surface2')

    cb_ax = f.add_axes([0.9, 0.1, 0.02, 0.8])  # 设置colarbar位置
    cbar = f.colorbar(im, cax=cb_ax)  # 共享colorbar

    plt.show()