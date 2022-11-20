import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

start_time = 86930.00
end_time = 86934.00
df = pd.read_csv('Data.csv', delimiter=',')


fig, ax = plt.subplots(1,1)
plt.subplots_adjust(bottom=0.25)
sax = plt.axes([0.25, 0.1, 0.65, 0.03])
slide = Slider(sax, 'time', start_time, end_time, valinit=start_time)

#Initial plot
act_data = df.loc[df['timestamp'] == float(int(start_time))]
s, = ax.plot(act_data.x, act_data.y, 'o')

def update(timestamp):
    act_data = df.loc[df['timestamp'] == float(int(timestamp))]
    X = act_data.x
    Y = act_data.y

    #Update data based on slider
    s.set_xdata(X)
    s.set_ydata(Y)

    #Reset axis limits
    ax.set_xlim([X.min()*0.9,X.max()*1.1])
    ax.set_ylim([Y.min()*0.9,Y.max()*1.1])

    fig.canvas.draw()

slide.on_changed(update)
plt.show()